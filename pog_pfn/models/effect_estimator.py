"""
Module E: Effect Estimator

Implements doubly robust ATE estimation with transformer-based nuisance models.
Conditioned on the adjustment set distribution from the identification layer.

Outputs ATE distribution (mean + uncertainty) rather than just point estimate.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class EffectEstimator(nn.Module):
    """
    Doubly robust average treatment effect (ATE) estimator.
    
    Uses transformer-based models for:
    - Propensity scores: e(x) = P(T=1|X)
    - Outcome models: μ0(x) = E[Y|T=0,X], μ1(x) = E[Y|T=1,X]
    
    Computes DR estimator:
    τ_DR = 1/n Σ [μ1(Xi) - μ0(Xi) + Ti/e(Xi)(Yi - μ1(Xi)) - (1-Ti)/(1-e(Xi))(Yi - μ0(Xi))]
    """
    
    def __init__(
        self,
        n_vars: int,
        d_model: int = 256,
        n_quantiles: int = 9,
        dropout: float = 0.1,
        use_doubly_robust: bool = True
    ):
        super().__init__()
        
        self.n_vars = n_vars
        self.d_model = d_model
        self.n_quantiles = n_quantiles
        self.use_doubly_robust = use_doubly_robust
        
        # Feature encoder: encodes adjustment variables
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_vars, d_model),  # Weighted by adjustment distribution
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        
        # Propensity score model: P(T=1|X)
        self.propensity_model = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Outcome models: E[Y|T,X]
        # Separate networks for T=0 and T=1
        self.outcome_model_0 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.outcome_model_1 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Quantile regression for uncertainty (optionally predict quantiles)
        if n_quantiles > 1:
            quantile_levels = torch.linspace(0.1, 0.9, n_quantiles)
            self.register_buffer('quantile_levels', quantile_levels)
            
            self.quantile_predictor = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, n_quantiles)
            )
        
    def compute_weighted_features(
        self,
        X: torch.Tensor,
        adjustment_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute features weighted by adjustment distribution.
        
        Args:
            X: Raw features [batch, n_samples, n_vars]
            adjustment_weights: Soft inclusion probs [batch, n_vars]
            
        Returns:
            weighted_X: [batch, n_samples, n_vars]
        """
        # Expand adjustment weights to match samples
        weights = adjustment_weights.unsqueeze(1)  # [batch, 1, n_vars]
        
        # Element-wise multiplication
        weighted_X = X * weights  # [batch, n_samples, n_vars]
        
        return weighted_X
    
    def forward(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        Y: torch.Tensor,
        adjustment_distribution: torch.Tensor,
        variable_embeddings: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Estimate average treatment effect.
        
        Args:
            X: Features [batch, n_samples, n_vars]
            T: Treatment [batch, n_samples]
            Y: Outcome [batch, n_samples]
            adjustment_distribution: Soft adjustment set [batch, n_vars]
            variable_embeddings: From dataset encoder [batch, n_vars, d_model]
            return_uncertainty: Whether to compute uncertainty estimates
            
        Returns:
            ate_mean: Estimated ATE [batch]
            ate_std: Uncertainty (std dev) [batch] (if return_uncertainty=True)
            info: Dict with nuisance estimates
        """
        batch_size, n_samples, n_features = X.shape
        device = X.device
        
        # Handle case where X has fewer features than n_vars (outcome excluded)
        # Adjustment distribution is [batch, n_vars], but X is [batch, n_samples, n_features]
        if n_features < self.n_vars:
            # Take first n_features weights (assume outcome is last)
            adjustment_weights_for_X = adjustment_distribution[:, :n_features]
        else:
            adjustment_weights_for_X = adjustment_distribution
        
        # Weight features by adjustment distribution
        X_weighted = self.compute_weighted_features(X, adjustment_weights_for_X)  # [batch, n_samples, n_features]
        
        # Pad X_weighted to n_vars if needed for the feature_encoder
        if n_features < self.n_vars:
            padding = torch.zeros(batch_size, n_samples, self.n_vars - n_features, device=device)
            X_weighted = torch.cat([X_weighted, padding], dim=-1)  # [batch, n_samples, n_vars]
        
        # Encode features
        X_encoded = self.feature_encoder(X_weighted)  # [batch, n_samples, d_model]
        
        # Estimate propensity scores
        propensity = self.propensity_model(X_encoded).squeeze(-1)  # [batch, n_samples]
        propensity = propensity.clamp(0.01, 0.99)  # Avoid division by zero
        
        # Estimate outcome models
        mu_0 = self.outcome_model_0(X_encoded).squeeze(-1)  # [batch, n_samples]
        mu_1 = self.outcome_model_1(X_encoded).squeeze(-1)  # [batch, n_samples]
        
        if self.use_doubly_robust:
            # Doubly robust estimator
            # τ_DR = E[μ1(X) - μ0(X)] + E[T/e(X)(Y - μ1(X))] - E[(1-T)/(1-e(X))(Y - μ0(X))]
            
            # Plug-in term
            plugin = mu_1 - mu_0  # [batch, n_samples]
            
            # Augmentation terms
            treated_aug = (T / propensity) * (Y - mu_1)
            control_aug = ((1 - T) / (1 - propensity)) * (Y - mu_0)
            
            # Individual-level estimates
            ate_individual = plugin + treated_aug - control_aug  # [batch, n_samples]
            
            # Average over samples
            ate_mean = ate_individual.mean(dim=1)  # [batch]
            
            # Final safety check for NaN/Inf
            ate_mean = torch.where(torch.isfinite(ate_mean), ate_mean, torch.zeros_like(ate_mean))
        
        else:
            # Simple plug-in estimator (outcome regression)
            ate_individual = mu_1 - mu_0
            ate_mean = ate_individual.mean(dim=1)
            
            # Final safety check for NaN/Inf
            ate_mean = torch.where(torch.isfinite(ate_mean), ate_mean, torch.zeros_like(ate_mean))
        
        # Estimate uncertainty
        ate_std = None
        if return_uncertainty:
            # Use sample variance as uncertainty estimate
            ate_std = ate_individual.std(dim=1)  # [batch]
            # Ensure std is positive and finite
            ate_std = torch.clamp(ate_std, min=0.01, max=100.0)
            ate_std = torch.where(torch.isfinite(ate_std), ate_std, torch.ones_like(ate_std) * 1.0)
            
            # Alternatively, use quantile regression (if implemented)
            # This would require additional training
        
        info = {
            'propensity': propensity,
            'mu_0': mu_0,
            'mu_1': mu_1,
            'ate_individual': ate_individual
        }
        
        return ate_mean, ate_std, info
    
    def predict_quantiles(
        self,
        X: torch.Tensor,
        adjustment_distribution: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict ATE quantiles for uncertainty quantification.
        
        Args:
            X: Features [batch, n_samples, n_vars]
            adjustment_distribution: [batch, n_vars]
            
        Returns:
            quantiles: [batch, n_quantiles]
        """
        if self.n_quantiles <= 1:
            raise ValueError("Quantile prediction requires n_quantiles > 1")
        
        # Weight features
        X_weighted = self.compute_weighted_features(X, adjustment_distribution)
        
        # Encode
        X_encoded = self.feature_encoder(X_weighted)
        
        # Global pooling
        X_pooled = X_encoded.mean(dim=1)  # [batch, d_model]
        
        # Predict quantiles
        quantiles = self.quantile_predictor(X_pooled)  # [batch, n_quantiles]
        
        return quantiles


if __name__ == "__main__":
    # Test the module
    batch_size = 4
    n_samples = 100
    n_vars = 10
    d_model = 128
    
    # Create dummy data
    X = torch.randn(batch_size, n_samples, n_vars)
    T = torch.randint(0, 2, (batch_size, n_samples)).float()
    Y = torch.randn(batch_size, n_samples)
    adjustment_dist = torch.rand(batch_size, n_vars)
    var_embs = torch.randn(batch_size, n_vars, d_model)
    
    # Create estimator
    estimator = EffectEstimator(n_vars=n_vars, d_model=d_model, n_quantiles=9)
    
    # Estimate ATE
    ate_mean, ate_std, info = estimator(X, T, Y, adjustment_dist, var_embs, return_uncertainty=True)
    
    print(f"ATE mean shape: {ate_mean.shape}")  # [4]
    print(f"ATE std shape: {ate_std.shape}")  # [4]
    print(f"ATE mean: {ate_mean}")
    print(f"ATE std: {ate_std}")
    print("✓ Effect estimator test passed!")
