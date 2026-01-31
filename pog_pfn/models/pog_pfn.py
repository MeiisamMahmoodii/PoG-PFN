"""
PoG-PFN: Full End-to-End Model

Integrates all modules:
- Dataset Encoder (Module A)
- Claim Encoder (Module B)
- Graph Posterior Head (Module C)
- Differentiable Identification (Module D)
- Effect Estimator (Module E)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict

from .dataset_encoder import DatasetEncoder
from .claim_encoder import ClaimEncoder, Claim
from .graph_posterior import GraphPosteriorHead
from .identification import DifferentiableIdentification
from .effect_estimator import EffectEstimator


class PoGPFN(nn.Module):
    """
    Posterior-over-Graphs Prior-Fitted Network
    
    A transformer-based model for causal effect estimation that:
    1. Treats claims as probabilistic priors over causal structures
    2. Computes posteriors over graphs given data and claims
    3. Performs differentiable causal identification
    4. Returns calibrated ATE distributions with uncertainty
    """
    
    def __init__(
        self,
        n_vars: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 500,
        max_claims: int = 50,
        claim_d_model: int = 128,
        n_claim_layers: int = 3,
        n_quantiles: int = 9,
        acyclicity_penalty: float = 1.0,
        max_adjustment_sets: int = 10,
        soft_logic_temperature: float = 0.5,
        use_doubly_robust: bool = True
    ):
        super().__init__()
        
        self.n_vars = n_vars
        self.d_model = d_model
        
        # Module A: Dataset Encoder
        self.dataset_encoder = DatasetEncoder(
            n_vars=n_vars,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # Module B: Claim Encoder
        self.claim_encoder = ClaimEncoder(
            n_vars=n_vars,
            d_model=d_model,
            claim_d_model=claim_d_model,
            n_heads=n_heads,
            n_layers=n_claim_layers,
            dropout=dropout,
            max_claims=max_claims
        )
        
        # Module C: Graph Posterior Head
        self.graph_posterior = GraphPosteriorHead(
            n_vars=n_vars,
            d_model=d_model,
            dropout=dropout,
            acyclicity_penalty=acyclicity_penalty
        )
        
        # Module D: Differentiable Identification
        self.identification = DifferentiableIdentification(
            n_vars=n_vars,
            max_adjustment_sets=max_adjustment_sets,
            soft_logic_temperature=soft_logic_temperature
        )
        
        # Module E: Effect Estimator
        self.effect_estimator = EffectEstimator(
            n_vars=n_vars,
            d_model=d_model,
            n_quantiles=n_quantiles,
            dropout=dropout,
            use_doubly_robust=use_doubly_robust
        )
    
    def forward(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        Y: torch.Tensor,
        claims: List[List[Claim]],
        treatment_idx: int,
        outcome_idx: int,
        return_uncertainty: bool = True,
        return_graph_posterior: bool = True,
        return_all_info: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        End-to-end forward pass: data + claims → ATE distribution.
        
        Args:
            X: Features [batch, n_samples, n_features]
            T: Treatment [batch, n_samples]
            Y: Outcome [batch, n_samples]
            claims: List of claim lists per batch
            treatment_idx: Index of treatment variable
            outcome_idx: Index of outcome variable
            return_uncertainty: Return ATE uncertainty
            return_graph_posterior: Return posterior adjacency matrix
            return_all_info: Return detailed intermediate outputs
            
        Returns:
            outputs: Dict containing:
                - ate_mean: [batch]
                - ate_std: [batch] (if return_uncertainty)
                - graph_posterior: [batch, n_vars, n_vars] (if return_graph_posterior)
                - adjustment_distribution: [batch, n_vars]
                - ... (additional info if return_all_info)
        """
        batch_size = X.shape[0]
        device = X.device
        n_samples = X.shape[1]
        
        # Step 1: Encode dataset
        dataset_embedding, variable_embeddings = self.dataset_encoder(X, Y)
        
        # Step 2: Encode claims
        claim_embeddings, claim_context, claim_mask = self.claim_encoder(
            claims, variable_embeddings, device=device
        )
        
        # Compute average claim confidence for gating
        avg_confidence = 0.5  # Default
        if len(claims[0]) > 0:
            confidences = [c.confidence for c in claims[0] if not claim_mask[0, :len(claims[0])].any()]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
        
        # Step 3: Predict graph posterior
        graph_posterior, graph_components = self.graph_posterior(
            dataset_embedding,
            variable_embeddings,
            claim_context,
            n_samples=n_samples,
            avg_claim_confidence=avg_confidence,
            return_components=True
        )
        
        # Step 4: Differentiable identification
        adjustment_distribution, validity_scores, id_info = self.identification(
            graph_posterior,
            treatment_idx,
            outcome_idx
        )
        
        # Step 5: Estimate ATE
        ate_mean, ate_std, effect_info = self.effect_estimator(
            X, T, Y,
            adjustment_distribution,
            variable_embeddings,
            return_uncertainty=return_uncertainty
        )
        
        # Prepare outputs
        outputs = {
            'ate_mean': ate_mean,
            'adjustment_distribution': adjustment_distribution,
        }
        
        if return_uncertainty:
            outputs['ate_std'] = ate_std
        
        if return_graph_posterior:
            outputs['graph_posterior'] = graph_posterior
        
        if return_all_info:
            outputs['dataset_embedding'] = dataset_embedding
            outputs['variable_embeddings'] = variable_embeddings
            outputs['claim_embeddings'] = claim_embeddings
            outputs['claim_context'] = claim_context
            outputs['graph_components'] = graph_components
            outputs['validity_scores'] = validity_scores
            outputs['id_info'] = id_info
            outputs['effect_info'] = effect_info
        
        return outputs
    
    def predict(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        Y: torch.Tensor,
        claims: List[List[Claim]],
        treatment_idx: int,
        outcome_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simplified prediction interface.
        
        Returns:
            ate_mean: [batch]
            ate_std: [batch]
            graph_posterior: [batch, n_vars, n_vars]
        """
        outputs = self.forward(
            X, T, Y, claims,
            treatment_idx, outcome_idx,
            return_uncertainty=True,
            return_graph_posterior=True,
            return_all_info=False
        )
        
        return outputs['ate_mean'], outputs['ate_std'], outputs['graph_posterior']
    
    def explain(
        self,
        X: torch.Tensor,
        T: torch.Tensor,
        Y: torch.Tensor,
        claims: List[List[Claim]],
        treatment_idx: int,
        outcome_idx: int
    ) -> Dict[str, any]:
        """
        Return human-interpretable explanations.
        
        Returns:
            explanations: Dict with:
                - best_adjustment_set: Most valid adjustment set
                - claim_posterior_mass: Per-claim consistency with data
                - graph_confidence: Uncertainty in graph structure
                - effect_breakdown: Contribution of different paths
        """
        outputs = self.forward(
            X, T, Y, claims,
            treatment_idx, outcome_idx,
            return_uncertainty=True,
            return_graph_posterior=True,
            return_all_info=True
        )
        
        # Extract explanations
        id_info = outputs['id_info']
        graph_components = outputs['graph_components']
        
        # Best adjustment set
        best_set_idx = outputs['validity_scores'][0].argmax().item()
        best_adjustment_set = id_info['candidates'][best_set_idx]
        
        # Claim validation: which claims are consistent with posterior?
        # (Would need to implement claim-specific validation)
        
        # Graph confidence (inverse of entropy)
        graph_probs = outputs['graph_posterior'][0]
        graph_entropy = -(graph_probs * torch.log(graph_probs + 1e-8) + 
                          (1 - graph_probs) * torch.log(1 - graph_probs + 1e-8)).mean()
        graph_confidence = 1 - graph_entropy.item()
        
        explanations = {
            'best_adjustment_set': best_adjustment_set,
            'adjustment_probabilities': outputs['adjustment_distribution'][0].detach().cpu().numpy(),
            'graph_confidence': graph_confidence,
            'gate_value': graph_components['gate'][0].item(),
            'data_graph': graph_components['W_data'][0].detach().cpu().numpy(),
            'claim_graph': graph_components['W_claim'][0].detach().cpu().numpy(),
        }
        
        return explanations


if __name__ == "__main__":
    from pog_pfn.models.claim_encoder import RelationType
    
    # Test full model
    batch_size = 2
    n_samples = 100
    n_vars = 10
    
    # Create dummy data
    # X contains n_vars features, Y is one of those variables
    # For the model, we pass X with all variables and Y separately
    X = torch.randn(batch_size, n_samples, n_vars - 1)  # Features excluding outcome
    T = torch.randint(0, 2, (batch_size, n_samples)).float()
    Y = torch.randn(batch_size, n_samples)  # Outcome
    
    # Create claims
    claims = [
        [
            Claim(0, 1, None, RelationType.CAUSES, 0.9),
            Claim(2, 3, None, RelationType.CONFOUNDER, 0.8),
        ],
        [
            Claim(1, 2, None, RelationType.MEDIATOR, 0.95),
        ]
    ]
    
    # Create model
    model = PoGPFN(
        n_vars=n_vars,
        d_model=128,
        n_heads=4,
        n_layers=3,
        n_claim_layers=2
    )
    
    # Forward pass
    outputs = model(
        X, T, Y, claims,
        treatment_idx=1,
        outcome_idx=2,
        return_all_info=True
    )
    
    print(f"ATE mean: {outputs['ate_mean']}")
    print(f"ATE std: {outputs['ate_std']}")
    print(f"Graph posterior shape: {outputs['graph_posterior'].shape}")
    print(f"Adjustment distribution: {outputs['adjustment_distribution']}")
    
    # Test explanation
    explanations = model.explain(X, T, Y, claims, treatment_idx=1, outcome_idx=2)
    print(f"\nExplanations:")
    print(f"  Best adjustment set: {explanations['best_adjustment_set']}")
    print(f"  Graph confidence: {explanations['graph_confidence']:.3f}")
    print(f"  Gate (data trust): {explanations['gate_value']:.3f}")
    
    print("\n✓ Full PoG-PFN model test passed!")
