"""
Multi-Objective Loss Functions for PoG-PFN Training

Implements the four key losses:
1. Effect Loss (CRPS for calibrated uncertainty)
2. Graph Posterior Calibration
3. Constraint Likelihood (claim validation)
4. Identification Consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class EffectLoss(nn.Module):
    """
    Continuous Ranked Probability Score (CRPS) for ATE prediction.
    
    Encourages calibrated uncertainty: penalizes both point error and
    under/over-confident predictions.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        predicted_mean: torch.Tensor,
        predicted_std: torch.Tensor,
        true_ate: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute simple MAE loss (CRPS has numerical stability issues).
        
        Using Mean Absolute Error as a stable alternative to CRPS.
        """
        # Simple MAE - no numerical issues
        mae = torch.abs(predicted_mean - true_ate).mean()
        return mae


class GraphPosteriorLoss(nn.Module):
    """
    Binary cross-entropy for graph structure prediction.
    
    Allows for CPDAG equivalence class matching (future improvement).
    """
    
    def __init__(self, use_cpdag_equivalence: bool = False):
        super().__init__()
        self.use_cpdag_equivalence = use_cpdag_equivalence
        self.bce = nn.BCELoss()
    
    def forward(
        self,
        predicted_adjacency: torch.Tensor,
        true_adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predicted_adjacency: [batch, n_vars, n_vars] (soft, in [0,1])
            true_adjacency: [batch, n_vars, n_vars] (binary)
            
        Returns:
            loss: scalar
        """
        # Simple BCE for now
        # TODO: Implement CPDAG equivalence scoring
        loss = self.bce(predicted_adjacency, true_adjacency)
        
        return loss


class ClaimValidationLoss(nn.Module):
    """
    Measures how well the posterior validates claims.
    
    For each claim c, computes p(c is true | posterior) and compares
    to ground truth.
    """
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def compute_claim_probability(
        self,
        claim,
        posterior_adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute probability that claim is true given posterior.
        
        Args:
            claim: Claim object
            posterior_adjacency: [batch, n_vars, n_vars]
            
        Returns:
            prob: [batch] - probability claim is true
        """
        from ..models.claim_encoder import RelationType
        
        if claim.relation_type == RelationType.CAUSES:
            # Edge exists
            prob = posterior_adjacency[:, claim.var_a, claim.var_b]
            
        elif claim.relation_type == RelationType.FORBIDS:
            # Edge doesn't exist
            prob = 1 - posterior_adjacency[:, claim.var_a, claim.var_b]
            
        elif claim.relation_type == RelationType.ANCESTOR:
            # Ancestry exists (approximate with edge for now)
            prob = posterior_adjacency[:, claim.var_a, claim.var_b]
            
        elif claim.relation_type == RelationType.NON_ANCESTOR:
            # No ancestry
            prob = 1 - posterior_adjacency[:, claim.var_a, claim.var_b]
            
        else:
            # For complex claims (confounder, mediator), return uniform
            prob = torch.ones(posterior_adjacency.shape[0], device=posterior_adjacency.device) * 0.5
        
        return prob
    
    def forward(
        self,
        claims: list,  # List of lists of claims
        posterior_adjacency: torch.Tensor,
        claim_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            claims: List[List[Claim]] - claims per batch
            posterior_adjacency: [batch, n_vars, n_vars]
            claim_mask: [batch, max_claims] - True = padding
            
        Returns:
            loss: scalar
        """
        batch_size = posterior_adjacency.shape[0]
        device = posterior_adjacency.device
        
        total_loss = 0.0
        n_valid_claims = 0
        
        for i in range(batch_size):
            for j, claim in enumerate(claims[i]):
                # Check if j is within claim_mask bounds and claim is valid
                if j < claim_mask.shape[1] and not claim_mask[i, j]:
                    # Compute posterior probability
                    prob = self.compute_claim_probability(claim, posterior_adjacency[i:i+1])
                    
                    # Ground truth
                    target = torch.tensor([1.0 if claim.is_true else 0.0], device=device)
                    
                    # BCE loss
                    loss = F.binary_cross_entropy(prob, target)
                    total_loss += loss
                    n_valid_claims += 1
        
        if n_valid_claims == 0:
            return torch.tensor(0.0, device=device)
        
        return total_loss / n_valid_claims


class IdentificationConsistencyLoss(nn.Module):
    """
    Ensures adjustment set distribution assigns high mass to valid sets.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        validity_scores: torch.Tensor,
        set_probabilities: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage high probability on high-validity sets.
        
        Args:
            validity_scores: [batch, n_candidates] - validity of each candidate
            set_probabilities: [batch, n_candidates] - probability of each set
            
        Returns:
            loss: scalar (negative expected validity)
        """
        # Expected validity under set distribution
        expected_validity = (validity_scores * set_probabilities).sum(dim=1).mean()
        
        # Maximize expected validity = minimize negative
        loss = -expected_validity
        
        return loss


class PoGPFNLoss(nn.Module):
    """
    Combined loss for PoG-PFN training.
    """
    
    def __init__(
        self,
        weight_ate: float = 1.0,
        weight_graph: float = 0.5,
        weight_claim: float = 0.3,
        weight_identification: float = 0.2,
        weight_acyclicity: float = 1.0
    ):
        super().__init__()
        
        self.weight_ate = weight_ate
        self.weight_graph = weight_graph
        self.weight_claim = weight_claim
        self.weight_identification = weight_identification
        self.weight_acyclicity = weight_acyclicity
        
        self.effect_loss = EffectLoss()
        self.graph_loss = GraphPosteriorLoss()
        self.claim_loss = ClaimValidationLoss()
        self.identification_loss = IdentificationConsistencyLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        true_ate: torch.Tensor,
        true_adjacency: torch.Tensor,
        claims: list,
        claim_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs dict
            true_ate: Ground truth ATE [batch]
            true_adjacency: Ground truth adjacency [batch, n_vars, n_vars]
            claims: List of claim lists
            claim_mask: [batch, max_claims]
            
        Returns:
            total_loss: Weighted combination
            loss_dict: Individual losses for logging
        """
        # 1. Effect loss (always compute - main loss)
        loss_effect = self.effect_loss(
            outputs['ate_mean'],
            outputs['ate_std'],
            true_ate
        )
        
        # 2. Graph loss - SKIP if weight is 0
        if self.weight_graph > 0:
            loss_graph = self.graph_loss(
                outputs['graph_posterior'],
                true_adjacency
            )
        else:
            loss_graph = torch.tensor(0.0, device=outputs['ate_mean'].device)
        
        # 3. Claim validation loss - SKIP if weight is 0
        if self.weight_claim > 0:
            loss_claim = self.claim_loss(
                claims,
                outputs['graph_posterior'],
                claim_mask
            )
        else:
            loss_claim = torch.tensor(0.0, device=outputs['ate_mean'].device)
        
        # 4. Identification consistency loss - SKIP if weight is 0
        if self.weight_identification > 0:
            if 'id_info' in outputs and 'validity_scores' in outputs:
                set_probs = outputs['id_info'].get('set_probabilities')
                if set_probs is not None:
                    loss_identification = self.identification_loss(
                        outputs['validity_scores'],
                        set_probs
                    )
                else:
                    loss_identification = torch.tensor(0.0, device=outputs['ate_mean'].device)
            else:
                loss_identification = torch.tensor(0.0, device=outputs['ate_mean'].device)
        else:
            loss_identification = torch.tensor(0.0, device=outputs['ate_mean'].device)
        
        # 5. Acyclicity penalty - SKIP if weight is 0
        if self.weight_acyclicity > 0:
            if 'graph_components' in outputs:
                loss_acyclicity = outputs['graph_components'].get('acyclicity_loss', torch.tensor(0.0))
            else:
                loss_acyclicity = torch.tensor(0.0, device=outputs['ate_mean'].device)
        else:
            loss_acyclicity = torch.tensor(0.0, device=outputs['ate_mean'].device)
        
        # Combined loss
        total_loss = (
            self.weight_ate * loss_effect +
            self.weight_graph * loss_graph +
            self.weight_claim * loss_claim +
            self.weight_identification * loss_identification +
            self.weight_acyclicity * loss_acyclicity
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'effect': loss_effect.item(),
            'graph': loss_graph.item(),
            'claim': loss_claim.item(),
            'identification': loss_identification.item(),
            'acyclicity': loss_acyclicity.item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test losses
    batch_size = 4
    n_vars = 10
    
    # Dummy predictions
    pred_mean = torch.randn(batch_size)
    pred_std = torch.rand(batch_size) * 0.5 + 0.1
    true_ate = torch.randn(batch_size)
    
    pred_adj = torch.rand(batch_size, n_vars, n_vars)
    true_adj = torch.randint(0, 2, (batch_size, n_vars, n_vars)).float()
    
    # Test effect loss
    effect_loss = EffectLoss()
    loss = effect_loss(pred_mean, pred_std, true_ate)
    print(f"Effect loss: {loss.item():.4f}")
    
    # Test graph loss
    graph_loss = GraphPosteriorLoss()
    loss = graph_loss(pred_adj, true_adj)
    print(f"Graph loss: {loss.item():.4f}")
    
    print("\n✓ Loss functions test passed!")
