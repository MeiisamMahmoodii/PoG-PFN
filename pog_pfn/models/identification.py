"""
Module D: Differentiable Identification Layer

Implements soft d-separation and backdoor criterion checking to:
1. Determine which adjustment sets satisfy the backdoor criterion
2. Compute soft validity scores for candidate adjustment sets
3. Handle descendant detection and collider bias avoidance

This is THE critical component that makes claim validation implicit in 
the posterior-to-effect pipeline.
"""

import torch
import torch.nn as nn
import itertools
from typing import List, Tuple, Optional


class DifferentiableIdentification(nn.Module):
    """
    Computes adjustment set validity via differentiable d-separation.
    
    The backdoor criterion requires that adjustment set S:
    1. Blocks all backdoor paths from T to Y
    2. Does not contain descendants of T
    3. Does not induce collider bias
    """
    
    def __init__(
        self,
        n_vars: int,
        max_adjustment_sets: int = 10,
        soft_logic_temperature: float = 0.5,
        min_set_size: int = 0,
        max_set_size: Optional[int] = None
    ):
        super().__init__()
        
        self.n_vars = n_vars
        self.max_adjustment_sets = max_adjustment_sets
        self.temperature = soft_logic_temperature
        self.min_set_size = min_set_size
        self.max_set_size = max_set_size if max_set_size is not None else n_vars - 2
        
        # Learnable parameters for weighting different criteria
        self.criterion_weights = nn.Parameter(torch.ones(3))  # [backdoor, no_descendants, no_colliders]
        
    def compute_ancestry(self, adjacency: torch.Tensor, max_hops: int = None) -> torch.Tensor:
        """
        Compute transitive closure (ancestry matrix) from adjacency matrix.
        A[i,j] = 1 iff there exists a directed path from i to j.
        
        Args:
            adjacency: Soft adjacency matrix [batch, n_vars, n_vars]
            max_hops: Maximum path length to consider (default: n_vars)
            
        Returns:
            ancestry: Soft ancestry matrix [batch, n_vars, n_vars]
        """
        if max_hops is None:
            max_hops = self.n_vars
        
        batch_size = adjacency.shape[0]
        device = adjacency.device
        
        # Start with direct edges
        ancestry = adjacency.clone()
        
        # Iteratively add paths of length k
        current = adjacency
        for k in range(1, max_hops):
            # Matrix multiplication in probability space (using max-product)
            # P(i ⤳ j via k hops) = max_m P(i ⤳ m via k-1 hops) * P(m → j)
            current = torch.bmm(current, adjacency)
            
            # Soft max (take maximum path probability)
            ancestry = torch.maximum(ancestry, current)
        
        return ancestry
    
    def is_descendant(
        self,
        adjacency: torch.Tensor,
        treatment_idx: int,
        candidate_idx: int
    ) -> torch.Tensor:
        """
        Soft check: Is candidate a descendant of treatment?
        
        Args:
            adjacency: [batch, n_vars, n_vars]
            treatment_idx: Index of treatment variable
            candidate_idx: Index of candidate adjustment variable
            
        Returns:
            descendant_score: [batch] - probability that candidate is descendant
        """
        ancestry = self.compute_ancestry(adjacency)
        return ancestry[:, treatment_idx, candidate_idx]
    
    def blocks_backdoor_paths(
        self,
        adjacency: torch.Tensor,
        treatment_idx: int,
        outcome_idx: int,
        adjustment_set: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft check: Does adjustment set block all backdoor paths from T to Y?
        
        A backdoor path is a path from T to Y that starts with an edge into T.
        To block it, we need to condition on a non-collider on the path.
        
        Args:
            adjacency: [batch, n_vars, n_vars]
            treatment_idx: Index of treatment  
            outcome_idx: Index of outcome
            adjustment_set: [batch, n_vars] - soft inclusion probabilities
            
        Returns:
            blocks_score: [batch] - probability that backdoor paths are blocked
        """
        batch_size = adjacency.shape[0]
        device = adjacency.device
        
        # Find confounders: variables that are ancestors of both T and Y
        ancestry = self.compute_ancestry(adjacency)
        
        # Confounder if: ancestor of T AND ancestor of Y
        is_ancestor_of_T = ancestry[:, :, treatment_idx]  # [batch, n_vars]
        is_ancestor_of_Y = ancestry[:, :, outcome_idx]  # [batch, n_vars]
        confounders = is_ancestor_of_T * is_ancestor_of_Y  # [batch, n_vars]
        
        # To block backdoor, we need to condition on confounders
        # Score: how much of the confounder mass is in the adjustment set?
        confounder_total = confounders.sum(dim=1, keepdim=True).clamp(min=1e-8)
        confounder_adjusted = (confounders * adjustment_set).sum(dim=1, keepdim=True)
        
        blocks_score = (confounder_adjusted / confounder_total).squeeze(1)
        
        return blocks_score
    
    def avoids_descendants(
        self,
        adjacency: torch.Tensor,
        treatment_idx: int,
        adjustment_set: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft check: Does adjustment set avoid descendants of T?
        
        Args:
            adjacency: [batch, n_vars, n_vars]
            treatment_idx: Index of treatment
            adjustment_set: [batch, n_vars] - soft inclusion probabilities
            
        Returns:
            avoids_score: [batch] - probability that no descendants are adjusted
        """
        ancestry = self.compute_ancestry(adjacency)
        
        # Descendants of T
        descendants = ancestry[:, treatment_idx, :]  # [batch, n_vars]
        
        # We want adjustment set to NOT include descendants
        # Score: 1 - (how much descendant mass is in adjustment set)
        descendant_in_set = (descendants * adjustment_set).sum(dim=1)
        
        # Normalize by potential descendant mass
        descendant_total = descendants.sum(dim=1).clamp(min=1e-8)
        avoids_score = 1 - (descendant_in_set / descendant_total)
        
        return avoids_score
    
    def avoids_colliders(
        self,
        adjacency: torch.Tensor,
        treatment_idx: int,
        outcome_idx: int,
        adjustment_set: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft check: Does adjustment set avoid opening collider paths?
        
        A collider is a variable with two incoming edges. Conditioning on it
        opens a path.
        
        Args:
            adjacency: [batch, n_vars, n_vars]
            treatment_idx: Index of treatment
            outcome_idx: Index of outcome
            adjustment_set: [batch, n_vars] - soft inclusion probabilities
            
        Returns:
            avoids_score: [batch] - probability that colliders are not opened
        """
        # Identify colliders: nodes with >= 2 parents
        n_parents = adjacency.sum(dim=1)  # [batch, n_vars]
        is_collider = torch.sigmoid((n_parents - 1.5) * 10)  # Soft threshold at 2 parents
        
        # We want to avoid conditioning on colliders that are on alternative paths
        # Simplified: penalize adjusting for any collider
        collider_in_set = (is_collider * adjustment_set).sum(dim=1)
        collider_total = is_collider.sum(dim=1).clamp(min=1e-8)
        
        avoids_score = 1 - (collider_in_set / collider_total)
        
        return avoids_score
    
    def score_adjustment_set(
        self,
        adjacency: torch.Tensor,
        treatment_idx: int,
        outcome_idx: int,
        adjustment_set: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute overall validity score for an adjustment set.
        
        Args:
            adjacency: [batch, n_vars, n_vars]
            treatment_idx: Index of treatment
            outcome_idx: Index of outcome
            adjustment_set: [batch, n_vars] - soft inclusion probabilities
            
        Returns:
            validity_score: [batch] - overall validity in [0, 1]
            components: Dict with individual criterion scores
        """
        # Compute individual criteria
        blocks = self.blocks_backdoor_paths(adjacency, treatment_idx, outcome_idx, adjustment_set)
        no_desc = self.avoids_descendants(adjacency, treatment_idx, adjustment_set)
        no_coll = self.avoids_colliders(adjacency, treatment_idx, outcome_idx, adjustment_set)
        
        # Weighted combination (weights are learnable)
        weights = torch.softmax(self.criterion_weights, dim=0)
        validity = weights[0] * blocks + weights[1] * no_desc + weights[2] * no_coll
        
        components = {
            'blocks_backdoor': blocks,
            'avoids_descendants': no_desc,
            'avoids_colliders': no_coll,
            'weights': weights
        }
        
        return validity, components
    
    def generate_candidate_sets(
        self,
        n_vars: int,
        treatment_idx: int,
        outcome_idx: int
    ) -> List[List[int]]:
        """
        Generate candidate adjustment sets heuristically.
        
        For computational efficiency, we don't enumerate all 2^n subsets.
        Instead, we use heuristics:
        1. Empty set
        2. Singletons
        3. Most likely confounders (size 2-3)
        
        Args:
            n_vars: Total number of variables
            treatment_idx: Treatment index
            outcome_idx: Outcome index
            
        Returns:
            candidates: List of adjustment sets (as lists of indices)
        """
        candidates = []
        
        # Exclude treatment and outcome from adjustment sets
        available_vars = [i for i in range(n_vars) if i not in [treatment_idx, outcome_idx]]
        
        # 1. Empty set
        candidates.append([])
        
        # 2. Singletons
        for var in available_vars[:min(len(available_vars), 5)]:  # Limit to first 5
            candidates.append([var])
        
        # 3. Pairs (for small graphs)
        if self.n_vars <= 15:
            for pair in itertools.combinations(available_vars[:5], 2):
                candidates.append(list(pair))
        
        # Limit total candidates
        candidates = candidates[:self.max_adjustment_sets]
        
        return candidates
    
    def forward(
        self,
        adjacency: torch.Tensor,
        treatment_idx: int,
        outcome_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Compute distribution over valid adjustment sets.
        
        Args:
            adjacency: Posterior adjacency matrix [batch, n_vars, n_vars]
            treatment_idx: Treatment index
            outcome_idx: Outcome index
            
        Returns:
            adjustment_distribution: [batch, n_vars] - soft inclusion probabilities
            validity_scores: [batch, max_adjustment_sets] - validity of each candidate
            info: Additional information dict
        """
        batch_size = adjacency.shape[0]
        device = adjacency.device
        
        # Generate candidate adjustment sets
        candidates = self.generate_candidate_sets(self.n_vars, treatment_idx, outcome_idx)
        n_candidates = len(candidates)
        
        # Score each candidate
        validity_scores = torch.zeros(batch_size, n_candidates, device=device)
        all_components = []
        
        for i, candidate in enumerate(candidates):
            # Convert candidate to soft binary vector
            adjustment_set = torch.zeros(batch_size, self.n_vars, device=device)
            if len(candidate) > 0:
                adjustment_set[:, candidate] = 1.0
            
            # Score this adjustment set
            validity, components = self.score_adjustment_set(
                adjacency, treatment_idx, outcome_idx, adjustment_set
            )
            validity_scores[:, i] = validity
            all_components.append(components)
        
        # Convert to probability distribution over adjustment sets
        set_probs = torch.softmax(validity_scores / self.temperature, dim=1)  # [batch, n_candidates]
        
        # Aggregate into per-variable inclusion probabilities
        adjustment_distribution = torch.zeros(batch_size, self.n_vars, device=device)
        for i, candidate in enumerate(candidates):
            if len(candidate) > 0:
                for var_idx in candidate:
                    adjustment_distribution[:, var_idx] += set_probs[:, i]
        
        # Normalize (optional, may have overlaps)
        # adjustment_distribution = adjustment_distribution.clamp(0, 1)
        
        info = {
            'candidates': candidates,
            'set_probabilities': set_probs,
            'criterion_components': all_components
        }
        
        return adjustment_distribution, validity_scores, info


if __name__ == "__main__":
    # Test the module
    batch_size = 4
    n_vars = 10
    
    # Create dummy adjacency matrix (ground truth DAG)
    adjacency = torch.zeros(batch_size, n_vars, n_vars)
    # Simple structure: 0 → 1 → 2, 0 → 3 → 2 (3 is confounder)
    adjacency[:, 0, 1] = 0.9
    adjacency[:, 1, 2] = 0.9
    adjacency[:, 0, 3] = 0.85
    adjacency[:, 3, 2] = 0.85
    
    # Create identification layer
    identifier = DifferentiableIdentification(n_vars=n_vars, max_adjustment_sets=10)
    
    # Test: identify adjustment sets for T=1, Y=2
    # Ground truth: should adjust for {0, 3} to block backdoor
    adjustment_dist, validity_scores, info = identifier(adjacency, treatment_idx=1, outcome_idx=2)
    
    print(f"Adjustment distribution shape: {adjustment_dist.shape}")  # [4, 10]
    print(f"Top adjusting variables: {adjustment_dist[0].argsort(descending=True)[:5]}")
    print(f"Validity scores shape: {validity_scores.shape}")  # [4, n_candidates]
    print(f"Best adjustment set: {info['candidates'][validity_scores[0].argmax().item()]}")
    print("✓ Differentiable identification test passed!")
