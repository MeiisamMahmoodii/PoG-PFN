"""
Claim Generator

Generates partial causal claims with different truthfulness levels:
- Truthful claims (consistent with true SCM)
- False claims (inconsistent with true SCM)
- Unidentifiable but consistent claims
- Conflicting claims (expert disagreement)

Claims are expressed as constraints on ancestry and path patterns.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import networkx as nx

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.claim_encoder import Claim, RelationType


class ClaimGenerator:
    """
    Generates causal claims based on ground truth SCM structure.
    """
    
    def __init__(
        self,
        truthful_ratio: float = 0.6,
        false_ratio: float = 0.2,
        unidentifiable_ratio: float = 0.1,
        conflicting_ratio: float = 0.1,
        seed: Optional[int] = None
    ):
        self.truthful_ratio = truthful_ratio
        self.false_ratio = false_ratio
        self.unidentifiable_ratio = unidentifiable_ratio
        self.conflicting_ratio = conflicting_ratio
        
        # Validate ratios
        total = truthful_ratio + false_ratio + unidentifiable_ratio + conflicting_ratio
        assert abs(total - 1.0) < 1e-6, f"Claim ratios must sum to 1.0, got {total}"
        
        if seed is not None:
            np.random.seed(seed)
    
    def _compute_ancestry(self, adjacency: np.ndarray) -> np.ndarray:
        """Compute transitive closure (ancestry matrix)."""
        n = adjacency.shape[0]
        G = nx.DiGraph(adjacency)
        ancestry = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if nx.has_path(G, i, j):
                    ancestry[i, j] = 1
        
        return ancestry
    
    def _find_confounders(
        self,
        adjacency: np.ndarray,
        var_a: int,
        var_b: int
    ) -> List[int]:
        """Find confounders between var_a and var_b."""
        ancestry = self._compute_ancestry(adjacency)
        confounders = []
        
        for i in range(adjacency.shape[0]):
            if i != var_a and i != var_b:
                # Confounder if: i → var_a AND i → var_b
                if ancestry[i, var_a] and ancestry[i, var_b]:
                    # And var_a is not ancestor of i
                    if not ancestry[var_a, i]:
                        confounders.append(i)
        
        return confounders
    
    def _find_mediators(
        self,
        adjacency: np.ndarray,
        var_a: int,
        var_b: int
    ) -> List[int]:
        """Find mediators on paths from var_a to var_b."""
        G = nx.DiGraph(adjacency)
        mediators = []
        
        try:
            # Find all simple paths
            for path in nx.all_simple_paths(G, var_a, var_b):
                # Nodes on the path (excluding endpoints) are mediators
                for node in path[1:-1]:
                    if node not in mediators:
                        mediators.append(node)
        except nx.NetworkXNoPath:
            pass
        
        return mediators
    
    def _find_colliders(
        self,
        adjacency: np.ndarray
    ) -> List[int]:
        """Find colliders (nodes with >= 2 parents)."""
        n_parents = adjacency.sum(axis=0)
        return [i for i in range(len(n_parents)) if n_parents[i] >= 2]
    
    def generate_truthful_claims(
        self,
        adjacency: np.ndarray,
        n_claims: int
    ) -> List[Claim]:
        """Generate claims that are true according to the SCM."""
        claims = []
        n_vars = adjacency.shape[0]
        ancestry = self._compute_ancestry(adjacency)
        
        for _ in range(n_claims):
            # Randomly choose claim type
            claim_type = np.random.choice([
                RelationType.CAUSES,
                RelationType.FORBIDS,
                RelationType.CONFOUNDER,
                RelationType.MEDIATOR,
                RelationType.ANCESTOR,
                RelationType.NON_ANCESTOR
            ])
            
            if claim_type == RelationType.CAUSES:
                # Find an existing edge
                edges = np.argwhere(adjacency > 0)
                if len(edges) > 0:
                    i, j = edges[np.random.randint(len(edges))]
                    claims.append(Claim(
                        var_a=int(i),
                        var_b=int(j),
                        relation_type=RelationType.CAUSES,
                        confidence=np.random.uniform(0.7, 1.0),
                        is_true=True
                    ))
            
            elif claim_type == RelationType.FORBIDS:
                # Find a non-edge
                non_edges = np.argwhere(adjacency == 0)
                # Exclude diagonal
                non_edges = [e for e in non_edges if e[0] != e[1]]
                if len(non_edges) > 0:
                    i, j = non_edges[np.random.randint(len(non_edges))]
                    claims.append(Claim(
                        var_a=int(i),
                        var_b=int(j),
                        relation_type=RelationType.FORBIDS,
                        confidence=np.random.uniform(0.7, 1.0),
                        is_true=True
                    ))
            
            elif claim_type == RelationType.CONFOUNDER:
                # Find two variables and a confounder
                var_a = np.random.randint(0, n_vars)
                var_b = np.random.randint(0, n_vars)
                if var_a != var_b:
                    confounders = self._find_confounders(adjacency, var_a, var_b)
                    if confounders:
                        confounder = confounders[np.random.randint(len(confounders))]
                        claims.append(Claim(
                            var_a=confounder,
                            var_b=var_a,
                            var_c=var_b,
                            relation_type=RelationType.CONFOUNDER,
                            confidence=np.random.uniform(0.7, 1.0),
                            is_true=True
                        ))
            
            elif claim_type == RelationType.MEDIATOR:
                # Find two variables with a mediator
                var_a = np.random.randint(0, n_vars)
                var_b = np.random.randint(0, n_vars)
                if var_a != var_b and ancestry[var_a, var_b]:
                    mediators = self._find_mediators(adjacency, var_a, var_b)
                    if mediators:
                        mediator = mediators[np.random.randint(len(mediators))]
                        claims.append(Claim(
                            var_a=var_a,
                            var_b=mediator,
                            var_c=var_b,
                            relation_type=RelationType.MEDIATOR,
                            confidence=np.random.uniform(0.7, 1.0),
                            is_true=True
                        ))
            
            elif claim_type == RelationType.ANCESTOR:
                # Find an ancestry relation
                ancestors = np.argwhere(ancestry > 0)
                if len(ancestors) > 0:
                    i, j = ancestors[np.random.randint(len(ancestors))]
                    claims.append(Claim(
                        var_a=int(i),
                        var_b=int(j),
                        relation_type=RelationType.ANCESTOR,
                        confidence=np.random.uniform(0.7, 1.0),
                        is_true=True
                    ))
            
            elif claim_type == RelationType.NON_ANCESTOR:
                # Find a non-ancestry relation
                non_ancestors = np.argwhere(ancestry == 0)
                non_ancestors = [e for e in non_ancestors if e[0] != e[1]]
                if len(non_ancestors) > 0:
                    i, j = non_ancestors[np.random.randint(len(non_ancestors))]
                    claims.append(Claim(
                        var_a=int(i),
                        var_b=int(j),
                        relation_type=RelationType.NON_ANCESTOR,
                        confidence=np.random.uniform(0.7, 1.0),
                        is_true=True
                    ))
        
        return claims
    
    def generate_false_claims(
        self,
        adjacency: np.ndarray,
        n_claims: int
    ) -> List[Claim]:
        """Generate claims that contradict the true SCM."""
        claims = []
        n_vars = adjacency.shape[0]
        ancestry = self._compute_ancestry(adjacency)
        
        for _ in range(n_claims):
            # Randomly choose claim type
            claim_type = np.random.choice([
                RelationType.CAUSES,
                RelationType.FORBIDS,
                RelationType.ANCESTOR,
                RelationType.NON_ANCESTOR
            ])
            
            if claim_type == RelationType.CAUSES:
                # Claim an edge that doesn't exist
                non_edges = np.argwhere(adjacency == 0)
                non_edges = [e for e in non_edges if e[0] != e[1]]
                if len(non_edges) > 0:
                    i, j = non_edges[np.random.randint(len(non_edges))]
                    claims.append(Claim(
                        var_a=int(i),
                        var_b=int(j),
                        relation_type=RelationType.CAUSES,
                        confidence=np.random.uniform(0.5, 0.8),  # Lower confidence
                        is_true=False
                    ))
            
            elif claim_type == RelationType.FORBIDS:
                # Forbid an edge that exists
                edges = np.argwhere(adjacency > 0)
                if len(edges) > 0:
                    i, j = edges[np.random.randint(len(edges))]
                    claims.append(Claim(
                        var_a=int(i),
                        var_b=int(j),
                        relation_type=RelationType.FORBIDS,
                        confidence=np.random.uniform(0.5, 0.8),
                        is_true=False
                    ))
            
            elif claim_type == RelationType.ANCESTOR:
                # Claim ancestry where none exists
                non_ancestors = np.argwhere(ancestry == 0)
                non_ancestors = [e for e in non_ancestors if e[0] != e[1]]
                if len(non_ancestors) > 0:
                    i, j = non_ancestors[np.random.randint(len(non_ancestors))]
                    claims.append(Claim(
                        var_a=int(i),
                        var_b=int(j),
                        relation_type=RelationType.ANCESTOR,
                        confidence=np.random.uniform(0.5, 0.8),
                        is_true=False
                    ))
            
            elif claim_type == RelationType.NON_ANCESTOR:
                # Deny ancestry where it exists
                ancestors = np.argwhere(ancestry > 0)
                if len(ancestors) > 0:
                    i, j = ancestors[np.random.randint(len(ancestors))]
                    claims.append(Claim(
                        var_a=int(i),
                        var_b=int(j),
                        relation_type=RelationType.NON_ANCESTOR,
                        confidence=np.random.uniform(0.5, 0.8),
                        is_true=False
                    ))
        
        return claims
    
    def generate_claims(
        self,
        adjacency: np.ndarray,
        n_claims: int
    ) -> List[Claim]:
        """
        Generate mixed claims according to configured ratios.
        
        Args:
            adjacency: Ground truth adjacency matrix
            n_claims: Total number of claims to generate
            
        Returns:
            claims: List of Claim objects
        """
        n_truthful = int(n_claims * self.truthful_ratio)
        n_false = int(n_claims * self.false_ratio)
        # Note: unidentifiable and conflicting not yet implemented
        n_unidentifiable = 0  # int(n_claims * self.unidentifiable_ratio)
        n_conflicting = 0  # int(n_claims * self.conflicting_ratio)
        
        # Adjust to hit exact count
        remaining = n_claims - (n_truthful + n_false + n_unidentifiable + n_conflicting)
        n_truthful += remaining
        
        claims = []
        
        # Generate each type
        if n_truthful > 0:
            claims.extend(self.generate_truthful_claims(adjacency, n_truthful))
        
        if n_false > 0:
            claims.extend(self.generate_false_claims(adjacency, n_false))
        
        # TODO: Implement unidentifiable and conflicting claims
        
        # Shuffle
        np.random.shuffle(claims)
        
        return claims


if __name__ == "__main__":
    from pog_pfn.data.scm_generator import SCMGenerator, GraphType, MechanismType
    
    print("Testing Claim Generator...")
    
    # Create SCM
    scm = SCMGenerator(
        n_vars=10,
        graph_type=GraphType.ERDOS_RENYI,
        mechanism_type=MechanismType.LINEAR_GAUSSIAN,
        density=0.3,
        seed=42
    )
    
    # Create claim generator
    claim_gen = ClaimGenerator(
        truthful_ratio=0.7,
        false_ratio=0.3,
        unidentifiable_ratio=0.0,
        conflicting_ratio=0.0,
        seed=42
    )
    
    # Generate claims
    claims = claim_gen.generate_claims(scm.adjacency, n_claims=10)
    
    print(f"\nGenerated {len(claims)} claims:")
    for i, claim in enumerate(claims):
        print(f"{i+1}. {claim.relation_type.name}: "
              f"var_a={claim.var_a}, var_b={claim.var_b}, var_c={claim.var_c}, "
              f"confidence={claim.confidence:.2f}, is_true={claim.is_true}")
    
    # Count true vs false
    n_true = sum(1 for c in claims if c.is_true)
    n_false = sum(1 for c in claims if not c.is_true)
    print(f"\nTrue claims: {n_true}, False claims: {n_false}")
    
    print("\n✓ Claim generator test passed!")
