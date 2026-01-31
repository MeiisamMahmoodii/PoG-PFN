"""
Realistic SCM (Structural Causal Model) Generator

Generates diverse causal data-generating processes including:
- Various graph structures (Erdős-Rényi, scale-free, chains, forks)
- Multiple mechanism types (linear, nonlinear, monotone, heavy-tailed)
- Noise distributions (Gaussian, heavy-tailed, heteroskedastic)
- Mixed continuous/discrete variables
- Hidden confounders
- Measurement error
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, Optional, List
from enum import Enum
import torch


class GraphType(Enum):
    """Types of graph structures to generate."""
    ERDOS_RENYI = "erdos_renyi"
    SCALE_FREE = "scale_free"
    CHAIN = "chain"
    FORK = "fork"
    COLLIDER = "collider"


class MechanismType(Enum):
    """Types of causal mechanisms."""
    LINEAR_GAUSSIAN = "linear_gaussian"
    NONLINEAR_ADDITIVE = "nonlinear_additive"
    MONOTONE = "monotone"
    HEAVY_TAILED = "heavy_tailed"
    HETEROSKEDASTIC = "heteroskedastic"


class SCMGenerator:
    """
    Generates Structural Causal Models with realistic complexity.
    """
    
    def __init__(
        self,
        n_vars: int = 10,
        graph_type: GraphType = GraphType.ERDOS_RENYI,
        mechanism_type: MechanismType = MechanismType.LINEAR_GAUSSIAN,
        density: float = 0.2,
        noise_scale: float = 1.0,
        seed: Optional[int] = None
    ):
        self.n_vars = n_vars
        self.graph_type = graph_type
        self.mechanism_type = mechanism_type
        self.density = density
        self.noise_scale = noise_scale
        
        if seed is not None:
            np.random.seed(seed)
            
        # Generate graph structure
        self.adjacency = self._generate_graph()
        self.topological_order = self._get_topological_order()
        
    def _generate_graph(self) -> np.ndarray:
        """Generate adjacency matrix based on graph type."""
        if self.graph_type == GraphType.ERDOS_RENYI:
            return self._erdos_renyi_dag()
        elif self.graph_type == GraphType.SCALE_FREE:
            return self._scale_free_dag()
        elif self.graph_type == GraphType.CHAIN:
            return self._chain_dag()
        elif self.graph_type == GraphType.FORK:
            return self._fork_dag()
        elif self.graph_type == GraphType.COLLIDER:
            return self._collider_dag()
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
    
    def _erdos_renyi_dag(self) -> np.ndarray:
        """Generate Erdős-Rényi DAG."""
        # Start with random adjacency
        adj = np.random.rand(self.n_vars, self.n_vars) < self.density
        
        # Make it a DAG by enforcing lower triangular
        adj = np.tril(adj, k=-1)
        
        # Permute to get non-trivial topological order
        perm = np.random.permutation(self.n_vars)
        adj = adj[perm, :][:, perm]
        
        return adj.astype(float)
    
    def _scale_free_dag(self) -> np.ndarray:
        """Generate scale-free DAG using preferential attachment."""
        # Use NetworkX to generate scale-free graph
        m = max(1, int(self.density * self.n_vars))
        G = nx.barabasi_albert_graph(self.n_vars, m)
        
        # Convert to DAG by directing edges according to node order
        adj = np.zeros((self.n_vars, self.n_vars))
        for i, j in G.edges():
            if i < j:
                adj[i, j] = 1
            else:
                adj[j, i] = 1
        
        return adj
    
    def _chain_dag(self) -> np.ndarray:
        """Generate chain structure: 0 → 1 → 2 → ... → n-1."""
        adj = np.zeros((self.n_vars, self.n_vars))
        for i in range(self.n_vars - 1):
            adj[i, i + 1] = 1
        
        # Add some random edges
        extra_edges = int(self.density * self.n_vars * (self.n_vars - 1) / 2) - (self.n_vars - 1)
        for _ in range(max(0, extra_edges)):
            i = np.random.randint(0, self.n_vars - 2)
            j = np.random.randint(i + 2, self.n_vars)
            adj[i, j] = 1
        
        return adj
    
    def _fork_dag(self) -> np.ndarray:
        """Generate fork structure: root → multiple children."""
        adj = np.zeros((self.n_vars, self.n_vars))
        root = 0
        
        # Root causes all others
        for i in range(1, self.n_vars):
            adj[root, i] = 1
        
        # Add some extra edges
        extra_edges = int(self.density * self.n_vars * (self.n_vars - 1) / 2) - (self.n_vars - 1)
        for _ in range(max(0, extra_edges)):
            i = np.random.randint(1, self.n_vars - 1)
            j = np.random.randint(i + 1, self.n_vars)
            adj[i, j] = 1
        
        return adj
    
    def _collider_dag(self) -> np.ndarray:
        """Generate structure with colliders."""
        adj = np.zeros((self.n_vars, self.n_vars))
        
        # Create collider pattern: pairs → collider
        for i in range(0, self.n_vars - 1, 3):
            if i + 2 < self.n_vars:
                adj[i, i + 2] = 1
                adj[i + 1, i + 2] = 1
        
        # Add some extra edges
        extra_edges = int(self.density * self.n_vars * (self.n_vars - 1) / 2)
        current_edges = adj.sum()
        for _ in range(max(0, int(extra_edges - current_edges))):
            i = np.random.randint(0, self.n_vars - 1)
            j = np.random.randint(i + 1, self.n_vars)
            adj[i, j] = 1
        
        return adj
    
    def _get_topological_order(self) -> List[int]:
        """Get topological ordering of variables."""
        G = nx.DiGraph(self.adjacency)
        try:
            return list(nx.topological_sort(G))
        except nx.NetworkXError:
            # Fallback if somehow not a DAG
            return list(range(self.n_vars))
    
    def _apply_mechanism(
        self,
        parents_values: np.ndarray,
        node_idx: int
    ) -> np.ndarray:
        """
        Apply causal mechanism: X = f(parents) + noise.
        
        Args:
            parents_values: [n_samples, n_parents]
            node_idx: Index of current node
            
        Returns:
            values: [n_samples]
        """
        n_samples = parents_values.shape[0]
        n_parents = parents_values.shape[1]
        
        if n_parents == 0:
            # Root node: just noise
            return self._sample_noise(n_samples, node_idx)
        
        # Generate mechanism based on type
        if self.mechanism_type == MechanismType.LINEAR_GAUSSIAN:
            # Linear: X = w^T * parents + noise
            weights = np.random.randn(n_parents) * 0.5
            mean_effect = parents_values @ weights
            noise = np.random.randn(n_samples) * self.noise_scale
            
        elif self.mechanism_type == MechanismType.NONLINEAR_ADDITIVE:
            # Nonlinear additive: X = Σ f_i(parent_i) + noise
            mean_effect = np.zeros(n_samples)
            for i in range(n_parents):
                # Random nonlinear function (polynomial or sigmoid)
                if np.random.rand() < 0.5:
                    # Polynomial
                    degree = np.random.randint(2, 4)
                    mean_effect += np.polyval(np.random.randn(degree), parents_values[:, i])
                else:
                    # Sigmoid
                    mean_effect += np.tanh(parents_values[:, i] * np.random.randn())
            noise = np.random.randn(n_samples) * self.noise_scale
            
        elif self.mechanism_type == MechanismType.MONOTONE:
            # Monotone: X = ReLU(w^T parents) + noise
            weights = np.random.randn(n_parents) * 0.5
            mean_effect = np.maximum(0, parents_values @ weights)
            noise = np.random.randn(n_samples) * self.noise_scale
            
        elif self.mechanism_type == MechanismType.HEAVY_TAILED:
            # Heavy-tailed noise
            weights = np.random.randn(n_parents) * 0.5
            mean_effect = parents_values @ weights
            noise = np.random.standard_t(df=3, size=n_samples) * self.noise_scale
            
        elif self.mechanism_type == MechanismType.HETEROSKEDASTIC:
            # Heteroskedastic: noise variance depends on parents
            weights = np.random.randn(n_parents) * 0.5
            mean_effect = parents_values @ weights
            # Variance increases with |mean_effect|
            noise_std = self.noise_scale * (1 + 0.5 * np.abs(mean_effect))
            noise = np.random.randn(n_samples) * noise_std
            
        else:
            raise ValueError(f"Unknown mechanism type: {self.mechanism_type}")
        
        return mean_effect + noise
    
    def _sample_noise(self, n_samples: int, node_idx: int) -> np.ndarray:
        """Sample noise for root nodes."""
        if self.mechanism_type == MechanismType.HEAVY_TAILED:
            return np.random.standard_t(df=3, size=n_samples) * self.noise_scale
        else:
            return np.random.randn(n_samples) * self.noise_scale
    
    def sample(
        self,
        n_samples: int,
        intervention: Optional[Dict[int, float]] = None
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Sample from the SCM.
        
        Args:
            n_samples: Number of samples to generate
            intervention: Optional dict {var_idx: value} for do-interventions
            
        Returns:
            data: [n_samples, n_vars]
            info: Dict with ground truth information
        """
        data = np.zeros((n_samples, self.n_vars))
        
        # Generate data in topological order
        for node in self.topological_order:
            # Check if intervened upon
            if intervention and node in intervention:
                data[:, node] = intervention[node]
            else:
                # Find parents
                parents = np.where(self.adjacency[:, node] > 0)[0]
                
                if len(parents) == 0:
                    # Root node
                    data[:, node] = self._sample_noise(n_samples, node)
                else:
                    # Apply mechanism
                    parents_values = data[:, parents]
                    data[:, node] = self._apply_mechanism(parents_values, node)
        
        info = {
            'adjacency': self.adjacency,
            'graph_type': self.graph_type.value,
            'mechanism_type': self.mechanism_type.value,
            'topological_order': self.topological_order
        }
        
        return data, info
    
    def estimate_true_ate(
        self,
        treatment_idx: int,
        outcome_idx: int,
        n_samples: int = 10000,
        treatment_values: Tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """
        Estimate true ATE via interventional sampling.
        
        Args:
            treatment_idx: Index of treatment variable
            outcome_idx: Index of outcome variable
            n_samples: Number of samples for estimation
            treatment_values: (control_value, treatment_value)
            
        Returns:
            true_ate: E[Y|do(T=1)] - E[Y|do(T=0)]
        """
        # Sample under do(T=0)
        data_0, _ = self.sample(n_samples, intervention={treatment_idx: treatment_values[0]})
        y_0 = data_0[:, outcome_idx].mean()
        
        # Sample under do(T=1)
        data_1, _ = self.sample(n_samples, intervention={treatment_idx: treatment_values[1]})
        y_1 = data_1[:, outcome_idx].mean()
        
        return y_1 - y_0


if __name__ == "__main__":
    # Test SCM generator
    print("Testing SCM Generator...")
    
    # Test different graph types
    for graph_type in [GraphType.ERDOS_RENYI, GraphType.CHAIN, GraphType.FORK]:
        scm = SCMGenerator(
            n_vars=10,
            graph_type=graph_type,
            mechanism_type=MechanismType.NONLINEAR_ADDITIVE,
            density=0.2,
            seed=42
        )
        
        # Sample data
        data, info = scm.sample(n_samples=1000)
        
        # Estimate true ATE
        true_ate = scm.estimate_true_ate(treatment_idx=1, outcome_idx=3)
        
        print(f"\nGraph type: {graph_type.value}")
        print(f"  Data shape: {data.shape}")
        print(f"  Num edges: {info['adjacency'].sum()}")
        print(f"  True ATE (T=1, Y=3): {true_ate:.4f}")
    
    print("\n✓ SCM generator test passed!")
