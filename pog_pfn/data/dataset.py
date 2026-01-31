"""PyTorch Dataset for PoG-PFN"""

import torch
from torch.utils.data import Dataset
import numpy as np
import random
from typing import List, Tuple, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pog_pfn.data.scm_generator import SCMGenerator, GraphType, MechanismType
from pog_pfn.data.claim_generator import ClaimGenerator
from pog_pfn.models.claim_encoder import Claim


class PoGPFNDataset(Dataset):
    """
    Dataset that generates SCMs, samples data, and generates claims on-the-fly.
    """
    
    def __init__(
        self,
        n_tasks: int = 1000,
        n_vars: int = 10,
        n_samples_per_task: int = 500,
        n_claims_per_task: int = 5,
        graph_types: List[GraphType] = None,
        mechanism_types: List[MechanismType] = None,
        densities: List[float] = None,
        claim_config: dict = None,
        treatment_idx: Optional[int] = None,
        outcome_idx: Optional[int] = None,
        seed: Optional[int] = None
    ):
        self.n_tasks = n_tasks
        self.n_vars = n_vars
        self.n_samples_per_task = n_samples_per_task
        self.n_claims_per_task = n_claims_per_task
        
        # Default configurations
        self.graph_types = graph_types or [GraphType.ERDOS_RENYI, GraphType.CHAIN, GraphType.FORK]
        self.mechanism_types = mechanism_types or [
            MechanismType.LINEAR_GAUSSIAN,
            MechanismType.NONLINEAR_ADDITIVE
        ]
        self.densities = densities or [0.2, 0.3]
        
        claim_config = claim_config or {}
        self.claim_generator = ClaimGenerator(
            truthful_ratio=claim_config.get('truthful_ratio', 0.6),
            false_ratio=claim_config.get('false_ratio', 0.2),
            unidentifiable_ratio=claim_config.get('unidentifiable_ratio', 0.1),
            conflicting_ratio=claim_config.get('conflicting_ratio', 0.1),
            seed=seed
        )
        
        # Treatment/outcome indices
        self.treatment_idx = treatment_idx
        self.outcome_idx = outcome_idx
        
        self.seed = seed
        if seed is None:
            self.seed = np.random.randint(0, 1000000)
    
    def __len__(self) -> int:
        return self.n_tasks
    
    def __getitem__(self, idx: int) -> dict:
        """
        Generate one task (SCM + data + claims).
        
        Returns:
            dict with:
                - X: features [n_samples, n_features]
                - T: treatment [n_samples]
                - Y: outcome [n_samples]
                - claims: List[Claim]
                - true_ate: scalar
                - true_adjacency: [n_vars, n_vars]
                - treatment_idx: int
                - outcome_idx: int
        """
        # Set task-specific seed
        task_seed = self.seed + idx
        np.random.seed(task_seed)
        random.seed(task_seed)
        
        # Sample graph configuration (use random.choice for enums)
        graph_type = random.choice(self.graph_types)
        mechanism_type = random.choice(self.mechanism_types)
        density = np.random.choice(self.densities)
        
        # Generate SCM
        scm = SCMGenerator(
            n_vars=self.n_vars,
            graph_type=graph_type,
            mechanism_type=mechanism_type,
            density=density,
            seed=task_seed
        )
        
        # Sample observational data
        data, info = scm.sample(n_samples=self.n_samples_per_task)
        
        # Select treatment and outcome
        if self.treatment_idx is None:
            treatment_idx = np.random.randint(0, self.n_vars - 1)
        else:
            treatment_idx = self.treatment_idx
        
        if self.outcome_idx is None:
            outcome_idx = np.random.randint(treatment_idx + 1, self.n_vars)
        else:
            outcome_idx = self.outcome_idx
        
        # Binarize treatment (for simplicity)
        T = (data[:, treatment_idx] > np.median(data[:, treatment_idx])).astype(float)
        Y = data[:, outcome_idx]
        
        # Features: all variables except outcome
        feature_indices = [i for i in range(self.n_vars) if i != outcome_idx]
        X = data[:, feature_indices]
        
        # Generate claims
        claims = self.claim_generator.generate_claims(
            adjacency=info['adjacency'],
            n_claims=self.n_claims_per_task
        )
        
        # Compute true ATE
        true_ate = scm.estimate_true_ate(
            treatment_idx=treatment_idx,
            outcome_idx=outcome_idx,
            n_samples=5000,
            treatment_values=(0.0, 1.0)
        )
        
        return {
            'X': torch.from_numpy(X).float(),
            'T': torch.from_numpy(T).float(),
            'Y': torch.from_numpy(Y).float(),
            'claims': claims,
            'true_ate': torch.tensor(true_ate).float(),
            'true_adjacency': torch.from_numpy(info['adjacency']).float(),
            'treatment_idx': treatment_idx,
            'outcome_idx': outcome_idx,
            'n_vars': self.n_vars
        }


def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate function to handle variable-length claims.
    """
    # Stack tensors
    X = torch.stack([item['X'] for item in batch])
    T = torch.stack([item['T'] for item in batch])
    Y = torch.stack([item['Y'] for item in batch])
    true_ate = torch.stack([item['true_ate'] for item in batch])
    true_adjacency = torch.stack([item['true_adjacency'] for item in batch])
    
    # Collect claims (variable length)
    claims = [item['claims'] for item in batch]
    
    # Treatment/outcome indices (assume same for all in batch for simplicity)
    treatment_idx = batch[0]['treatment_idx']
    outcome_idx = batch[0]['outcome_idx']
    n_vars = batch[0]['n_vars']
    
    return {
        'X': X,
        'T': T,
        'Y': Y,
        'claims': claims,
        'true_ate': true_ate,
        'true_adjacency': true_adjacency,
        'treatment_idx': treatment_idx,
        'outcome_idx': outcome_idx,
        'n_vars': n_vars
    }


if __name__ == "__main__":
    # Test dataset
    dataset = PoGPFNDataset(
        n_tasks=10,
        n_vars=8,
        n_samples_per_task=100,
        n_claims_per_task=5,
        seed=42
    )
    
    # Test __getitem__
    sample = dataset[0]
    print(f"X shape: {sample['X'].shape}")
    print(f"T shape: {sample['T'].shape}")
    print(f"Y shape: {sample['Y'].shape}")
    print(f"Number of claims: {len(sample['claims'])}")
    print(f"True ATE: {sample['true_ate'].item():.4f}")
    print(f"True adjacency shape: {sample['true_adjacency'].shape}")
    
    # Test DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    batch = next(iter(loader))
    print(f"\nBatch X shape: {batch['X'].shape}")
    print(f"Batch claims: {len(batch['claims'])} lists")
    
    print("\n✓ Dataset test passed!")
