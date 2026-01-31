"""
Utility functions for PoG-PFN
"""

import torch
import numpy as np
import random
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(use_cuda: bool = True) -> torch.device:
    """Get the appropriate device (CPU/CUDA)."""
    if use_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def create_dag_mask(n_vars: int, adjacency: torch.Tensor) -> torch.Tensor:
    """
    Create a mask that enforces DAG structure (no cycles).
    
    Args:
        n_vars: Number of variables
        adjacency: Soft adjacency matrix [n_vars, n_vars]
        
    Returns:
        mask: Binary mask where 1 indicates valid edges
    """
    # For simplicity, enforce lower triangular structure
    # (assumes some topological ordering, which is a reasonable prior)
    mask = torch.tril(torch.ones(n_vars, n_vars), diagonal=-1)
    return mask


def compute_h_acyclicity(adjacency: torch.Tensor) -> torch.Tensor:
    """
    Compute acyclicity constraint using NOTEARS formulation.
    h(W) = 0 iff W represents a DAG.
    
    Args:
        adjacency: Soft adjacency matrix [batch, n_vars, n_vars] or [n_vars, n_vars]
        
    Returns:
        h: Acyclicity penalty (0 for DAGs)
    """
    if adjacency.dim() == 2:
        adjacency = adjacency.unsqueeze(0)
    
    batch_size, d = adjacency.shape[0], adjacency.shape[1]
    
    # NOTEARS: h(W) = tr(e^(W ∘ W)) - d
    W_squared = adjacency * adjacency
    M = torch.matrix_exp(W_squared)
    h = torch.trace(M) - d
    
    return h.mean() if batch_size > 1 else h.squeeze()


def mask_fill_diagonal(tensor: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    """Fill diagonal of tensor with specified value."""
    n = tensor.shape[-1]
    mask = torch.eye(n, device=tensor.device, dtype=torch.bool)
    if tensor.dim() == 3:
        mask = mask.unsqueeze(0).expand(tensor.shape[0], -1, -1)
    return tensor.masked_fill(mask, value)


def soft_threshold(tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Apply soft thresholding to convert soft adjacency to binary."""
    return torch.sigmoid((tensor - threshold) * 10)  # Steep sigmoid around threshold


def compute_adjustment_validity(
    adjacency: torch.Tensor,
    treatment_idx: int,
    outcome_idx: int,
    adjustment_set_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute soft score for whether adjustment_set satisfies backdoor criterion.
    
    This is a simplified version - full implementation in identification.py
    
    Args:
        adjacency: Soft adjacency matrix [n_vars, n_vars]
        treatment_idx: Index of treatment variable
        outcome_idx: Index of outcome variable
        adjustment_set_mask: Binary mask [n_vars] indicating adjustment set
        
    Returns:
        validity_score: Soft score in [0, 1]
    """
    # Placeholder - will be implemented properly in identification module
    return torch.tensor(0.5)


def batch_gather(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Gather elements from tensor using indices.
    
    Args:
        tensor: [batch, seq_len, dim]
        indices: [batch, num_indices]
        
    Returns:
        gathered: [batch, num_indices, dim]
    """
    batch_size, seq_len, dim = tensor.shape
    indices = indices.unsqueeze(-1).expand(-1, -1, dim)
    return torch.gather(tensor, 1, indices)
