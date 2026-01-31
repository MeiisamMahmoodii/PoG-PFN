"""
Simple training script for PoG-PFN

Demonstrates end-to-end training with synthetic data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pog_pfn.models.pog_pfn import PoGPFN
from pog_pfn.data.dataset import PoGPFNDataset, collate_fn
from pog_pfn.data.scm_generator import GraphType, MechanismType
from pog_pfn.training.losses import PoGPFNLoss
from pog_pfn.utils import set_seed, get_device, count_parameters


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for batch in dataloader:
        # Move to device
        X = batch['X'].to(device)
        T = batch['T'].to(device)
        Y = batch['Y'].to(device)
        true_ate = batch['true_ate'].to(device)
        true_adj = batch['true_adjacency'].to(device)
        claims = batch['claims']
        treatment_idx = batch['treatment_idx']
        outcome_idx = batch['outcome_idx']
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            X, T, Y, claims,
            treatment_idx, outcome_idx,
            return_uncertainty=True,
            return_graph_posterior=True,
            return_all_info=True
        )
        
        # Compute loss
        claim_mask = torch.zeros(X.shape[0], len(claims[0]), dtype=torch.bool, device=device)
        loss, loss_dict = criterion(outputs, true_ate, true_adj, claims, claim_mask)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def main():
    # Configuration
    config = {
        'n_vars': 10,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 3,
        'n_claim_layers': 2,
        'batch_size': 8,
        'n_tasks': 100,
        'n_epochs': 5,
        'lr': 1e-4,
        'seed': 42
    }
    
    print("=" * 60)
    print("PoG-PFN Training Demo")
    print("=" * 60)
    
    # Set seed
    set_seed(config['seed'])
    device = get_device()
    print(f"\nDevice: {device}")
    
    # Create model
    print("\nInitializing model...")
    model = PoGPFN(
        n_vars=config['n_vars'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        n_claim_layers=config['n_claim_layers'],
        dropout=0.1,
        max_seq_len=500,
        acyclicity_penalty=1.0
    ).to(device)
    
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = PoGPFNDataset(
        n_tasks=config['n_tasks'],
        n_vars=config['n_vars'],
        n_samples_per_task=200,
        n_claims_per_task=5,
        graph_types=[GraphType.ERDOS_RENYI, GraphType.CHAIN],
        mechanism_types=[MechanismType.LINEAR_GAUSSIAN, MechanismType.NONLINEAR_ADDITIVE],
        densities=[0.2, 0.3],
        seed=config['seed']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for debugging
    )
    
    print(f"Dataset size: {len(dataset)} tasks")
    
    # Create optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    criterion = PoGPFNLoss(
        weight_ate=1.0,
        weight_graph=0.5,
        weight_claim=0.3,
        weight_identification=0.2,
        weight_acyclicity=1.0
    )
    
    # Training loop
    print("\nStarting training...")
    print("-" * 60)
    
    for epoch in range(config['n_epochs']):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{config['n_epochs']} - Loss: {avg_loss:.4f}")
    
    print("-" * 60)
    print("\n✓ Training completed successfully!")
    
    # Test prediction
    print("\nTesting prediction on a sample...")
    model.eval()
    sample = dataset[0]
    
    with torch.no_grad():
        outputs = model(
            sample['X'].unsqueeze(0).to(device),
            sample['T'].unsqueeze(0).to(device),
            sample['Y'].unsqueeze(0).to(device),
            [sample['claims']],
            sample['treatment_idx'],
            sample['outcome_idx'],
            return_uncertainty=True,
            return_graph_posterior=True,
            return_all_info=False
        )
    
    pred_ate = outputs['ate_mean'].item()
    pred_std = outputs['ate_std'].item()
    true_ate = sample['true_ate'].item()
    
    print(f"  Predicted ATE: {pred_ate:.4f} ± {pred_std:.4f}")
    print(f"  True ATE: {true_ate:.4f}")
    print(f"  Absolute Error: {abs(pred_ate - true_ate):.4f}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
