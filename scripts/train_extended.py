"""
Extended Training Script with Logging and Metrics

Trains PoG-PFN with comprehensive logging, visualization, and evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pog_pfn.models.pog_pfn import PoGPFN
from pog_pfn.data.dataset import PoGPFNDataset, collate_fn
from pog_pfn.training.losses import PoGPFNLoss


class MetricsTracker:
    """Track and visualize training metrics."""
    
    def __init__(self, save_dir='results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_effect': [],
            'train_graph': [],
            'train_claim': [],
            'train_acyclicity': [],
            'val_loss': [],
            'val_mae': [],
            'val_crps': [],
        }
    
    def update(self, epoch, metrics):
        """Update metrics history."""
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def plot_losses(self):
        """Plot training losses."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total loss
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
        if self.history['val_loss'] and any(x is not None for x in self.history['val_loss']):
            valid_epochs = [e for e, v in zip(self.history['epoch'], self.history['val_loss']) if v is not None]
            valid_losses = [v for v in self.history['val_loss'] if v is not None]
            axes[0, 0].plot(valid_epochs, valid_losses, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Effect loss
        axes[0, 1].plot(self.history['epoch'], self.history['train_effect'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Effect Loss (CRPS)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Graph loss
        axes[1, 0].plot(self.history['epoch'], self.history['train_graph'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Graph Posterior Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Claim validation loss
        axes[1, 1].plot(self.history['epoch'], self.history['train_claim'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Claim Validation Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_losses.png', dpi=150)
        plt.close()
    
    def plot_metrics(self):
        """Plot evaluation metrics."""
        if not any(self.history['val_mae']):
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        valid_epochs = [e for e, v in zip(self.history['epoch'], self.history['val_mae']) if v is not None]
        valid_mae = [v for v in self.history['val_mae'] if v is not None]
        valid_crps = [v for v in self.history['val_crps'] if v is not None]
        
        # MAE
        axes[0].plot(valid_epochs, valid_mae, marker='o')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MAE')
        axes[0].set_title('Mean Absolute Error')
        axes[0].grid(True, alpha=0.3)
        
        # CRPS
        axes[1].plot(valid_epochs, valid_crps, marker='o')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('CRPS')
        axes[1].set_title('Continuous Ranked Probability Score')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'validation_metrics.png', dpi=150)
        plt.close()
    
    def save_history(self):
        """Save metrics history to JSON."""
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def print_summary(self):
        """Print training summary."""
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        print(f"Total epochs: {len(self.history['epoch'])}")
        print(f"Final train loss: {self.history['train_loss'][-1]:.4f}")
        if self.history['val_loss'][-1] is not None:
            print(f"Final val loss: {self.history['val_loss'][-1]:.4f}")
        if any(self.history['val_mae']):
            valid_mae = [v for v in self.history['val_mae'] if v is not None]
            if valid_mae:
                print(f"Best val MAE: {min(valid_mae):.4f}")
        print("="*60 + "\n")


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_crps = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X = batch['X'].to(device)
            T = batch['T'].to(device)
            Y = batch['Y'].to(device)
            claims = batch['claims']
            true_ate = batch['true_ate'].to(device)
            true_adj = batch['true_adjacency'].to(device)
            treatment_idx = batch['treatment_idx']
            outcome_idx = batch['outcome_idx']
            
            # Forward pass
            outputs = model(X, T, Y, claims, treatment_idx, outcome_idx)
            
            # Compute loss
            claim_mask = torch.zeros(len(claims), max(len(c) for c in claims), dtype=torch.bool, device=device)
            for i, claim_list in enumerate(claims):
                claim_mask[i, len(claim_list):] = True
            
            loss, loss_dict = criterion(outputs, true_ate, true_adj, claims, claim_mask)
            
            # Metrics
            mae = torch.abs(outputs['ate_mean'] - true_ate).mean()
            
            # CRPS (using the effect loss which is CRPS)
            crps = loss_dict['effect']
            
            total_loss += loss.item()
            total_mae += mae.item()
            total_crps += crps
            n_batches += 1
    
    return {
        'val_loss': total_loss / n_batches,
        'val_mae': total_mae / n_batches,
        'val_crps': total_crps / n_batches,
    }


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    """Train for one epoch with progress bar."""
    model.train()
    total_loss = 0.0
    loss_components = {
        'effect': 0.0,
        'graph': 0.0,
        'claim': 0.0,
        'acyclicity': 0.0,
    }
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs}', leave=True)
    for batch in pbar:
        X = batch['X'].to(device)
        T = batch['T'].to(device)
        Y = batch['Y'].to(device)
        claims = batch['claims']
        true_ate = batch['true_ate'].to(device)
        true_adj = batch['true_adjacency'].to(device)
        treatment_idx = batch['treatment_idx']
        outcome_idx = batch['outcome_idx']
        
        # Forward pass
        outputs = model(X, T, Y, claims, treatment_idx, outcome_idx)
        
        # Compute loss
        # Create claim mask
        claim_mask = torch.zeros(len(claims), max(len(c) for c in claims), dtype=torch.bool, device=device)
        for i, claim_list in enumerate(claims):
            claim_mask[i, len(claim_list):] = True
        
        loss, loss_dict = criterion(outputs, true_ate, true_adj, claims, claim_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'effect': f'{loss_dict["effect"]:.4f}',
        })
    
    n_batches = len(dataloader)
    return {
        'train_loss': total_loss / n_batches,
        'train_effect': loss_components['effect'] / n_batches,
        'train_graph': loss_components['graph'] / n_batches,
        'train_claim': loss_components['claim'] / n_batches,
        'train_acyclicity': loss_components['acyclicity'] / n_batches,
    }


def main():
    print("="*60)
    print("PoG-PFN Extended Training")
    print("="*60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    n_vars = 10
    n_epochs = 20
    batch_size = 8
    learning_rate = 1e-4
    
    print(f"Configuration:")
    print(f"  Variables: {n_vars}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Create model
    print("\nInitializing model...")
    model = PoGPFN(
        n_vars=n_vars,
        d_model=256,
        n_heads=4,
        n_layers=3,
        n_claim_layers=2
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = PoGPFNDataset(
        n_tasks=200,
        n_vars=n_vars,
        n_samples_per_task=500,
        n_claims_per_task=5,
        seed=42
    )
    
    val_dataset = PoGPFNDataset(
        n_tasks=50,
        n_vars=n_vars,
        n_samples_per_task=500,
        n_claims_per_task=5,
        seed=123
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Train size: {len(train_dataset)} tasks")
    print(f"Val size: {len(val_dataset)} tasks")
    
    # Create optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = PoGPFNLoss(
        weight_ate=1.0,
        weight_graph=0.5,
        weight_claim=0.3,
        weight_acyclicity=0.1
    )
    
    # Metrics tracker
    tracker = MetricsTracker()
    
    # Training loop
    print("\nStarting training...")
    print("-"*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, n_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch, n_epochs)
        
        # Validate every 5 epochs
        if epoch % 5 == 0 or epoch == n_epochs:
            val_metrics = evaluate(model, val_loader, criterion, device)
            print(f"\nEpoch {epoch} - Val Loss: {val_metrics['val_loss']:.4f}, Val MAE: {val_metrics['val_mae']:.4f}")
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                torch.save(model.state_dict(), 'results/best_model.pt')
                print(f"  → Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            val_metrics = {'val_loss': None, 'val_mae': None, 'val_crps': None}
        
        # Update metrics
        tracker.update(epoch, {**train_metrics, **val_metrics})
        
        # Step scheduler
        scheduler.step()
    
    print("\n" + "-"*60)
    print("✓ Training completed!")
    
    # Save and visualize
    print("\nSaving results...")
    tracker.save_history()
    tracker.plot_losses()
    tracker.plot_metrics()
    tracker.print_summary()
    
    print(f"Results saved to: results/")
    print(f"  - training_history.json")
    print(f"  - training_losses.png")
    print(f"  - validation_metrics.png")
    print(f"  - best_model.pt")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
