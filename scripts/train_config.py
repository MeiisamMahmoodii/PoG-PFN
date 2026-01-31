"""
Configurable PoG-PFN Training Script

Fully configurable training script with hyperparameter controls,
comprehensive logging, evaluation, and visualization.

All parameters can be modified at the top of main().
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pog_pfn.models.pog_pfn import PoGPFN
from pog_pfn.data.dataset import PoGPFNDataset, collate_fn
from pog_pfn.training.losses import PoGPFNLoss


class TrainingConfig:
    """Configuration for training hyperparameters."""
    
    def __init__(self):
        # ==================== DATA CONFIGURATION ====================
        self.n_vars = 10  # 20 vars with 2000 samples = 40K seq length (TOO LARGE!)
        self.n_train_tasks = 200
        self.n_val_tasks = 50
        self.n_test_tasks = 100
        self.n_samples_per_task = 500  # 2000 creates 40K sequence (OOM!)
        self.n_claims_per_task = 5
        
        # ==================== MODEL CONFIGURATION ====================
        self.d_model = 256
        self.n_heads = 4
        self.n_layers = 3  # Deep models need more memory
        self.n_claim_layers = 2
        self.dropout = 0.1
        
        # ==================== TRAINING CONFIGURATION ====================
        self.n_epochs = 30
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.grad_clip_norm = 1.0
        
        # Learning rate scheduling
        self.use_scheduler = True
        self.scheduler_type = 'cosine'  # 'cosine', 'step', 'plateau'
        self.warmup_epochs = 3
        
        # ==================== LOSS WEIGHTS ====================
        self.weight_ate = 1.0
        self.weight_graph = 0.5
        self.weight_claim = 0.3
        self.weight_identification = 0.2
        self.weight_acyclicity = 0.1
        
        # ==================== EVALUATION ====================
        self.validate_every = 5
        self.eval_during_training = True
        
        # ==================== LOGGING & SAVING ====================
        self.save_dir = Path('results')
        self.save_checkpoints = True
        self.save_best_only = True
        self.log_interval = 10  # batches
        
        # ==================== DEVICE ====================
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # ==================== RANDOM SEEDS ====================
        self.train_seed = 42
        self.val_seed = 123
        self.test_seed = 999
    
    def print_config(self):
        """Print configuration."""
        print("\n" + "="*70)
        print("TRAINING CONFIGURATION")
        print("="*70)
        print(f"\nData:")
        print(f"  Variables: {self.n_vars}")
        print(f"  Train tasks: {self.n_train_tasks}")
        print(f"  Val tasks: {self.n_val_tasks}")
        print(f"  Test tasks: {self.n_test_tasks}")
        print(f"  Samples/task: {self.n_samples_per_task}")
        print(f"  Claims/task: {self.n_claims_per_task}")
        
        print(f"\nModel:")
        print(f"  d_model: {self.d_model}")
        print(f"  n_heads: {self.n_heads}")
        print(f"  n_layers: {self.n_layers}")
        print(f"  n_claim_layers: {self.n_claim_layers}")
        print(f"  dropout: {self.dropout}")
        
        print(f"\nTraining:")
        print(f"  Epochs: {self.n_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Weight decay: {self.weight_decay}")
        print(f"  Gradient clipping: {self.grad_clip_norm}")
        print(f"  Scheduler: {self.scheduler_type if self.use_scheduler else 'None'}")
        
        print(f"\nLoss Weights:")
        print(f"  ATE: {self.weight_ate}")
        print(f"  Graph: {self.weight_graph}")
        print(f"  Claim: {self.weight_claim}")
        print(f"  Identification: {self.weight_identification}")
        print(f"  Acyclicity: {self.weight_acyclicity}")
        
        print(f"\nDevice: {self.device}")
        print("="*70 + "\n")


class MetricsTracker:
    """Track and save training metrics."""
    
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_effect': [],
            'train_graph': [],
            'train_claim': [],
            'val_loss': [],
            'val_mae': [],
            'test_mae': [],
        }
    
    def update(self, epoch, metrics):
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def save_json(self):
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total loss
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], label='Train')
        valid_val = [(e, v) for e, v in zip(self.history['epoch'], self.history['val_loss']) if v is not None]
        if valid_val:
            epochs, vals = zip(*valid_val)
            axes[0, 0].plot(epochs, vals, label='Val', marker='o')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Effect loss
        axes[0, 1].plot(self.history['epoch'], self.history['train_effect'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('CRPS Loss')
        axes[0, 1].set_title('Effect Loss (CRPS)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Graph loss
        axes[1, 0].plot(self.history['epoch'], self.history['train_graph'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Graph Posterior Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # MAE
        valid_val_mae = [(e, v) for e, v in zip(self.history['epoch'], self.history['val_mae']) if v is not None]
        valid_test_mae = [(e, v) for e, v in zip(self.history['epoch'], self.history['test_mae']) if v is not None]
        if valid_val_mae:
            epochs, vals = zip(*valid_val_mae)
            axes[1, 1].plot(epochs, vals, label='Val MAE', marker='o')
        if valid_test_mae:
            epochs, vals = zip(*valid_test_mae)
            axes[1, 1].plot(epochs, vals, label='Test MAE', marker='s')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('Mean Absolute Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150)
        plt.close()


def train_epoch(model, dataloader, optimizer, criterion, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {'effect': 0.0, 'graph': 0.0, 'claim': 0.0}
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        X = batch['X'].to(device)
        T = batch['T'].to(device)
        Y = batch['Y'].to(device)
        claims = batch['claims']
        true_ate = batch['true_ate'].to(device)
        true_adj = batch['true_adjacency'].to(device)
        treatment_idx = batch['treatment_idx']
        outcome_idx = batch['outcome_idx']
        
        # Forward
        outputs = model(X, T, Y, claims, treatment_idx, outcome_idx)
        
        # Loss
        claim_mask = torch.zeros(len(claims), max(len(c) for c in claims), 
                                dtype=torch.bool, device=device)
        for i, claim_list in enumerate(claims):
            claim_mask[i, len(claim_list):] = True
        
        loss, loss_dict = criterion(outputs, true_ate, true_adj, claims, claim_mask)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        
        # Track
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    n_batches = len(dataloader)
    return {
        'train_loss': total_loss / n_batches,
        'train_effect': loss_components['effect'] / n_batches,
        'train_graph': loss_components['graph'] / n_batches,
        'train_claim': loss_components['claim'] / n_batches,
    }


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            X = batch['X'].to(device)
            T = batch['T'].to(device)
            Y = batch['Y'].to(device)
            claims = batch['claims']
            true_ate = batch['true_ate'].to(device)
            treatment_idx = batch['treatment_idx']
            outcome_idx = batch['outcome_idx']
            
            outputs = model(X, T, Y, claims, treatment_idx, outcome_idx)
            
            all_preds.append(outputs['ate_mean'])
            all_trues.append(true_ate)
    
    preds = torch.cat(all_preds)
    trues = torch.cat(all_trues)
    
    mae = torch.abs(preds - trues).mean().item()
    return {'mae': mae}


def main():
    # ========================================================================
    # CONFIGURATION - MODIFY THESE VALUES TO CHANGE HYPERPARAMETERS
    # ========================================================================
    config = TrainingConfig()
    
    # You can override any parameter here:
    # config.n_epochs = 50
    # config.learning_rate = 5e-5
    # config.batch_size = 16
    # config.weight_claim = 0.5
    # ... etc
    
    config.print_config()
    
    # ========================================================================
    # SETUP
    # ========================================================================
    device = torch.device(config.device)
    
    # Create model
    print("Initializing model...")
    # Calculate max sequence length: n_samples * n_vars
    max_seq_len = config.n_samples_per_task * config.n_vars
    
    model = PoGPFN(
        n_vars=config.n_vars,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        n_claim_layers=config.n_claim_layers,
        dropout=config.dropout,
        max_seq_len=max_seq_len  # Pass calculated buffer size (2000*20=40000)
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = PoGPFNDataset(
        n_tasks=config.n_train_tasks,
        n_vars=config.n_vars,
        n_samples_per_task=config.n_samples_per_task,
        n_claims_per_task=config.n_claims_per_task,
        seed=config.train_seed
    )
    
    val_dataset = PoGPFNDataset(
        n_tasks=config.n_val_tasks,
        n_vars=config.n_vars,
        n_samples_per_task=config.n_samples_per_task,
        n_claims_per_task=config.n_claims_per_task,
        seed=config.val_seed
    )
    
    test_dataset = PoGPFNDataset(
        n_tasks=config.n_test_tasks,
        n_vars=config.n_vars,
        n_samples_per_task=config.n_samples_per_task,
        n_claims_per_task=config.n_claims_per_task,
        seed=config.test_seed
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn)
    
    print(f"Train: {len(train_dataset)} tasks")
    print(f"Val: {len(val_dataset)} tasks")
    print(f"Test: {len(test_dataset)} tasks\n")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    if config.use_scheduler:
        if config.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.n_epochs
            )
        elif config.scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.5
            )
        elif config.scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
    
    # Loss function
    criterion = PoGPFNLoss(
        weight_ate=config.weight_ate,
        weight_graph=config.weight_graph,
        weight_claim=config.weight_claim,
        weight_identification=config.weight_identification,
        weight_acyclicity=config.weight_acyclicity
    )
    
    # Metrics tracker
    tracker = MetricsTracker(config.save_dir)
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    print("Starting training...")
    print("="*70 + "\n")
    
    best_val_mae = float('inf')
    
    for epoch in range(1, config.n_epochs + 1):
        print(f"Epoch {epoch}/{config.n_epochs}")
        print("-"*70)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, config)
        print(f"Train Loss: {train_metrics['train_loss']:.4f} | " +
              f"Effect: {train_metrics['train_effect']:.4f} | " +
              f"Graph: {train_metrics['train_graph']:.4f}")
        
        # Validate
        val_metrics = {'val_loss': None, 'val_mae': None}
        test_metrics = {'test_mae': None}
        
        if epoch % config.validate_every == 0 or epoch == config.n_epochs:
            val_metrics = evaluate(model, val_loader, device)
            test_metrics = evaluate(model, test_loader, device)
            
            print(f"Val MAE: {val_metrics['mae']:.4f} | Test MAE: {test_metrics['mae']:.4f}")
            
            val_metrics['val_mae'] = val_metrics.pop('mae')
            test_metrics['test_mae'] = test_metrics.pop('mae')
            
            # Save best
            if val_metrics['val_mae'] < best_val_mae:
                best_val_mae = val_metrics['val_mae']
                if config.save_checkpoints:
                    torch.save(model.state_dict(), config.save_dir / 'best_model.pt')
                    print(f"→ Saved best model (val_mae: {best_val_mae:.4f})")
        
        # Update metrics
        tracker.update(epoch, {**train_metrics, **val_metrics, **test_metrics})
        
        # Step scheduler
        if config.use_scheduler:
            if config.scheduler_type == 'plateau' and val_metrics['val_mae'] is not None:
                scheduler.step(val_metrics['val_mae'])
            else:
                scheduler.step()
        
        print()
    
    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    print("="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    # Load best model
    if config.save_checkpoints and (config.save_dir / 'best_model.pt').exists():
        model.load_state_dict(torch.load(config.save_dir / 'best_model.pt'))
        print("Loaded best model\n")
    
    final_val = evaluate(model, val_loader, device)
    final_test = evaluate(model, test_loader, device)
    
    print(f"Final Val MAE: {final_val['mae']:.4f}")
    print(f"Final Test MAE: {final_test['mae']:.4f}")
    print(f"Best Val MAE: {best_val_mae:.4f}")
    
    # Save results
    tracker.save_json()
    tracker.plot_training()
    
    print(f"\nResults saved to: {config.save_dir}/")
    print("  - training_history.json")
    print("  - training_curves.png")
    print("  - best_model.pt")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
