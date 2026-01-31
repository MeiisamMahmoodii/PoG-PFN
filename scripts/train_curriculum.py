"""
PoG-PFN Training with Curriculum Learning and Advanced Scheduling

Features:
- Curriculum learning: progressively increase task complexity
- Learning rate warmup + cosine annealing
- Adaptive loss weighting
- All hyperparameters configurable at top
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


class CurriculumConfig:
    """Full configuration with curriculum learning."""
    
    def __init__(self):
        # ==================== CURRICULUM LEARNING ====================
        self.use_curriculum = True
        self.curriculum_stages = [
            # Stage 1: Easy - more samples, simpler graphs (sparse)
            {'n_vars': 10, 'n_samples': 500, 'density': 0.2, 'epochs': 10},
            # Stage 2: Medium - same samples, denser graphs
            {'n_vars': 10, 'n_samples': 500, 'density': 0.3, 'epochs': 10},
            # Stage 3: Hard - full complexity (densest graphs)
            {'n_vars': 10, 'n_samples': 500, 'density': 0.4, 'epochs': 10},
        ]
        
        #==================== CLAIM CONFIGURATION ====================
        self.n_claims_per_task = 5
        self.truthful_ratio = 0.7  # 70% true claims
        self.false_ratio = 0.3     # 30% false claims
        
        # ==================== DATA CONFIGURATION ====================
        self.n_train_tasks = 200
        self.n_val_tasks = 50
        self.n_test_tasks = 100
        
        # ==================== MODEL CONFIGURATION ====================
        self.d_model = 256
        self.n_heads = 4
        self.n_layers = 3
        self.n_claim_layers = 2
        self.dropout = 0.1
        
        # ==================== TRAINING CONFIGURATION ====================
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.grad_clip_norm = 1.0
        
        # ==================== LEARNING RATE SCHEDULE ====================
        self.use_warmup = True
        self.warmup_epochs = 5
        self.min_lr = 1e-6
        self.scheduler_type = 'cosine'  # 'cosine', 'plateau', 'step'
        
        # ==================== ADAPTIVE LOSS WEIGHTS ====================
        self.adaptive_weights = True
        self.initial_weights = {
            'ate': 1.0,
            'graph': 0.5,
            'claim': 0.3,
            'identification': 0.2,
            'acyclicity': 0.1
        }
        # Increase claim weight over time to focus on claim discrimination
        self.final_weights = {
            'ate': 1.0,
            'graph': 0.3,
            'claim': 0.7,  # Increases
            'identification': 0.3,
            'acyclicity': 0.05
        }
        
        # ==================== VALIDATION ====================
        self.validate_every = 5
        
        # ==================== SAVING ====================
        self.save_dir = Path('results_curriculum')
        self.save_checkpoints = True
        
        # ==================== DEVICE ====================
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # ==================== SEEDS ====================
        self.train_seed = 42
        self.val_seed = 123
        self.test_seed = 999
    
    def get_curriculum_params(self, stage_idx: int):
        """Get parameters for curriculum stage."""
        if not self.use_curriculum or stage_idx >= len(self.curriculum_stages):
            # Default to hardest stage
            return self.curriculum_stages[-1] if self.use_curriculum else {
                'n_vars': 10,
                'n_samples': 500,
                'density': 0.4,
                'epochs': 30
            }
        return self.curriculum_stages[stage_idx]
    
    def get_loss_weights(self, progress: float):
        """Get interpolated loss weights based on training progress [0,1]."""
        if not self.adaptive_weights:
            return self.initial_weights
        
        weights = {}
        for key in self.initial_weights:
            initial = self.initial_weights[key]
            final = self.final_weights[key]
            weights[key] = initial + (final - initial) * progress
        return weights
    
    def print_config(self, stage_idx=None):
        """Print configuration."""
        print("\n" + "="*70)
        print("TRAINING CONFIGURATION")
        if self.use_curriculum and stage_idx is not None:
            stage = self.curriculum_stages[stage_idx]
            print(f" - Curriculum Stage {stage_idx + 1}/{len(self.curriculum_stages)}")
        print("="*70)
        
        print(f"\nCurriculum Learning: {'ENABLED' if self.use_curriculum else 'DISABLED'}")
        if self.use_curriculum:
            print(f"  Stages: {len(self.curriculum_stages)}")
            if stage_idx is not None:
                stage = self.curriculum_stages[stage_idx]
                print(f"  Current: n_vars={stage['n_vars']}, " +
                      f"n_samples={stage['n_samples']}, " +
                      f"density={stage['density']}, " +
                      f"epochs={stage['epochs']}")
        
        print(f"\nClaims:")
        print(f"  Per task: {self.n_claims_per_task}")
        print(f"  Truthful: {self.truthful_ratio*100:.0f}%")
        print(f"  False: {self.false_ratio*100:.0f}%")
        
        print(f"\nModel: d_model={self.d_model}, n_heads={self.n_heads}, " +
              f"n_layers={self.n_layers}")
        print(f"Learning Rate: {self.learning_rate} (warmup: {self.use_warmup})")
        print(f"Adaptive Weights: {self.adaptive_weights}")
        print(f"Device: {self.device}")
        print("="*70 + "\n")


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


def create_dataset(config, stage_params, split='train'):
    """Create dataset for current curriculum stage."""
    if split == 'train':
        seed = config.train_seed
        n_tasks = config.n_train_tasks
    elif split == 'val':
        seed = config.val_seed
        n_tasks = config.n_val_tasks
    else:  # test
        seed = config.test_seed
        n_tasks = config.n_test_tasks
    
    claim_config = {
        'truthful_ratio': config.truthful_ratio,
        'false_ratio': config.false_ratio,
        'unidentifiable_ratio': 0.0,
        'conflicting_ratio': 0.0
    }
    
    return PoGPFNDataset(
        n_tasks=n_tasks,
        n_vars=stage_params['n_vars'],
        n_samples_per_task=stage_params['n_samples'],
        n_claims_per_task=config.n_claims_per_task,
        seed=seed,
        claim_config=claim_config
    )


def train_epoch(model, dataloader, optimizer, criterion, device, config):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    
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
        
        outputs = model(X, T, Y, claims, treatment_idx, outcome_idx)
        
        claim_mask = torch.zeros(len(claims), max(len(c) for c in claims), 
                                dtype=torch.bool, device=device)
        for i, claim_list in enumerate(claims):
            claim_mask[i, len(claim_list):] = True
        
        loss, _ = criterion(outputs, true_ate, true_adj, claims, claim_mask)
        
        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    all_preds, all_trues = [], []
    
    with torch.no_grad():
        for batch in dataloader:
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
    return torch.abs(preds - trues).mean().item()


def main():
    config = CurriculumConfig()
    
    # You can override parameters here:
    # config.use_curriculum = False
    # config.batch_size = 16
    # etc.
    
    config.save_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device(config.device)
    
    # Determine total training plan
    if config.use_curriculum:
        stages = config.curriculum_stages
        total_epochs = sum(s['epochs'] for s in stages)
        n_stages = len(stages)
    else:
        stages = [config.get_curriculum_params(999)]  # Get default
        total_epochs = stages[0]['epochs']
        n_stages = 1
    
    config.print_config(0 if config.use_curriculum else None)
    
    # Initialize model with maximum n_vars
    max_n_vars = max(s['n_vars'] for s in stages)
    model = PoGPFN(
        n_vars=max_n_vars,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        n_claim_layers=config.n_claim_layers,
        dropout=config.dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=total_epochs,
        min_lr=config.min_lr,
        base_lr=config.learning_rate
    )
    
    # Training history
    history = {'epoch': [], 'stage': [], 'train_loss': [], 'val_mae': [], 'lr': []}
    best_val_mae = float('inf')
    global_epoch = 0
    
    print("Starting curriculum training...")
    print("="*70 + "\n")
    
    # Train through curriculum stages
    for stage_idx, stage_params in enumerate(stages):
        if config.use_curriculum:
            print(f"\n{'='*70}")
            print(f"CURRICULUM STAGE {stage_idx + 1}/{n_stages}")
            print(f"n_vars={stage_params['n_vars']}, " +
                  f"n_samples={stage_params['n_samples']}, " +
                  f"density={stage_params['density']}")
            print(f"{'='*70}\n")
        
        # Create datasets for this stage
        train_dataset = create_dataset(config, stage_params, 'train')
        val_dataset = create_dataset(config, stage_params, 'val')
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                 shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                               shuffle=False, collate_fn=collate_fn)
        
        # Train this stage
        for epoch in range(stage_params['epochs']):
            global_epoch += 1
            
            # Update loss weights based on progress
            progress = global_epoch / total_epochs
            weights = config.get_loss_weights(progress)
            
            criterion = PoGPFNLoss(
                weight_ate=weights['ate'],
                weight_graph=weights['graph'],
                weight_claim=weights['claim'],
                weight_identification=weights['identification'],
                weight_acyclicity=weights['acyclicity']
            )
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config)
            
            # LR step
            lr = scheduler.step()
            
            # Validate
            val_mae = None
            if global_epoch % config.validate_every == 0 or global_epoch == total_epochs:
                val_mae = evaluate(model, val_loader, device)
                print(f"Epoch {global_epoch}/{total_epochs} | " +
                      f"Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | LR: {lr:.2e}")
                
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    if config.save_checkpoints:
                        torch.save(model.state_dict(), config.save_dir / 'best_model.pt')
                        print(f"  → Saved best model (MAE: {best_val_mae:.4f})")
            
            # Record
            history['epoch'].append(global_epoch)
            history['stage'].append(stage_idx)
            history['train_loss'].append(train_loss)
            history['val_mae'].append(val_mae)
            history['lr'].append(lr)
    
    # Final test
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    test_dataset = create_dataset(config, stages[-1], 'test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn)
    
    if config.save_checkpoints:
        model.load_state_dict(torch.load(config.save_dir / 'best_model.pt'))
    
    test_mae = evaluate(model, test_loader, device)
    print(f"\nTest MAE: {test_mae:.4f}")
    print(f"Best Val MAE: {best_val_mae:.4f}")
    
    # Save history
    with open(config.save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nResults saved to: {config.save_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
