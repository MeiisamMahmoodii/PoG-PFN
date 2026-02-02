"""
Multi-GPU Distributed Training Script for PoG-PFN

Supports training on multiple GPUs using PyTorch DistributedDataParallel (DDP).
Optimized for 4x A100 GPUs but works with any number of GPUs.

Usage:
    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 scripts/train_distributed.py
    
    # Or use the wrapper script
    bash scripts/launch_distributed.sh
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
import os
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pog_pfn.models.pog_pfn import PoGPFN
from pog_pfn.data.dataset import PoGPFNDataset, collate_fn
from pog_pfn.training.losses import PoGPFNLoss


class DistributedTrainingConfig:
    """Configuration for distributed training."""
    
    def __init__(self):
        # ==================== DISTRIBUTED SETUP ====================
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        
        # ==================== DATA CONFIGURATION ====================
        self.n_vars = 10  # 20 vars causes huge seq_len = 10K
        self.n_train_tasks = 1000  # Will be split across GPUs
        self.n_val_tasks = 200
        self.n_test_tasks = 200
        self.n_samples_per_task = 500  # 10 vars × 500 = 5K seq_len (manageable)
        self.n_claims_per_task = 5
        
        # ==================== MODEL CONFIGURATION ====================
        self.d_model = 256
        self.n_heads = 8
        self.n_layers = 8  # 12 layers is too heavy for this batch size
        self.n_claim_layers = 4
        self.dropout = 0.1
        
        # ==================== TRAINING CONFIGURATION ====================
        self.n_epochs = 100
        self.batch_size = 8  # Per GPU batch size (total = 8 × 4 = 32)
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.grad_clip_norm = 1.0
        
        # Learning rate scaling for multi-GPU
        # Linear scaling rule: lr *= world_size
        if self.world_size > 1:
            self.learning_rate *= self.world_size
        
        # Learning rate scheduling
        self.use_scheduler = True
        self.scheduler_type = 'cosine'
        self.warmup_epochs = 5
        
        # ==================== LOSS WEIGHTS ====================
        # Using only MAE loss - other losses have numerical stability bugs
        self.weight_ate = 1.0          # MAE loss - stable
        self.weight_graph = 0.0        # Disabled - uses CRPS internally
        self.weight_claim = 0.0        # Disabled - uses CRPS internally
        self.weight_identification = 0.0  # Disabled - uses CRPS internally
        self.weight_acyclicity = 0.0   # Disabled - BCE assertion failures.1
        
        # ==================== EVALUATION ====================
        self.validate_every = 5
        self.eval_during_training = True
        
        # ==================== LOGGING & SAVING ====================
        self.save_dir = Path('results_distributed')
        self.save_checkpoints = True
        self.save_best_only = True
        self.log_interval = 10
        
        # ==================== RANDOM SEEDS ====================
        self.train_seed = 42
        self.val_seed = 123
        self.test_seed = 999
    
    def print_config(self):
        """Print configuration (only on rank 0)."""
        if self.rank != 0:
            return
            
        print("\n" + "="*70)
        print("DISTRIBUTED TRAINING CONFIGURATION")
        print("="*70)
        print(f"\nDistributed:")
        print(f"  World size: {self.world_size}")
        print(f"  GPUs: {self.world_size}")
        print(f"  Effective batch size: {self.batch_size * self.world_size}")
        
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
        print(f"  Batch size per GPU: {self.batch_size}")
        print(f"  Learning rate (scaled): {self.learning_rate}")
        print(f"  Weight decay: {self.weight_decay}")
        print(f"  Gradient clipping: {self.grad_clip_norm}")
        print(f"  Scheduler: {self.scheduler_type if self.use_scheduler else 'None'}")
        
        print(f"\nLoss Weights:")
        print(f"  ATE: {self.weight_ate}")
        print(f"  Graph: {self.weight_graph}")
        print(f"  Claim: {self.weight_claim}")
        print(f"  Identification: {self.weight_identification}")
        print(f"  Acyclicity: {self.weight_acyclicity}")
        
        print("="*70 + "\n")


def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        n_gpus = torch.cuda.device_count()
        # CRITICAL: set_device MUST happen BEFORE init_process_group for NCCL
        # Use modulo to handle cases where processes only see a subset of GPUs
        torch.cuda.set_device(local_rank % n_gpus)
        dist.init_process_group(backend='nccl')
        return True
    return False


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train_epoch(model, dataloader, optimizer, criterion, device, config, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {'effect': 0.0, 'graph': 0.0, 'claim': 0.0}
    
    # Only show progress bar on rank 0
    if config.rank == 0:
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    else:
        pbar = dataloader
    
    for batch_idx, batch in enumerate(pbar):
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
        
        # Check for NaNs in outputs
        if torch.isnan(outputs['ate_mean']).any():
            print(f"[Rank {config.rank}] CRITICAL: NaN detected in ate_mean at batch {batch_idx}")
            # Try to catch it early
            if torch.isnan(X).any() or torch.isnan(T).any() or torch.isnan(Y).any():
                print(f"[Rank {config.rank}] NaN detected in input data (X/T/Y)")
            if torch.isnan(true_ate).any():
                print(f"[Rank {config.rank}] NaN detected in true_ate")
        
        # Loss
        claim_mask = torch.zeros(len(claims), max(len(c) for c in claims), 
                                dtype=torch.bool, device=device)
        for i, claim_list in enumerate(claims):
            claim_mask[i, :len(claim_list)] = False
            claim_mask[i, len(claim_list):] = True
        
        loss, loss_dict = criterion(outputs, true_ate, true_adj, claims, claim_mask)
        
        # Backward
        optimizer.zero_grad()
        
        if torch.isnan(loss):
            print(f"[Rank {config.rank}] CRITICAL: Loss is NaN at batch {batch_idx}")
            # Skip this batch to prevent model corruption
            continue
            
        loss.backward()
        
        if config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        
        optimizer.step()
        
        # Track
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        
        if config.rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    n_batches = len(dataloader)
    return {
        'train_loss': total_loss / n_batches,
        'train_effect': loss_components['effect'] / n_batches,
        'train_graph': loss_components['graph'] / n_batches,
        'train_claim': loss_components['claim'] / n_batches,
    }


def evaluate(model, dataloader, device, config):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_trues = []
    
    # Only show progress bar on rank 0
    if config.rank == 0:
        pbar = tqdm(dataloader, desc='Evaluating')
    else:
        pbar = dataloader
    
    with torch.no_grad():
        for batch in pbar:
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
    
    # Gather results from all GPUs
    if dist.is_initialized():
        all_preds_gathered = [torch.zeros_like(preds) for _ in range(config.world_size)]
        all_trues_gathered = [torch.zeros_like(trues) for _ in range(config.world_size)]
        dist.all_gather(all_preds_gathered, preds)
        dist.all_gather(all_trues_gathered, trues)
        
        if config.rank == 0:
            preds = torch.cat(all_preds_gathered)
            trues = torch.cat(all_trues_gathered)
    
    if config.rank == 0:
        mae = torch.abs(preds - trues).mean().item()
        return {'mae': mae}
    else:
        return {'mae': None}


def main():
    # Setup distributed training
    is_distributed = setup_distributed()
    
    config = DistributedTrainingConfig()
    config.print_config()
    
    # Device setup
    n_gpus = torch.cuda.device_count()
    device = torch.device(f'cuda:{config.local_rank % n_gpus}')
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.train_seed + config.rank)
    np.random.seed(config.train_seed + config.rank)
    
    # ========================================================================
    # MODEL SETUP
    # ========================================================================
    if config.rank == 0:
        print("Initializing model...")
    
    max_seq_len = config.n_samples_per_task * config.n_vars
    
    model = PoGPFN(
        n_vars=config.n_vars,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        n_claim_layers=config.n_claim_layers,
        dropout=config.dropout,
        max_seq_len=max_seq_len
    ).to(device)
    
    # Wrap model with DDP
    if is_distributed:
        model = DDP(model, device_ids=[config.local_rank % n_gpus], find_unused_parameters=True)
    
    if config.rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}\n")
    
    # ========================================================================
    # DATA SETUP
    # ========================================================================
    if config.rank == 0:
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
    
    # Distributed samplers
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=config.train_seed
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=False
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        sampler=test_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    if config.rank == 0:
        print(f"Train: {len(train_dataset)} tasks")
        print(f"Val: {len(val_dataset)} tasks")
        print(f"Test: {len(test_dataset)} tasks\n")
    
    # ========================================================================
    # OPTIMIZER & SCHEDULER
    # ========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    if config.use_scheduler and config.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.n_epochs
        )
    else:
        scheduler = None
    
    criterion = PoGPFNLoss(
        weight_ate=config.weight_ate,
        weight_graph=config.weight_graph,
        weight_claim=config.weight_claim,
        weight_identification=config.weight_identification,
        weight_acyclicity=config.weight_acyclicity
    )
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    if config.rank == 0:
        print("Starting distributed training...")
        print("="*70 + "\n")
        config.save_dir.mkdir(exist_ok=True, parents=True)
    
    best_val_mae = float('inf')
    
    for epoch in range(1, config.n_epochs + 1):
        # Set epoch for distributed sampler
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        if config.rank == 0:
            print(f"Epoch {epoch}/{config.n_epochs}")
            print("-"*70)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, config, epoch)
        
        if config.rank == 0:
            print(f"Train Loss: {train_metrics['train_loss']:.4f} | " +
                  f"Effect: {train_metrics['train_effect']:.4f} | " +
                  f"Graph: {train_metrics['train_graph']:.4f}")
        
        # Validate
        if epoch % config.validate_every == 0 or epoch == config.n_epochs:
            val_metrics = evaluate(model, val_loader, device, config)
            test_metrics = evaluate(model, test_loader, device, config)
            
            if config.rank == 0 and val_metrics['mae'] is not None:
                print(f"Val MAE: {val_metrics['mae']:.4f} | Test MAE: {test_metrics['mae']:.4f}")
                
                # Save best model (only rank 0)
                if val_metrics['mae'] < best_val_mae:
                    best_val_mae = val_metrics['mae']
                    if config.save_checkpoints:
                        # Save DDP model properly
                        model_to_save = model.module if is_distributed else model
                        torch.save(model_to_save.state_dict(), 
                                 config.save_dir / 'best_model.pt')
                        print(f"→ Saved best model (val_mae: {best_val_mae:.4f})")
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        if config.rank == 0:
            print()
    
    # ========================================================================
    # FINAL EVALUATION
    # ========================================================================
    if config.rank == 0:
        print("="*70)
        print("FINAL EVALUATION")
        print("="*70)
    
    # All ranks load the best model for final evaluation
    if config.save_checkpoints and (config.save_dir / 'best_model.pt').exists():
        model_to_load = model.module if is_distributed else model
        # Load weights on all ranks to ensure consistent evaluation
        state_dict = torch.load(config.save_dir / 'best_model.pt', map_location=device)
        model_to_load.load_state_dict(state_dict)
        if config.rank == 0:
            print("Loaded best model\n")
    
    # All ranks MUST participate in collective evaluation
    final_val = evaluate(model, val_loader, device, config)
    final_test = evaluate(model, test_loader, device, config)
    
    if config.rank == 0:
        print(f"Final Val MAE: {final_val['mae']:.4f}")
        print(f"Final Test MAE: {final_test['mae']:.4f}")
        print(f"Best Val MAE: {best_val_mae:.4f}")
        
        print("\n" + "="*70)
        print("DISTRIBUTED TRAINING COMPLETE!")
        print("="*70)
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
