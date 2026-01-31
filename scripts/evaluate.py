"""
Evaluation Script with Calibration Plots and CRPS Metrics

Evaluates trained PoG-PFN model with comprehensive metrics and visualizations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pog_pfn.models.pog_pfn import PoGPFN
from pog_pfn.data.dataset import PoGPFNDataset, collate_fn


def compute_crps(predictions_mean, predictions_std, targets):
    """Compute CRPS for Gaussian predictions."""
    error = torch.abs(predictions_mean - targets)
    z = (targets - predictions_mean) / (predictions_std + 1e-8)
    z = z.clamp(min=-10.0, max=10.0)
    
    normal = torch.distributions.Normal(0, 1)
    phi = torch.exp(normal.log_prob(z))
    Phi = normal.cdf(z)
    
    crps = error + predictions_std * (2 * phi + z * (2 * Phi - 1) - 1 / (torch.pi ** 0.5))
    return crps.mean().item()


def compute_calibration(predictions_mean, predictions_std, targets, n_bins=10):
    """Compute calibration curve for uncertainty estimates."""
    # Convert to numpy
    pred_mean = predictions_mean.cpu().numpy()
    pred_std = predictions_std.cpu().numpy()
    true_ate = targets.cpu().numpy()
    
    # Compute z-scores
    z_scores = (true_ate - pred_mean) / (pred_std + 1e-8)
    
    # Expected vs observed coverage for different confidence levels
    confidence_levels = np.linspace(0.1, 0.9, n_bins)
    expected_coverage = []
    observed_coverage = []
    
    for alpha in confidence_levels:
        # Expected: alpha fraction should fall within +/- Phi^{-1}(alpha/2 + 0.5) std
        z_critical = torch.distributions.Normal(0, 1).icdf(torch.tensor(alpha/2 + 0.5)).item()
        
        # Observed: fraction actually within this range
        in_range = np.abs(z_scores) <= z_critical
        observed = in_range.mean()
        
        expected_coverage.append(alpha)
        observed_coverage.append(observed)
    
    return np.array(expected_coverage), np.array(observed_coverage), z_scores


def plot_calibration(expected, observed, save_path):
    """Plot calibration curve."""
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2, alpha=0.7)
    plt.plot(expected, observed, 'o-', label='Model calibration', linewidth=2, markersize=8)
    plt.xlabel('Expected Coverage', fontsize=12)
    plt.ylabel('Observed Coverage', fontsize=12)
    plt.title('Calibration Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_prediction_intervals(predictions_mean, predictions_std, targets, save_path, n_show=50):
    """Plot prediction intervals."""
    pred_mean = predictions_mean[:n_show].cpu().numpy()
    pred_std = predictions_std[:n_show].cpu().numpy()
    true_ate = targets[:n_show].cpu().numpy()
    
    indices = np.arange(n_show)
    
    plt.figure(figsize=(14, 6))
    
    # Plot predictions with uncertainty
    plt.fill_between(indices, pred_mean - 2*pred_std, pred_mean + 2*pred_std, 
                     alpha=0.3, label='95% CI', color='blue')
    plt.fill_between(indices, pred_mean - pred_std, pred_mean + pred_std, 
                     alpha=0.5, label='68% CI', color='blue')
    plt.plot(indices, pred_mean, 'b-', label='Predicted mean', linewidth=2)
    plt.plot(indices, true_ate, 'ro', label='True ATE', markersize=6, alpha=0.7)
    
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('ATE', fontsize=12)
    plt.title('Prediction Intervals', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_error_distribution(predictions_mean, targets, save_path):
    """Plot error distribution."""
    errors = (predictions_mean - targets).cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(errors, bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    axes[0].set_xlabel('Prediction Error', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_model(model, dataloader, device, save_dir='results'):
    """Run comprehensive evaluation."""
    model.eval()
    
    all_pred_means = []
    all_pred_stds = []
    all_true_ates = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch in dataloader:
            X = batch['X'].to(device)
            T = batch['T'].to(device)
            Y = batch['Y'].to(device)
            claims = batch['claims']
            true_ate = batch['true_ate'].to(device)
            treatment_idx = batch['treatment_idx']
            outcome_idx = batch['outcome_idx']
            
            # Forward pass
            outputs = model(X, T, Y, claims, treatment_idx, outcome_idx, return_uncertainty=True)
            
            all_pred_means.append(outputs['ate_mean'])
            all_pred_stds.append(outputs['ate_std'])
            all_true_ates.append(true_ate)
    
    # Concatenate results
    pred_means = torch.cat(all_pred_means)
    pred_stds = torch.cat(all_pred_stds)
    true_ates = torch.cat(all_true_ates)
    
    # Compute metrics
    mae = torch.abs(pred_means - true_ates).mean().item()
    rmse = torch.sqrt(((pred_means - true_ates) ** 2).mean()).item()
    crps = compute_crps(pred_means, pred_stds, true_ates)
    
    # Calibration
    expected, observed, z_scores = compute_calibration(pred_means, pred_stds, true_ates)
    calibration_error = np.abs(expected - observed).mean()
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"CRPS: {crps:.4f}")
    print(f"Calibration Error: {calibration_error:.4f}")
    print(f"Mean Predicted Std: {pred_stds.mean().item():.4f}")
    print("="*60 + "\n")
    
    # Generate plots
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    print("Generating plots...")
    plot_calibration(expected, observed, save_dir / 'calibration_plot.png')
    plot_prediction_intervals(pred_means, pred_stds, true_ates, save_dir / 'prediction_intervals.png')
    plot_error_distribution(pred_means, true_ates, save_dir / 'error_distribution.png')
    
    print(f"Plots saved to: {save_dir}/")
    print("  - calibration_plot.png")
    print("  - prediction_intervals.png")
    print("  - error_distribution.png")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'crps': crps,
        'calibration_error': calibration_error,
        'z_scores': z_scores,
    }


def main():
    print("="*60)
    print("PoG-PFN Model Evaluation")
    print("="*60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    n_vars = 10
    
    # Load model
    print("\nLoading model...")
    model = PoGPFN(
        n_vars=n_vars,
        d_model=256,
        n_heads=4,
        n_layers=3,
        n_claim_layers=2
    ).to(device)
    
    model.load_state_dict(torch.load('results/best_model.pt', map_location=device))
    print("✓ Model loaded successfully")
    
    # Create test dataset
    print("\nCreating test dataset...")
    test_dataset = PoGPFNDataset(
        n_tasks=100,
        n_vars=n_vars,
        n_samples_per_task=500,
        n_claims_per_task=5,
        seed=999  # Different seed for test
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Test size: {len(test_dataset)} tasks")
    
    # Run evaluation
    metrics = evaluate_model(model, test_loader, device)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
