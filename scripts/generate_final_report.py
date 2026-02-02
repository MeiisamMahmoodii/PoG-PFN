"""
Final Report Generation Script for PoG-PFN

Runs comprehensive evaluation across different scenarios:
1. Standard Test Set (same distribution as train)
2. Out-of-Distribution (OOD) - Novel Mechanism Types
3. OOD - Higher Graph Density
4. Scaling - Variable number of variables (if applicable)

Generates a summary report and saves metrics.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys

sys.path.append(str(Path(__file__).parent.parent))

from pog_pfn.models.pog_pfn import PoGPFN
from pog_pfn.data.dataset import PoGPFNDataset, collate_fn, GraphType, MechanismType
from scripts.evaluate import evaluate_model


def run_scenario(model, device, name, dataset_params, save_dir):
    """Run evaluation for a specific scenario."""
    print(f"\n>>> Running Scenario: {name}")
    print("-" * 50)
    
    dataset = PoGPFNDataset(**dataset_params)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    scenario_dir = save_dir / name.lower().replace(" ", "_")
    metrics = evaluate_model(model, loader, device, save_dir=scenario_dir)
    return metrics


def main():
    # Configuration (matching DistributedTrainingConfig defaults)
    n_vars = 10
    d_model = 256
    n_heads = 8
    n_layers = 8
    n_claim_layers = 4
    max_samples = 500
    max_seq_len = max_samples * n_vars  # Must match training (500 * 10 = 5000)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path('evaluation_report')
    save_dir.mkdir(exist_ok=True)
    
    ckpt_path = Path('results_distributed/best_model.pt')
    if not ckpt_path.exists():
        ckpt_path = Path('results/best_model.pt')
        
    if not ckpt_path.exists():
        print(f"❌ Error: Could not find checkpoint at {ckpt_path}")
        return

    # Load Model
    print(f"Loading model from {ckpt_path}...")
    model = PoGPFN(
        n_vars=n_vars,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_claim_layers=n_claim_layers,
        max_seq_len=max_seq_len
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    results = []

    # 1. Standard Test Set
    standard_metrics = run_scenario(
        model, device, "Standard Test",
        {
            'n_tasks': 200,
            'n_vars': n_vars,
            'seed': 999
        },
        save_dir
    )
    results.append({'Scenario': 'Standard Test', **standard_metrics})

    # 2. OOD: Novel Mechanisms (Heavy Tailed & Heteroskedastic)
    ood_mech_metrics = run_scenario(
        model, device, "OOD Mechanisms",
        {
            'n_tasks': 100,
            'n_vars': n_vars,
            'mechanism_types': [MechanismType.HEAVY_TAILED, MechanismType.HETEROSKEDASTIC],
            'seed': 1000
        },
        save_dir
    )
    results.append({'Scenario': 'OOD Mechanisms', **ood_mech_metrics})

    # 3. OOD: High Density Graphs
    ood_density_metrics = run_scenario(
        model, device, "OOD High Density",
        {
            'n_tasks': 100,
            'n_vars': n_vars,
            'densities': [0.5, 0.8],
            'seed': 1001
        },
        save_dir
    )
    results.append({'Scenario': 'OOD Density', **ood_density_metrics})

    # 4. Realistic Mix
    realistic_metrics = run_scenario(
        model, device, "Realistic Mix",
        {
            'n_tasks': 100,
            'n_vars': n_vars,
            'mechanism_types': list(MechanismType),
            'graph_types': list(GraphType),
            'seed': 1002
        },
        save_dir
    )
    results.append({'Scenario': 'Realistic Mix', **realistic_metrics})

    # Create Summary Table
    df = pd.DataFrame(results)
    # Drop z_scores for the table
    summary_df = df.drop(columns=['z_scores'])
    
    print("\n" + "="*80)
    print("FINAL EVALUATION SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

    # Save to CSV and JSON
    summary_df.to_csv(save_dir / 'metrics_summary.csv', index=False)
    # For JSON, we might want to keep more info but handle numpy/tensor types
    with open(save_dir / 'metrics_summary.json', 'w') as f:
        json_results = []
        for r in results:
            clean_r = {k: (v.tolist() if isinstance(v, (np.ndarray, torch.Tensor)) else v) for k, v in r.items()}
            json_results.append(clean_r)
        json.dump(json_results, f, indent=4)

    print(f"\nFull report saved to: {save_dir}/")
    print("  - metrics_summary.csv")
    print("  - metrics_summary.json")
    print("  - Individual scenario subfolders with plots")


if __name__ == "__main__":
    main()
