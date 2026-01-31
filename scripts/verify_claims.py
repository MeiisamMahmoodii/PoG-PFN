"""
Verification Script: Claim Distribution Analysis

Verifies that the model receives both true and false claims during training.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pog_pfn.data.dataset import PoGPFNDataset
import numpy as np

print("="*70)
print("CLAIM DIVERSITY ANALYSIS")
print("="*70)

# Create dataset
print("\n1. Creating dataset with default claim ratios...")
dataset = PoGPFNDataset(
    n_tasks=100,
    n_vars=10,
    n_samples_per_task=500,
    n_claims_per_task=10,
    seed=42
)

# Analyze claims
total_claims = 0
true_claims = 0
false_claims = 0
confidence_scores = []

print("2. Analyzing 100 tasks...\n")

for i in range(len(dataset)):
    sample = dataset[i]
    claims = sample['claims']
    
    for claim in claims:
        total_claims += 1
        if claim.is_true:
            true_claims += 1
        else:
            false_claims += 1
        confidence_scores.append(claim.confidence)

true_ratio = true_claims / total_claims
false_ratio = false_claims / total_claims

print("RESULTS:")
print("-"*70)
print(f"Total claims: {total_claims}")
print(f"True claims: {true_claims} ({true_ratio*100:.1f}%)")
print(f"False claims: {false_claims} ({false_ratio*100:.1f}%)")
print(f"\nAverage confidence: {np.mean(confidence_scores):.3f}")
print(f"Confidence range: [{np.min(confidence_scores):.3f}, {np.max(confidence_scores):.3f}]")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if false_ratio > 0.2:
    print("✅ YES - Model receives FALSE claims during training!")
    print(f"   Approximately {false_ratio*100:.0f}% of claims are false")
    print("\nThis is GOOD because:")
    print("  - Model learns to distinguish reliable from unreliable claims")
    print("  - Develops robustness to misinformation")
    print("  - Learns implicit claim validation")
else:
    print("⚠️  WARNING: Very few false claims in dataset")

print("="*70)
