"""
Statistical analysis script.
Reads redacted_results.json, performs chi-square test,
calculates effect sizes, and generates visualizations.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact

def load_data(input_file="redacted_results.json"):
    """Load cleaned experiment results."""
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def create_contingency_table(data):
    """Create contingency table of model tier vs. refusal."""
    df = pd.DataFrame(data)
    
    # Remove errors (is_refusal = None)
    df = df[df['is_refusal'].notna()]
    
    # Create contingency table
    contingency = pd.crosstab(df['model_tier'], df['is_refusal'])
    
    return contingency, df

def chi_square_test(contingency):
    """Perform chi-square test of independence."""
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    # Calculate Cramér's V (effect size)
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    
    return chi2, p_value, dof, cramers_v, expected

def post_hoc_tests(data):
    """Perform pairwise Fisher's exact tests with Bonferroni correction."""
    df = pd.DataFrame(data)
    df = df[df['is_refusal'].notna()]
    
    tiers = ["Small", "Medium", "Large"]
    pairs = [("Small", "Medium"), ("Small", "Large"), ("Medium", "Large")]
    
    print("\n=== Post-Hoc Pairwise Comparisons (Fisher's Exact Test) ===")
    print("Bonferroni-corrected α = 0.05/3 = 0.017")
    
    for tier1, tier2 in pairs:
        data1 = df[df['model_tier'] == tier1]
        data2 = df[df['model_tier'] == tier2]
        
        refusals1 = (data1['is_refusal'] == True).sum()
        compliances1 = (data1['is_refusal'] == False).sum()
        refusals2 = (data2['is_refusal'] == True).sum()
        compliances2 = (data2['is_refusal'] == False).sum()
        
        # Create 2x2 contingency table
        table_2x2 = [[refusals1, compliances1], [refusals2, compliances2]]
        
        # Fisher's exact test
        odds_ratio, p_value = fisher_exact(table_2x2)
        
        significance = "✓ Significant" if p_value < 0.017 else "✗ Not significant"
        print(f"{tier1} vs {tier2}: p = {p_value:.4f} {significance}")

def calculate_refusal_rates(data):
    """Calculate refusal rates by model tier."""
    df = pd.DataFrame(data)
    df = df[df['is_refusal'].notna()]
    
    print("\n=== Refusal Rates by Model Tier ===")
    for tier in ["Small", "Medium", "Large"]:
        tier_data = df[df['model_tier'] == tier]
        refusals = (tier_data['is_refusal'] == True).sum()
        total = len(tier_data)
        rate = refusals / total * 100
        print(f"{tier}: {refusals}/{total} ({rate:.0f}%)")

def generate_visualizations(contingency, output_dir="results"):
    """Generate bar chart of refusal rates."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate refusal rates
    refusal_rates = contingency[True] / contingency.sum(axis=1) * 100
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    tiers = refusal_rates.index
    rates = refusal_rates.values
    
    colors = ['#1E5BA8', '#C73E1D', '#1E5BA8']  # Blue, Red, Blue
    bars = ax.bar(tiers, rates, color=colors, alpha=0.8, edgecolor='#2C2C2C', linewidth=2)
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.0f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Refusal Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Tier', fontsize=12, fontweight='bold')
    ax.set_title('Refusal Rates by Model Tier', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/refusal_rates_by_tier.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_dir}/refusal_rates_by_tier.png")
    plt.close()

def main():
    """Run complete analysis pipeline."""
    print("=== Adversarial Robustness Analysis ===\n")
    
    # Load data
    data = load_data()
    
    # Create contingency table
    contingency, df = create_contingency_table(data)
    print(f"\nTotal valid observations: {len(df)}")
    print(f"\nContingency Table:")
    print(contingency)
    
    # Calculate refusal rates
    calculate_refusal_rates(data)
    
    # Chi-square test
    chi2, p_value, dof, cramers_v, expected = chi_square_test(contingency)
    
    print(f"\n=== Chi-Square Test Results ===")
    print(f"χ²({dof}) = {chi2:.2f}")
    print(f"p-value = {p_value:.6f}")
    print(f"Cramér's V = {cramers_v:.2f} (effect size)")
    
    if p_value < 0.05:
        print(f"Result: SIGNIFICANT (p < 0.05) - Reject H₀")
    else:
        print(f"Result: NOT SIGNIFICANT (p ≥ 0.05) - Fail to reject H₀")
    
    # Post-hoc tests
    post_hoc_tests(data)
    
    # Generate visualizations
    generate_visualizations(contingency)
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
