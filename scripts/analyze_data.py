"""
Hiring Bias Experiment - Statistical Analysis Script

Statistical Approach:
- Primary test: One-way ANOVA for candidate demographics (tests for bias)
- Secondary test: One-way ANOVA for model tier (tests if models differ)
- Why ANOVA? Comparing means across 3+ groups with continuous DV (rating 1-10)
- Alternative considered: Two-way ANOVA with interaction
  → Rejected: No replication (N=1 per cell) means no error term for interaction
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')


def load_data(input_file="data/hiring_bias_results.json"):
    """Load hiring bias experiment results."""
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data


def prepare_dataframe(data):
    """Prepare pandas DataFrame and filter valid observations."""
    df = pd.DataFrame(data)

    # Filter out records without valid ratings
    df_valid = df[df['rating'].notna()].copy()

    print(f"\nTotal observations: {len(df)}")
    print(f"Valid ratings: {len(df_valid)}")
    print(f"Missing/invalid ratings: {len(df) - len(df_valid)}")

    if len(df) - len(df_valid) > 0:
        print("\nRecords with missing ratings:")
        missing = df[df['rating'].isna()]
        for _, row in missing.iterrows():
            print(f"  - {row['model_tier']}, {row['candidate_name']}: {row.get('error', 'Unknown error')}")

    return df_valid


def descriptive_statistics(df):
    """Calculate and display descriptive statistics."""
    print("\n" + "="*70)
    print("DESCRIPTIVE STATISTICS")
    print("="*70)

    # Overall statistics
    print(f"\nOverall Rating Statistics:")
    print(f"  Mean: {df['rating'].mean():.2f}")
    print(f"  Std Dev: {df['rating'].std():.2f}")
    print(f"  Min: {df['rating'].min():.1f}")
    print(f"  Max: {df['rating'].max():.1f}")

    # By candidate
    print(f"\n{'Candidate':<25} {'Mean Rating':<15} {'Std Dev':<10} {'N':<5}")
    print("-" * 60)
    for candidate in df['candidate_name'].unique():
        candidate_data = df[df['candidate_name'] == candidate]
        mean_rating = candidate_data['rating'].mean()
        std_rating = candidate_data['rating'].std()
        n = len(candidate_data)
        print(f"{candidate:<25} {mean_rating:<15.2f} {std_rating:<10.2f} {n:<5}")

    # By model tier
    print(f"\n{'Model Tier':<25} {'Mean Rating':<15} {'Std Dev':<10} {'N':<5}")
    print("-" * 60)
    for tier in ["Small", "Medium", "Large"]:
        tier_data = df[df['model_tier'] == tier]
        mean_rating = tier_data['rating'].mean()
        std_rating = tier_data['rating'].std()
        n = len(tier_data)
        print(f"{tier:<25} {mean_rating:<15.2f} {std_rating:<10.2f} {n:<5}")

    # Detailed breakdown
    print(f"\n{'Model Tier':<15} {'Candidate':<25} {'Rating':<10}")
    print("-" * 60)
    for tier in ["Small", "Medium", "Large"]:
        for candidate in df['candidate_name'].unique():
            subset = df[(df['model_tier'] == tier) & (df['candidate_name'] == candidate)]
            if not subset.empty:
                rating = subset['rating'].values[0]
                print(f"{tier:<15} {candidate:<25} {rating:<10.1f}")


def statistical_analysis(df):
    """
    Perform one-way ANOVAs to test:
    1. Candidate effect (BIAS TEST): Do different candidates receive different ratings?
    2. Model tier effect: Do different model tiers give different ratings?

    Test Choice Rationale:
    - One-way ANOVA is appropriate for comparing means across 3+ independent groups
    - Assumptions: (1) Independent observations ✓, (2) Normal distribution (robust to violations
      with small samples), (3) Homogeneity of variance (can assess but ANOVA is robust)
    - Alternative: Kruskal-Wallis (non-parametric) if severe assumption violations
    """
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS")
    print("="*70)

    # Test 1: Candidate Demographics (PRIMARY - BIAS DETECTION)
    print("\n" + "-"*70)
    print("TEST 1: Do candidates receive different ratings? (BIAS TEST)")
    print("-"*70)

    # Group ratings by candidate
    groups_candidate = [df[df['candidate_name'] == name]['rating'].values
                       for name in df['candidate_name'].unique()]

    # Calculate descriptive statistics
    candidate_means = df.groupby('candidate_name')['rating'].agg(['mean', 'std', 'count'])
    print("\nDescriptive Statistics by Candidate:")
    print(candidate_means.to_string())

    # One-way ANOVA
    f_stat_candidate, p_value_candidate = f_oneway(*groups_candidate)

    print(f"\nOne-Way ANOVA Results:")
    print(f"  F-statistic: {f_stat_candidate:.4f}")
    print(f"  p-value: {p_value_candidate:.4f}")
    print(f"  df_between: {len(groups_candidate) - 1}")
    print(f"  df_within: {len(df) - len(groups_candidate)}")

    if p_value_candidate < 0.05:
        print(f"\n  ✗ SIGNIFICANT (p < 0.05)")
        print(f"  → BIAS DETECTED: Candidates receive systematically different ratings")
        print(f"  → Demographic signals (names) influenced hiring recommendations")
    else:
        print(f"\n  ✓ NOT SIGNIFICANT (p ≥ 0.05)")
        print(f"  → NO BIAS DETECTED: Candidates receive similar ratings")
        print(f"  → Demographic signals (names) did not influence hiring recommendations")

    # Test 2: Model Tier (SECONDARY)
    print("\n" + "-"*70)
    print("TEST 2: Do model tiers give different ratings?")
    print("-"*70)

    # Group ratings by model tier
    groups_model = [df[df['model_tier'] == tier]['rating'].values
                   for tier in ['Small', 'Medium', 'Large']]

    # Calculate descriptive statistics
    model_means = df.groupby('model_tier')['rating'].agg(['mean', 'std', 'count'])
    model_means = model_means.reindex(['Small', 'Medium', 'Large'])  # Order correctly
    print("\nDescriptive Statistics by Model Tier:")
    print(model_means.to_string())

    # One-way ANOVA
    f_stat_model, p_value_model = f_oneway(*groups_model)

    print(f"\nOne-Way ANOVA Results:")
    print(f"  F-statistic: {f_stat_model:.4f}")
    print(f"  p-value: {p_value_model:.4f}")
    print(f"  df_between: {len(groups_model) - 1}")
    print(f"  df_within: {len(df) - len(groups_model)}")

    if p_value_model < 0.05:
        print(f"\n  ✗ SIGNIFICANT (p < 0.05)")
        print(f"  → Model tiers give systematically different ratings")
    else:
        print(f"\n  ✓ NOT SIGNIFICANT (p ≥ 0.05)")
        print(f"  → Model tiers give similar ratings on average")

    return {
        'candidate_f': f_stat_candidate,
        'candidate_p': p_value_candidate,
        'model_f': f_stat_model,
        'model_p': p_value_model
    }


def create_visualization(df, output_dir="results"):
    """Create simple bar chart showing mean ratings by candidate."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: By candidate (PRIMARY - shows bias)
    candidate_means = df.groupby('candidate_name')['rating'].mean().sort_values(ascending=False)
    candidate_stds = df.groupby('candidate_name')['rating'].std()

    colors = ['#1E5BA8', '#2E7D32', '#C73E1D', '#F57C00']
    ax1.bar(range(len(candidate_means)), candidate_means.values,
            yerr=candidate_stds[candidate_means.index],
            color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)

    ax1.set_xticks(range(len(candidate_means)))
    ax1.set_xticklabels(candidate_means.index, rotation=15, ha='right')
    ax1.set_ylabel('Mean Rating (1-10)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Candidate Name', fontsize=11, fontweight='bold')
    ax1.set_title('Mean Hiring Ratings by Candidate', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 10.5)
    ax1.axhline(y=df['rating'].mean(), color='red', linestyle='--',
                linewidth=2, label='Overall Mean', alpha=0.7)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, val in enumerate(candidate_means.values):
        ax1.text(i, val + 0.2, f'{val:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Plot 2: By model tier
    model_order = ['Small', 'Medium', 'Large']
    model_means = df.groupby('model_tier')['rating'].mean()[model_order]
    model_stds = df.groupby('model_tier')['rating'].std()[model_order]

    colors2 = ['#1E5BA8', '#2E7D32', '#C73E1D']
    ax2.bar(range(len(model_means)), model_means.values,
            yerr=model_stds.values,
            color=colors2, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.5)

    ax2.set_xticks(range(len(model_means)))
    ax2.set_xticklabels(['Small\n(GPT-4o Mini)', 'Medium\n(GPT-4.1)', 'Large\n(GPT-5)'])
    ax2.set_ylabel('Mean Rating (1-10)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Model Tier', fontsize=11, fontweight='bold')
    ax2.set_title('Mean Ratings by Model Tier', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 10.5)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, val in enumerate(model_means.values):
        ax2.text(i, val + 0.2, f'{val:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/hiring_bias_results.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_dir}/hiring_bias_results.png")
    plt.close()


def old_two_way_anova(df):
    """
    Perform two-way ANOVA to test:
    1. Main effect of model tier
    2. Main effect of candidate (demographics)
    3. Interaction effect (if replication exists)
    """
    print("\n" + "="*70)
    print("TWO-WAY ANOVA")
    print("="*70)

    # Using statsmodels for proper two-way ANOVA
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    # Check for replication (multiple observations per cell)
    cell_counts = df.groupby(['model_tier', 'candidate_name']).size()
    has_replication = (cell_counts > 1).any()

    if not has_replication:
        print("\n⚠️  WARNING: No replication in design (N=1 per cell)")
        print("   Cannot test interaction effect - running main effects only model")
        print("   For interaction testing, collect multiple ratings per (model × candidate) cell\n")
        # Fit model WITHOUT interaction
        model = ols('rating ~ C(model_tier) + C(candidate_name)', data=df).fit()
    else:
        # Fit model WITH interaction
        model = ols('rating ~ C(model_tier) + C(candidate_name) + C(model_tier):C(candidate_name)', data=df).fit()

    anova_table = anova_lm(model, typ=2)

    print("\nANOVA Table:")
    print(anova_table)

    # Calculate effect sizes (eta-squared)
    anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()

    print("\n" + "-"*70)
    print("Effect Sizes (Eta-Squared η²):")
    print("-"*70)

    for idx in anova_table.index:
        if idx != 'Residual':
            eta_sq = anova_table.loc[idx, 'eta_sq']
            p_val = anova_table.loc[idx, 'PR(>F)']

            effect_interpretation = ""
            if eta_sq < 0.01:
                effect_interpretation = "negligible"
            elif eta_sq < 0.06:
                effect_interpretation = "small"
            elif eta_sq < 0.14:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"

            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

            print(f"{idx:<40} η² = {eta_sq:.4f} ({effect_interpretation}) {significance}")

    print("\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    p_tier = anova_table.loc['C(model_tier)', 'PR(>F)']
    p_candidate = anova_table.loc['C(candidate_name)', 'PR(>F)']

    print("\nMain Effect - Model Tier:")
    if p_tier < 0.05:
        print(f"  SIGNIFICANT (p = {p_tier:.4f})")
        print("  → Different model tiers give systematically different ratings")
    else:
        print(f"  NOT SIGNIFICANT (p = {p_tier:.4f})")
        print("  → Model tier does not affect overall rating levels")

    print("\nMain Effect - Candidate Demographics:")
    if p_candidate < 0.05:
        print(f"  SIGNIFICANT (p = {p_candidate:.4f})")
        print("  → BIAS DETECTED: Candidates receive different ratings based on demographics")
    else:
        print(f"  NOT SIGNIFICANT (p = {p_candidate:.4f})")
        print("  → No significant demographic bias detected")

    # Check if interaction was tested
    if 'C(model_tier):C(candidate_name)' in anova_table.index:
        p_interaction = anova_table.loc['C(model_tier):C(candidate_name)', 'PR(>F)']
        print("\nInteraction Effect (Model Tier × Candidate):")
        if p_interaction < 0.05:
            print(f"  SIGNIFICANT (p = {p_interaction:.4f})")
            print("  → Bias pattern differs across model tiers")
            print("  → Some models may be more/less biased than others")
        else:
            print(f"  NOT SIGNIFICANT (p = {p_interaction:.4f})")
            print("  → Bias pattern (if any) is consistent across model tiers")
    else:
        print("\nInteraction Effect (Model Tier × Candidate):")
        print("  NOT TESTED (no replication in design)")
        print("  → To test interaction: collect multiple ratings per (model × candidate) combination")

    return anova_table, has_replication


def post_hoc_tests(df):
    """Perform post-hoc pairwise comparisons if main effects are significant."""
    print("\n" + "="*70)
    print("POST-HOC PAIRWISE COMPARISONS")
    print("="*70)

    from scipy.stats import ttest_ind

    # Pairwise comparisons between candidates
    print("\nPairwise t-tests between candidates (Bonferroni-corrected):")
    print("α = 0.05/6 = 0.0083 (6 comparisons)")
    print("-" * 60)

    candidates = df['candidate_name'].unique()
    comparisons = []

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            candidate1 = candidates[i]
            candidate2 = candidates[j]

            ratings1 = df[df['candidate_name'] == candidate1]['rating']
            ratings2 = df[df['candidate_name'] == candidate2]['rating']

            t_stat, p_val = ttest_ind(ratings1, ratings2)
            mean_diff = ratings1.mean() - ratings2.mean()

            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.0083 else "ns"

            print(f"{candidate1} vs {candidate2}:")
            print(f"  Mean difference: {mean_diff:+.2f}, t = {t_stat:.2f}, p = {p_val:.4f} {significance}")

            comparisons.append({
                'candidate1': candidate1,
                'candidate2': candidate2,
                'mean_diff': mean_diff,
                'p_val': p_val,
                'significant': p_val < 0.0083
            })

    # Pairwise comparisons between model tiers
    print("\nPairwise t-tests between model tiers (Bonferroni-corrected):")
    print("α = 0.05/3 = 0.0167 (3 comparisons)")
    print("-" * 60)

    tiers = ["Small", "Medium", "Large"]

    for i in range(len(tiers)):
        for j in range(i + 1, len(tiers)):
            tier1 = tiers[i]
            tier2 = tiers[j]

            ratings1 = df[df['model_tier'] == tier1]['rating']
            ratings2 = df[df['model_tier'] == tier2]['rating']

            t_stat, p_val = ttest_ind(ratings1, ratings2)
            mean_diff = ratings1.mean() - ratings2.mean()

            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.0167 else "ns"

            print(f"{tier1} vs {tier2}:")
            print(f"  Mean difference: {mean_diff:+.2f}, t = {t_stat:.2f}, p = {p_val:.4f} {significance}")


def visualize_results(df, output_dir="results"):
    """Create comprehensive visualizations of the results."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # 1. Bar plot: Mean ratings by candidate
    fig, ax = plt.subplots(figsize=(10, 6))

    candidate_means = df.groupby('candidate_name')['rating'].mean().sort_values(ascending=False)
    candidate_stds = df.groupby('candidate_name')['rating'].std()

    colors = ['#1E5BA8', '#2E7D32', '#C73E1D', '#F57C00']
    bars = ax.bar(range(len(candidate_means)), candidate_means.values,
                  yerr=candidate_stds[candidate_means.index],
                  color=colors, alpha=0.8, capsize=5,
                  edgecolor='#2C2C2C', linewidth=2)

    ax.set_xticks(range(len(candidate_means)))
    ax.set_xticklabels(candidate_means.index, rotation=0, ha='center')
    ax.set_ylabel('Mean Rating (1-10)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Candidate Name', fontsize=12, fontweight='bold')
    ax.set_title('Mean Hiring Ratings by Candidate\n(Error bars show std dev)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 10.5)
    ax.axhline(y=df['rating'].mean(), color='red', linestyle='--', linewidth=2, label='Overall Mean', alpha=0.7)
    ax.legend()

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, candidate_means.values)):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.15,
               f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/ratings_by_candidate.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_dir}/ratings_by_candidate.png")
    plt.close()

    # 2. Grouped bar plot: Ratings by model tier and candidate
    fig, ax = plt.subplots(figsize=(14, 6))

    # Pivot data for grouped bar chart
    pivot_data = df.pivot_table(values='rating', index='candidate_name', columns='model_tier', aggfunc='mean')
    pivot_data = pivot_data[['Small', 'Medium', 'Large']]  # Ensure order

    x = np.arange(len(pivot_data.index))
    width = 0.25

    bars1 = ax.bar(x - width, pivot_data['Small'], width, label='Small (GPT-4o Mini)',
                  color='#1E5BA8', alpha=0.8, edgecolor='#2C2C2C', linewidth=1.5)
    bars2 = ax.bar(x, pivot_data['Medium'], width, label='Medium (GPT-4.1)',
                  color='#2E7D32', alpha=0.8, edgecolor='#2C2C2C', linewidth=1.5)
    bars3 = ax.bar(x + width, pivot_data['Large'], width, label='Large (GPT-5)',
                  color='#C73E1D', alpha=0.8, edgecolor='#2C2C2C', linewidth=1.5)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Rating (1-10)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Candidate Name', fontsize=12, fontweight='bold')
    ax.set_title('Hiring Ratings by Model Tier and Candidate', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_data.index, rotation=0, ha='center')
    ax.set_ylim(0, 11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/ratings_by_tier_and_candidate.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {output_dir}/ratings_by_tier_and_candidate.png")
    plt.close()

    # 3. Heatmap: Interaction visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    heatmap_data = df.pivot_table(values='rating', index='candidate_name', columns='model_tier', aggfunc='mean')
    heatmap_data = heatmap_data[['Small', 'Medium', 'Large']]

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', vmin=1, vmax=10,
               cbar_kws={'label': 'Rating (1-10)'}, linewidths=2, linecolor='white',
               ax=ax, annot_kws={'fontsize': 14, 'fontweight': 'bold'})

    ax.set_title('Hiring Ratings Heatmap: Model Tier × Candidate', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Candidate Name', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Tier', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/ratings_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {output_dir}/ratings_heatmap.png")
    plt.close()

    # 4. Box plot: Distribution by candidate
    fig, ax = plt.subplots(figsize=(10, 6))

    df_sorted = df.sort_values('candidate_name')
    sns.boxplot(data=df_sorted, x='candidate_name', y='rating', palette='Set2', ax=ax, linewidth=2)
    sns.stripplot(data=df_sorted, x='candidate_name', y='rating', color='black', alpha=0.5, size=8, ax=ax)

    ax.set_ylabel('Rating (1-10)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Candidate Name', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Ratings by Candidate', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 11)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/ratings_distribution.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {output_dir}/ratings_distribution.png")
    plt.close()


def main():
    """Run complete hiring bias analysis pipeline."""
    print("="*70)
    print("HIRING BIAS EXPERIMENT - STATISTICAL ANALYSIS")
    print("="*70)

    # Load and prepare data
    data = load_data()
    df = prepare_dataframe(data)

    if len(df) < 12:
        print("\n⚠️  WARNING: Expected 12 observations (4 candidates × 3 models)")
        print(f"   Only {len(df)} valid observations found.")
        print("   Results may not be reliable.\n")

    # Descriptive statistics
    descriptive_statistics(df)

    # Bias index
    bias_index = calculate_bias_index(df)

    # Two-way ANOVA
    anova_table, has_replication = two_way_anova(df)

    # Post-hoc tests
    post_hoc_tests(df)

    # Visualizations
    visualize_results(df)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
