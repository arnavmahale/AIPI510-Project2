"""
Hiring Bias Experiment - Statistical Analysis Script
Analyzes hiring ratings for bias across demographic groups.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_data(filepath="data/ratings.csv"):
    """Load and prepare data for analysis."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} ratings")
    print(f"Models: {df['model'].unique()}")
    print(f"Candidates: {df['candidate_name'].unique()}\n")
    return df


def descriptive_statistics(df):
    """Calculate and display descriptive statistics."""
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    summary = df.groupby(['demographic_signal', 'model'])['rating'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).round(2)

    print(summary)
    print()

    # Overall by demographic
    overall = df.groupby('demographic_signal')['rating'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count')
    ]).round(2)

    print("\nOverall by Demographic:")
    print(overall)
    print()

    return summary, overall


def check_assumptions(df):
    """Check statistical assumptions for ANOVA."""
    print("=" * 60)
    print("ASSUMPTION CHECKS")
    print("=" * 60)

    # Normality test (Shapiro-Wilk) for each group
    print("\nNormality Test (Shapiro-Wilk):")
    for demographic in df['demographic_signal'].unique():
        group_data = df[df['demographic_signal'] == demographic]['rating']
        stat, p_value = stats.shapiro(group_data)
        print(f"  {demographic:30s} p={p_value:.4f} {'(Normal)' if p_value > 0.05 else '(Not Normal)'}")

    # Levene's test for equal variances
    groups = [df[df['demographic_signal'] == d]['rating'].values
              for d in df['demographic_signal'].unique()]
    stat, p_value = stats.levene(*groups)
    print(f"\nEqual Variance Test (Levene): p={p_value:.4f} {'(Equal)' if p_value > 0.05 else '(Unequal)'}")
    print()

    return p_value > 0.05


def perform_anova(df):
    """Perform one-way ANOVA."""
    print("=" * 60)
    print("ONE-WAY ANOVA")
    print("=" * 60)

    # Prepare groups
    groups = [df[df['demographic_signal'] == d]['rating'].values
              for d in df['demographic_signal'].unique()]

    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Result: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at α=0.05")

    # Effect size (eta-squared)
    grand_mean = df['rating'].mean()
    ss_between = sum(len(group) * (group.mean() - grand_mean)**2 for group in groups)
    ss_total = sum((df['rating'] - grand_mean)**2)
    eta_squared = ss_between / ss_total

    print(f"Effect size (η²): {eta_squared:.4f}")
    print()

    return p_value < 0.05, p_value


def perform_posthoc(df):
    """Perform Tukey HSD post-hoc test."""
    print("=" * 60)
    print("POST-HOC TEST (Tukey HSD)")
    print("=" * 60)

    from scipy.stats import tukey_hsd

    # Prepare groups
    demographics = df['demographic_signal'].unique()
    groups = [df[df['demographic_signal'] == d]['rating'].values for d in demographics]

    # Perform Tukey HSD
    result = tukey_hsd(*groups)

    print("\nPairwise comparisons:")
    for i, demo1 in enumerate(demographics):
        for j, demo2 in enumerate(demographics):
            if i < j:
                mean_diff = groups[i].mean() - groups[j].mean()
                ci_low = result.confidence_interval().low[i, j]
                ci_high = result.confidence_interval().high[i, j]
                p_val = result.pvalue[i, j]

                print(f"\n{demo1} vs {demo2}:")
                print(f"  Mean difference: {mean_diff:.3f}")
                print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
                print(f"  p-value: {p_val:.4f} {'*' if p_val < 0.05 else ''}")
    print()


def create_visualizations(df):
    """Create plots for the analysis."""
    print("=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Box plot by demographic
    sns.boxplot(data=df, x='demographic_signal', y='rating', ax=axes[0, 0])
    axes[0, 0].set_title('Hiring Ratings by Demographic Group', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Demographic Signal')
    axes[0, 0].set_ylabel('Rating (1-10)')
    axes[0, 0].tick_params(axis='x', rotation=15)

    # Violin plot by demographic
    sns.violinplot(data=df, x='demographic_signal', y='rating', ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Ratings by Demographic', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Demographic Signal')
    axes[0, 1].set_ylabel('Rating (1-10)')
    axes[0, 1].tick_params(axis='x', rotation=15)

    # Bar plot with error bars
    means = df.groupby('demographic_signal')['rating'].mean()
    stds = df.groupby('demographic_signal')['rating'].std()
    axes[1, 0].bar(range(len(means)), means, yerr=stds, capsize=5, alpha=0.7)
    axes[1, 0].set_xticks(range(len(means)))
    axes[1, 0].set_xticklabels(means.index, rotation=15)
    axes[1, 0].set_title('Mean Ratings with Standard Deviation', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Demographic Signal')
    axes[1, 0].set_ylabel('Mean Rating (1-10)')
    axes[1, 0].set_ylim(0, 10)

    # Ratings by model and demographic
    model_demo = df.groupby(['model', 'demographic_signal'])['rating'].mean().unstack()
    model_demo.plot(kind='bar', ax=axes[1, 1], width=0.8)
    axes[1, 1].set_title('Mean Ratings by Model and Demographic', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('Mean Rating (1-10)')
    axes[1, 1].legend(title='Demographic', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig('results/analysis_plots.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to results/analysis_plots.png\n")


def main():
    """Run the complete statistical analysis."""
    import os
    os.makedirs("results", exist_ok=True)

    # Load data
    df = load_data()

    # Descriptive statistics
    descriptive_statistics(df)

    # Check assumptions
    equal_variance = check_assumptions(df)

    # Perform ANOVA
    is_significant, p_value = perform_anova(df)

    # Post-hoc if significant
    if is_significant:
        perform_posthoc(df)

    # Create visualizations
    create_visualizations(df)

    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to results/")


if __name__ == "__main__":
    main()
