"""
Statistical Analysis for Authority Resistance Experiment
Performs hypothesis tests on collected data
"""

## AI Tool Attribution: Built with assistance from Claude Code CLI (https://claude.ai/claude-code)
## Iteratively refined the statistical approach across multiple sessions. Started with
## comprehensive analysis suite, then simplified to focus on the core hypotheses.
## Helped identify and remove redundancies in the codebase.

import json
import numpy as np
from scipy import stats
import pandas as pd
import glob

# Auto-load most recent data file
data_files = glob.glob('data/authority_resistance_*.json')
if not data_files:
    print("Error: No data files found in data/")
    exit(1)

filename = max(data_files, key=lambda x: x)
print(f"Loading: {filename}\n")

with open(filename, 'r') as f:
    results = json.load(f)

df = pd.DataFrame(results)

print("="*70)
print("STATISTICAL ANALYSIS: Authority Resistance Experiment")
print("="*70)
print(f"\nDataset: {len(results)} observations")
print(f"Design: {df['model'].nunique()} models × {df['problem'].nunique()} questions × {df['trial'].max()} trials\n")

# PRIMARY HYPOTHESIS: Models change correct answers when challenged
print("="*70)
print("PRIMARY HYPOTHESIS: Binomial Test (Overall Susceptibility)")
print("="*70)
print("H₀: Models don't change correct answers (p = 0)")
print("H₁: Models change correct answers when challenged (p > 0)\n")

became_wrong = df['became_wrong'].sum()
total_initially_correct = df['initially_correct'].sum()

print(f"Observed: {became_wrong}/{total_initially_correct} became wrong ({became_wrong/total_initially_correct*100:.1f}%)")

# Binomial test against p=0 (no susceptibility)
binom_result = stats.binomtest(became_wrong, total_initially_correct, p=0.01, alternative='greater')
print(f"Binomial test: p = {binom_result.pvalue:.6f}")

if binom_result.pvalue < 0.05:
    print(f"Result: ✓ REJECT H₀ (p < 0.05)")
else:
    print(f"Result: ✗ FAIL TO REJECT H₀")

# Effect size and confidence interval (two-sided for reporting)
effect_size = became_wrong / total_initially_correct
# Calculate two-sided CI for the proportion
from statsmodels.stats.proportion import proportion_confint
ci_lower, ci_upper = proportion_confint(became_wrong, total_initially_correct, method='wilson')
print(f"\nEffect size: {effect_size:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")

# POST-HOC ANALYSIS: Models differ in susceptibility
print("\n" + "="*70)
print("POST-HOC ANALYSIS: Chi-Square Test (Model Differences)")
print("="*70)
print("Testing whether susceptibility varies across model types\n")

contingency = pd.crosstab(df['model'], df['became_wrong'])
print("Contingency Table:")
print(contingency)
print()

# Display percentages
print("Susceptibility Rates by Model:")
for model in df['model'].unique():
    model_data = df[df['model'] == model]
    rate = model_data['became_wrong'].sum() / len(model_data) * 100
    print(f"  {model}: {rate:.1f}%")

chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
print(f"\nχ²({dof}) = {chi2:.3f}, p = {p_chi:.6f}")

if p_chi < 0.05:
    print(f"Result: ✓ SIGNIFICANT (p < 0.05)")
else:
    print(f"Result: ✗ NOT SIGNIFICANT")

# Cramér's V effect size
n = contingency.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
effect_label = 'small' if cramers_v < 0.1 else 'medium' if cramers_v < 0.3 else 'large'
print(f"\nCramér's V: {cramers_v:.3f} ({effect_label} effect)")

# Summary
print("\n" + "="*70)
print("SUMMARY OF FINDINGS")
print("="*70)

print(f"\n1. PRIMARY HYPOTHESIS TEST:")
print(f"   - {became_wrong}/{total_initially_correct} initially correct answers became wrong ({effect_size*100:.1f}%)")
print(f"   - Binomial test: p = {binom_result.pvalue:.6f}")
print(f"   - 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"   - {'✓ REJECT H₀ - Significant susceptibility detected' if binom_result.pvalue < 0.05 else '✗ FAIL TO REJECT H₀'}")

print(f"\n2. POST-HOC ANALYSIS (Model Differences):")
print(f"   - Chi-square test: χ²({dof}) = {chi2:.3f}, p = {p_chi:.6f}")
print(f"   - Effect size: Cramér's V = {cramers_v:.3f} ({effect_label})")
print(f"   - {'✓ SIGNIFICANT - Models differ in susceptibility' if p_chi < 0.05 else '✗ NOT SIGNIFICANT'}")

print(f"\n3. INTERPRETATION:")
if binom_result.pvalue < 0.05:
    if p_chi < 0.05:
        print(f"   Significant susceptibility detected ({effect_size*100:.1f}%); varies by model capability.")
    else:
        print(f"   Significant susceptibility detected ({effect_size*100:.1f}%); similar across models.")
else:
    print(f"   No significant evidence of authority susceptibility.")
