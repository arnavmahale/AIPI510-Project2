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

# HYPOTHESIS 1: Models change correct answers when challenged (PRIMARY)
print("="*70)
print("HYPOTHESIS 1: Binomial Test (Became Wrong)")
print("="*70)
print("H₀: Models don't change correct answers (p = 0)")
print("H₁: Models change correct answers when challenged (p > 0)\n")

became_wrong = df['became_wrong'].sum()
total_initially_correct = df['initially_correct'].sum()

print(f"Observed: {became_wrong}/{total_initially_correct} became wrong ({became_wrong/total_initially_correct*100:.1f}%)")

# Binomial test against p=0 (no susceptibility)
binom_result = stats.binomtest(became_wrong, total_initially_correct, p=0.01, alternative='greater')
print(f"Binomial test: p = {binom_result.pvalue:.4f}")

if binom_result.pvalue < 0.05:
    print(f"Result: ✓ REJECT H₀ (significant at α=0.05)")
    print(f"Conclusion: Models ARE susceptible to false authority claims")
else:
    print(f"Result: ✗ FAIL TO REJECT H₀")

# Effect size
effect_size = became_wrong / total_initially_correct
ci_lower, ci_upper = binom_result.proportion_ci(confidence_level=0.95)
print(f"\nEffect size: {effect_size:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")

# HYPOTHESIS 2: Models differ in susceptibility
print("\n" + "="*70)
print("HYPOTHESIS 2: Chi-Square Test (Model × Became Wrong)")
print("="*70)
print("H₀: Susceptibility rate is independent of model")
print("H₁: Models differ in susceptibility rate\n")

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
print(f"\nχ²({dof}) = {chi2:.3f}, p = {p_chi:.4f}")

if p_chi < 0.05:
    print(f"Result: ✓ REJECT H₀ (significant at α=0.05)")
    print(f"Conclusion: Susceptibility DIFFERS across models")
else:
    print(f"Result: ✗ FAIL TO REJECT H₀")
    print(f"Conclusion: No significant difference between models")

# Cramér's V effect size
n = contingency.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
effect_label = 'small' if cramers_v < 0.1 else 'medium' if cramers_v < 0.3 else 'large'
print(f"\nCramér's V: {cramers_v:.3f} ({effect_label} effect)")

# Summary
print("\n" + "="*70)
print("SUMMARY OF FINDINGS")
print("="*70)

print(f"\n1. PRIMARY FINDING:")
print(f"   - {became_wrong}/{total_initially_correct} initially correct answers became wrong ({effect_size*100:.1f}%)")
print(f"   - Binomial test: p = {binom_result.pvalue:.4f}")
print(f"   - {'✓ SIGNIFICANT' if binom_result.pvalue < 0.05 else '✗ NOT SIGNIFICANT'}")

print(f"\n2. MODEL DIFFERENCES:")
print(f"   - Chi-square test: χ²({dof}) = {chi2:.3f}, p = {p_chi:.4f}")
print(f"   - Effect size: Cramér's V = {cramers_v:.3f}")
print(f"   - {'✓ SIGNIFICANT' if p_chi < 0.05 else '✗ NOT SIGNIFICANT'}")

print(f"\n3. INTERPRETATION:")
if binom_result.pvalue < 0.05:
    print(f"   AI models demonstrate significant susceptibility to false authority")
    print(f"   claims, changing {effect_size*100:.1f}% of correct answers after challenge.")
else:
    print(f"   No significant evidence of authority susceptibility.")

if p_chi < 0.05:
    print(f"   Model size/capability affects resistance to false authority.")
else:
    print(f"   All models show similar susceptibility patterns.")

print("\n✓ Analysis complete. Run visualizations.py for charts.")
