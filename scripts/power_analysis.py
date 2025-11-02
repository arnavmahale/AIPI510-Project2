"""
Power Analysis for Authority Resistance Experiment

Determines minimum sample size needed to detect effect with 80% power at α=0.05
"""

## AI Tool Attribution: Built with assistance from Claude Code CLI (https://claude.ai/claude-code)
## Developed a priori power analysis approach. Worked through identifying the right
## statistical test for the primary hypothesis and calculating appropriate sample sizes.

import numpy as np
from scipy import stats
from statsmodels.stats.power import GofChisquarePower

print("="*70)
print("POWER ANALYSIS: Authority Resistance Experiment")
print("="*70)

# Study parameters
alpha = 0.05
power = 0.80
n_models = 3
n_questions = 3

print(f"\nStudy Design:")
print(f"  - Models tested: {n_models} (GPT-3.5, GPT-4, GPT-5)")
print(f"  - Questions per model: {n_questions}")
print(f"  - Significance level (α): {alpha}")
print(f"  - Desired power (1-β): {power}")

# PRIMARY HYPOTHESIS: Proportion of models that become wrong
print("\n" + "="*70)
print("PRIMARY TEST: Binomial/Chi-square (Became Wrong > 0)")
print("="*70)
print("H₀: Models don't change correct answers (p = 0)")
print("H₁: Models change correct answers when challenged (p > 0)")

# Expected effect size
# Conservative estimate: detect if 25% of initially correct answers become wrong
# This is a medium-large effect in behavioral science
expected_proportion = 0.25

print(f"\nAssumptions:")
print(f"  - Expected proportion becoming wrong: {expected_proportion*100:.0f}%")
print(f"  - Baseline (null): 0% (no susceptibility)")

# Cohen's h for proportion difference
p1 = 0.0  # null hypothesis
p2 = expected_proportion  # alternative hypothesis
h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

print(f"  - Effect size (Cohen's h): {h:.3f}")
print(f"    ({['small', 'medium', 'large'][(h > 0.2) + (h > 0.5)]} effect)")

# Calculate required sample size
power_analysis = GofChisquarePower()
n_required = power_analysis.solve_power(
    effect_size=h,
    alpha=alpha,
    power=power,
    n_bins=2  # became_wrong: yes/no
)

print(f"\nSample Size Calculation:")
print(f"  - Minimum total observations: {np.ceil(n_required):.0f}")
print(f"  - Per model ({n_models} models): {np.ceil(n_required/n_models):.0f}")
print(f"  - Per cell ({n_models} models × {n_questions} questions): {np.ceil(n_required/(n_models*n_questions)):.0f} trials")

# Calculate actual study design
trials_per_cell = int(np.ceil(n_required / (n_models * n_questions)))
total_observations = n_models * n_questions * trials_per_cell

print(f"\n" + "="*70)
print("RECOMMENDED DESIGN")
print("="*70)
print(f"  - {n_models} models × {n_questions} questions × {trials_per_cell} trials = {total_observations} total observations")
print(f"  - This provides {power*100:.0f}% power to detect {expected_proportion*100:.0f}% susceptibility rate")
print(f"  - Estimated API calls: {total_observations * 2} (2 per trial)")

# Verification: What power do we actually have?
actual_h = h  # same effect size
actual_power = power_analysis.solve_power(
    effect_size=actual_h,
    alpha=alpha,
    power=None,
    n_bins=2,
    nobs=total_observations
)

print(f"\nVerification:")
print(f"  - Actual statistical power: {actual_power:.3f} ({actual_power*100:.1f}%)")
print(f"  - {'✓' if actual_power >= 0.80 else '✗'} Meets {power*100:.0f}% power requirement")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print(f"Recommended: {trials_per_cell} trials per (model × question) combination")
print(f"Total: {total_observations} observations provides adequate power to detect")
print(f"meaningful susceptibility to false authority claims.")
