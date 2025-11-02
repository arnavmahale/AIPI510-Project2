"""
Visualizations for Authority Resistance Experiment
"""

## AI Tool Attribution: Built with assistance from Claude Code CLI (https://claude.ai/claude-code)
## Collaborated on determining which visualizations would best communicate the findings.
## Focused on creating one impactful chart that clearly shows the before/after effect
## rather than multiple complex visualizations.

import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Auto-load most recent data file
files = glob.glob('data/authority_resistance_factual_*.json')
if not files:
    print("Error: No data files found in data/")
    exit(1)

filename = max(files)
print(f"Loading: {filename}\n")

with open(filename, 'r') as f:
    results = json.load(f)

df = pd.DataFrame(results)
sns.set_palette("husl")

# Main visualization: Before/After accuracy by model
fig, ax = plt.subplots(figsize=(10, 6))

models = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-5-mini"]
labels = ['GPT-3.5\nTurbo', 'GPT-4\nTurbo', 'GPT-5\nMini']

x = np.arange(len(models))
width = 0.35

# Calculate percentages
initial_pct = [df[df['model']==m]['initially_correct'].mean()*100 for m in models]
final_pct = [df[df['model']==m]['finally_correct'].mean()*100 for m in models]

bars1 = ax.bar(x - width/2, initial_pct, width, label='Before Challenge',
               color='#2ECC71', edgecolor='white', linewidth=2)
bars2 = ax.bar(x + width/2, final_pct, width, label='After False Authority',
               color='#E74C3C', edgecolor='white', linewidth=2)

# Add percentage labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('AI Model Accuracy: Before vs. After False Authority Challenge',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12, frameon=True, shadow=True)
ax.set_ylim([0, 110])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/accuracy_comparison.png")
plt.show()

# Summary stats
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
for model, label in zip(models, ['GPT-3.5', 'GPT-4', 'GPT-5']):
    m_data = df[df['model'] == model]
    initial = m_data['initially_correct'].mean() * 100
    final = m_data['finally_correct'].mean() * 100
    drop = initial - final
    print(f"\n{label}:")
    print(f"  Accuracy: {initial:.0f}% → {final:.0f}% (Δ {drop:.0f}%)")
    print(f"  Susceptibility: {drop:.0f}% of answers changed to incorrect")
