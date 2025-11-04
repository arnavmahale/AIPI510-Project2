"""Visualizations for Authority Resistance Experiment"""

## AI Tool Attribution: Built with assistance from Claude Code CLI (https://claude.ai/claude-code)
## Creates two visualizations:
## 1. Horizontal bar chart showing % of questions flipped to wrong answer per model
## 2. Pie chart showing the overall susceptibility rate across all models

import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load most recent data file
Path('results').mkdir(exist_ok=True)
filename = max(glob.glob('data/authority_resistance_factual_*.json'))
df = pd.DataFrame(json.load(open(filename)))

models = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-5-mini"]
labels = ['GPT-3.5\nTurbo', 'GPT-4\nTurbo', 'GPT-5\nMini']

initial_pct = [df[df['model']==m]['initially_correct'].mean()*100 for m in models]
final_pct = [df[df['model']==m]['finally_correct'].mean()*100 for m in models]

# 1. HORIZONTAL BAR CHART (showing % questions flipped to wrong answer)
fig, ax = plt.subplots(figsize=(10, 6))

drops = [initial_pct[i] - final_pct[i] for i in range(len(models))]

# Reverse the order: GPT-5 at top, GPT-3.5 at bottom
drops_reversed = drops[::-1]
labels_reversed = labels[::-1]
models_reversed = models[::-1]
final_pct_reversed = final_pct[::-1]

bar_colors_div = []
for i in range(len(models_reversed)):
    if final_pct_reversed[i] == 0:
        bar_colors_div.append('#E74C3C')  # Red - complete failure
    elif final_pct_reversed[i] == 100:
        bar_colors_div.append('#27AE60')  # Green - perfect resistance
    else:
        bar_colors_div.append('#F39C12')  # Yellow - mixed results

y_pos = np.arange(len(models_reversed))
bars = ax.barh(y_pos, drops_reversed, color=bar_colors_div, edgecolor='white', linewidth=2)

# Add percentage labels
for i, (bar, drop) in enumerate(zip(bars, drops_reversed)):
    width = bar.get_width()

    if width == 0:
        # Special handling for 0% - add a visual indicator
        ax.plot([0, 0], [bar.get_y(), bar.get_y() + bar.get_height()],
                color='#27AE60', linewidth=4, solid_capstyle='butt')
        ax.text(2, bar.get_y() + bar.get_height()/2, '0% - No Questions Flipped',
                ha='left', va='center', fontsize=12, fontweight='bold', color='#27AE60')
    else:
        # Regular percentage labels for non-zero bars
        label_x = width + 2
        ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{drop:.0f}%',
                ha='left', va='center', fontsize=14, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels_reversed, fontsize=13, fontweight='bold')
ax.set_xlabel('Questions Flipped to Wrong Answer (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Susceptibility to False Authority Varies by Generation', fontsize=16, fontweight='bold', pad=20)
ax.axvline(x=0, color='black', linewidth=1.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim([0, 100])
plt.tight_layout()
plt.savefig('results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/accuracy_comparison.png")
plt.close()

# 2. SUSCEPTIBILITY PIE CHART
fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
ax.set_facecolor('white')

initially_correct = df['initially_correct'].sum()
became_wrong = df['became_wrong'].sum()
pct_wrong = (became_wrong / initially_correct) * 100
pct_correct = 100 - pct_wrong

wedges, texts, autotexts = ax.pie([pct_wrong, pct_correct],
                                    labels=['Changed to Wrong', 'Stayed Correct'],
                                    colors=['#C84E00', '#003D7A'],
                                    autopct='%1.1f%%', startangle=90, explode=(0.05, 0),
                                    textprops={'fontsize': 14, 'weight': 'bold'},
                                    wedgeprops={'edgecolor': '#FFFFFF', 'linewidth': 2})

for text in texts:
    text.set_color('#2C3E50')
    text.set_fontsize(16)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(18)

ax.set_title(f'Overall Susceptibility Rate: {pct_wrong:.1f}%\n({became_wrong} out of {initially_correct} correct answers changed)',
             fontsize=16, fontweight='bold', color='#003D7A', pad=20)
plt.tight_layout()
plt.savefig('results/susceptibility_pie.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: results/susceptibility_pie.png")
plt.close()

