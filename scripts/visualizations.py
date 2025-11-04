"""Visualizations for Authority Resistance Experiment"""

## AI Tool Attribution: Built with assistance from Claude Code CLI (https://claude.ai/claude-code)
## Guided Claude to create visualizations that tell the story: a bar chart comparing accuracy
## before and after the false authority challenge, a pie chart showing the overall susceptibility
## rate, and a bar chart demonstrating we met power analysis requirements.

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

# 1. ACCURACY COMPARISON
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
width = 0.35

initial_pct = [df[df['model']==m]['initially_correct'].mean()*100 for m in models]
final_pct = [df[df['model']==m]['finally_correct'].mean()*100 for m in models]

bars1 = ax.bar(x - width/2, initial_pct, width, label='Before Challenge', color='#2ECC71', edgecolor='white', linewidth=2)
bars2 = ax.bar(x + width/2, final_pct, width, label='After False Authority', color='#E74C3C', edgecolor='white', linewidth=2)

# Add percentage labels to all bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.0f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add percentage labels and status labels to "after" bars
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.0f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add status labels
    if height == 0:
        status = '[X] COMPLETE\nFAILURE'
        y_pos = 8
        text_color = 'black'
    elif height == 100:
        status = '[✓] PERFECT\nRESISTANCE'
        y_pos = height - 15
        text_color = 'white'
    else:
        status = '[!] MIXED\nRESULTS'
        y_pos = height / 2
        text_color = 'white'

    ax.text(bar.get_x() + bar.get_width()/2., y_pos, status,
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=text_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E74C3C', alpha=0.3, edgecolor='none'))

ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Response Accuracy: Before vs. After False Authority Challenge', fontsize=16, fontweight='bold', pad=20)
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

