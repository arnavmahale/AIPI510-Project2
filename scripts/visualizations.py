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

# Color bars based on performance: Red (failure), Yellow (mixed), Green (perfect)
bar_colors = []
for pct in final_pct:
    if pct == 0:
        bar_colors.append('#E74C3C')  # Red for complete failure
    elif pct == 100:
        bar_colors.append('#27AE60')  # Darker green for perfect resistance
    else:
        bar_colors.append('#F39C12')  # Yellow for mixed results

bars2 = ax.bar(x + width/2, final_pct, width, label='After False Authority', color=bar_colors, edgecolor='white', linewidth=2)

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
        label_color = '#E74C3C'  # Red for failure
    elif height == 100:
        status = '[✓] PERFECT\nRESISTANCE'
        y_pos = height - 15
        text_color = 'white'
        label_color = '#27AE60'  # Darker green for perfect resistance
    else:
        status = '[!] MIXED\nRESULTS'
        y_pos = height / 2
        text_color = 'black'
        label_color = '#F39C12'  # Yellow for mixed results

    ax.text(bar.get_x() + bar.get_width()/2., y_pos, status,
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=text_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=label_color, alpha=0.3, edgecolor='none'))

ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Response Accuracy: Before vs. After False Authority Challenge', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylim([0, 110])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/accuracy_comparison.png")
plt.close()

# 2. SLOPE CHART
fig, ax = plt.subplots(figsize=(10, 6))

# Spread out the left-side labels to avoid overlap when all start at 100%
# Order: GPT-3.5 (bottom/red), GPT-4 (middle/yellow), GPT-5 (top/green) to match final positions
label_y_positions = [85, 100, 115]  # Stagger labels vertically with more spacing

for i, model in enumerate(models):
    # Determine color based on final performance
    final = final_pct[i]
    if final == 0:
        color = '#E74C3C'  # Red for complete failure
        linestyle = '-'
        linewidth = 3
    elif final == 100:
        color = '#27AE60'  # Green for perfect resistance
        linestyle = '-'
        linewidth = 3
    else:
        color = '#F39C12'  # Yellow for mixed
        linestyle = '-'
        linewidth = 3

    # Draw line connecting before and after
    ax.plot([0, 1], [initial_pct[i], final_pct[i]],
            color=color, linewidth=linewidth, linestyle=linestyle,
            marker='o', markersize=10, markeredgewidth=2, markeredgecolor='white')

    # Add model labels (spread vertically to avoid overlap)
    ax.text(-0.05, label_y_positions[i], labels[i], ha='right', va='center',
            fontsize=12, fontweight='bold', color=color)

    # Add percentage labels on the right side only (to reduce clutter)
    ax.text(1.02, final_pct[i], f'{final_pct[i]:.0f}%', ha='left', va='center',
            fontsize=11, fontweight='bold', color=color)

ax.set_xlim([-0.35, 1.35])
ax.set_ylim([-5, 125])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Before Challenge', 'After False Authority'], fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Response Accuracy: Slope Chart Showing Authority Resistance', fontsize=16, fontweight='bold', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('results/accuracy_slope_chart.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/accuracy_slope_chart.png")
plt.close()

# 3. SUSCEPTIBILITY PIE CHART
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

