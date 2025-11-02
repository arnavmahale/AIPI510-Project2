"""Visualizations for Authority Resistance Experiment"""

## AI Tool Attribution: Built with assistance from Claude Code CLI (https://claude.ai/claude-code)
## Guided Claude to create visualizations that tell the story: a bar chart comparing accuracy
## before and after the false authority challenge, a pie chart showing the overall susceptibility
## rate, gauge charts to illustrate confidence levels, and a bar chart demonstrating we met
## power analysis requirements.

import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Load most recent data file
Path('results').mkdir(exist_ok=True)
filename = max(glob.glob('data/authority_resistance_factual_*.json'))
df = pd.DataFrame(json.load(open(filename)))

models = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-5-mini"]
labels = ['GPT-3.5\nTurbo', 'GPT-4\nTurbo', 'GPT-5\nMini']

# 1. PERFECT GRADIENT: SUSCEPTIBILITY BY MODEL (★ STAR VISUAL)
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(models))

# Calculate susceptibility rate per model
susceptibility_pct = []
for m in models:
    model_df = df[df['model'] == m]
    susceptibility = (model_df['became_wrong'].sum() / len(model_df)) * 100
    susceptibility_pct.append(susceptibility)

# Color gradient: Red (high) → Yellow (medium) → Green (low)
colors = ['#E74C3C', '#F39C12', '#2ECC71']
edge_colors = ['#C0392B', '#D68910', '#27AE60']

bars = ax.bar(x, susceptibility_pct, width=0.6, color=colors, edgecolor=edge_colors, linewidth=3)

# Add percentage labels on bars
for i, (bar, pct) in enumerate(zip(bars, susceptibility_pct)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2, f'{pct:.0f}%',
            ha='center', va='bottom', fontsize=24, fontweight='bold', color='#2C3E50')

    # Add status labels
    if pct == 100:
        status = '[X] COMPLETE\nFAILURE'
        y_pos = height - 15
    elif pct == 0:
        status = '[✓] PERFECT\nRESISTANCE'
        y_pos = 10
    else:
        status = '[!] MIXED\nRESULTS'
        y_pos = height - 10

    ax.text(bar.get_x() + bar.get_width()/2., y_pos, status,
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='white' if pct > 20 else '#2C3E50',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3, edgecolor='none'))

ax.set_ylabel('Susceptibility to False Authority (%)', fontsize=14, fontweight='bold')
ax.set_title('Perfect Gradient: 100% → 47% → 0%\nModel Capability Determines Authority Resistance',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
ax.set_ylim([0, 115])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/susceptibility_gradient.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/susceptibility_gradient.png (★ STAR VISUAL)")
plt.close()

# 2. ACCURACY COMPARISON
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
width = 0.35

initial_pct = [df[df['model']==m]['initially_correct'].mean()*100 for m in models]
final_pct = [df[df['model']==m]['finally_correct'].mean()*100 for m in models]

bars1 = ax.bar(x - width/2, initial_pct, width, label='Before Challenge', color='#2ECC71', edgecolor='white', linewidth=2)
bars2 = ax.bar(x + width/2, final_pct, width, label='After False Authority', color='#E74C3C', edgecolor='white', linewidth=2)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.0f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('AI Model Accuracy: Before vs. After False Authority Challenge', fontsize=16, fontweight='bold', pad=20)
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

# 3. POWER ANALYSIS
fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
ax.set_facecolor('white')

total_collected = len(df)
minimum_needed = 9
values = [minimum_needed, total_collected]

bars = ax.bar(['Minimum\nNeeded', 'We\nCollected'], values, color=['#CCCCCC', '#003D7A'],
              width=0.5, edgecolor='#2C3E50', linewidth=2)

for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.5, f'{int(value)}',
            ha='center', va='bottom', fontsize=36, fontweight='bold', color='#2C3E50')

ax.text(1, total_collected + 5, '✓', fontsize=60, ha='center', color='#2ECC71', fontweight='bold')
ax.set_ylim(0, total_collected + 10)
ax.set_ylabel('Number of Observations', fontsize=14, fontweight='bold', color='#2C3E50')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--', color='#E5E7EB')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('results/power_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: results/power_analysis.png")
plt.close()

# 4. SUSCEPTIBILITY PIE CHART
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

# 5. CONFIDENCE GAUGES
def draw_gauge(ax, confidence, title, color):
    ax.set_facecolor('white')
    theta = np.linspace(np.pi, 0, 100)
    x, y = np.cos(theta), np.sin(theta)

    ax.plot(x, y, color='#E5E7EB', linewidth=20)

    theta_conf = np.linspace(np.pi, np.pi * (1 - confidence / 10), 100)
    x_conf, y_conf = np.cos(theta_conf), np.sin(theta_conf)
    ax.plot(x_conf, y_conf, color=color, linewidth=20)

    needle_angle = np.pi * (1 - confidence / 10)
    ax.plot([0, np.cos(needle_angle)], [0, np.sin(needle_angle)], color=color, linewidth=4)
    ax.add_patch(patches.Circle((0, 0), 0.08, color=color))

    for i in range(0, 11, 2):
        angle = np.pi * (1 - i / 10)
        ax.text(np.cos(angle) * 1.25, np.sin(angle) * 1.25, str(i),
                ha='center', va='center', fontsize=12, fontweight='bold', color='#2C3E50')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(0, -0.35, title, ha='center', va='top', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.text(0, 0.5, f'{confidence:.1f}/10', ha='center', va='center', fontsize=28, fontweight='bold', color=color)

wrong_answers = df[df['became_wrong'] == True]
correct_answers = df[(df['initially_correct'] == True) & (df['became_wrong'] == False)]
avg_confidence_wrong = wrong_answers['final_confidence'].mean() if len(wrong_answers) > 0 else 0
avg_confidence_correct = correct_answers['initial_confidence'].mean() if len(correct_answers) > 0 else 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), facecolor='white')
draw_gauge(ax1, avg_confidence_wrong, 'Avg Confidence in\nWrong Answer\n(Vulnerable Models)', '#C84E00')
draw_gauge(ax2, avg_confidence_correct, 'Avg Confidence in\nCorrect Answer\n(Resistant Models)', '#003D7A')
plt.suptitle('The Confidence Illusion in Vulnerable Models\nHigh Certainty ≠ Accuracy After Manipulation',
             fontsize=16, fontweight='bold', color='#003D7A', y=0.98)
plt.tight_layout()
plt.savefig('results/confidence_gauge.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: results/confidence_gauge.png")
plt.close()
