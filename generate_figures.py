#!/usr/bin/env python3
"""Generate all figures for the project report."""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "./output"

# Load results
with open(os.path.join(OUTPUT_DIR, "results.json")) as f:
    results = json.load(f)

with open(os.path.join(OUTPUT_DIR, "qualitative_examples.json")) as f:
    examples = json.load(f)

# ============================================================
# Figure 1: Training Loss Curve (improved version)
# ============================================================
all_losses = results["all_step_losses"]
epoch_eval_losses = results["epoch_eval_losses"]
steps_per_epoch = len(all_losses) // len(epoch_eval_losses)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# Smooth the training loss with moving average
window = 20
smoothed = np.convolve(all_losses, np.ones(window)/window, mode='valid')
ax.plot(range(window, len(all_losses)+1), smoothed, 'b-', linewidth=2, label='Training Loss (smoothed)', alpha=0.9)
ax.plot(range(1, len(all_losses)+1), all_losses, 'b-', linewidth=0.3, alpha=0.25, label='Training Loss (raw)')

# Eval loss
eval_x = [(i + 1) * steps_per_epoch for i in range(len(epoch_eval_losses))]
ax.plot(eval_x, epoch_eval_losses, 'r-o', linewidth=2.5, markersize=10, label='Eval Loss (per epoch)', zorder=5)

for i, (x, y) in enumerate(zip(eval_x, epoch_eval_losses)):
    ax.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(10, 10),
                fontsize=10, color='red', fontweight='bold')

ax.set_xlabel("Training Steps", fontsize=13)
ax.set_ylabel("Loss", fontsize=13)
ax.set_title("OFT Fine-tuning: Training & Evaluation Loss Curves", fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=-0.05)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_training_loss.png"), dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig1_training_loss.png")


# ============================================================
# Figure 2: Before vs After Accuracy Comparison (Bar Chart)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Bar chart
models = ['Base Model\n(Zero-shot)', 'OFT Fine-tuned']
accs = [results['base_accuracy'] * 100, results['finetuned_accuracy'] * 100]
colors = ['#e74c3c', '#2ecc71']

bars = axes[0].bar(models, accs, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
for bar, acc in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=16, fontweight='bold')

axes[0].set_ylabel("Accuracy (%)", fontsize=13)
axes[0].set_title("SST-2 Sentiment Classification Accuracy", fontsize=14, fontweight='bold')
axes[0].set_ylim(0, 110)
axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# Right: Parameter efficiency pie chart
trainable = 8_214_528
frozen = 1_551_928_832 - trainable

axes[1].pie([trainable, frozen],
            labels=[f'Trainable (OFT)\n{trainable/1e6:.1f}M (0.53%)',
                    f'Frozen\n{frozen/1e6:.1f}M (99.47%)'],
            colors=['#3498db', '#ecf0f1'],
            explode=(0.05, 0),
            autopct='',
            startangle=90,
            textprops={'fontsize': 11},
            wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
axes[1].set_title("Parameter Efficiency", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_accuracy_comparison.png"), dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig2_accuracy_comparison.png")


# ============================================================
# Figure 3: Qualitative Results Table (as image)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')

# Find interesting examples (where base != finetuned, or base wrong)
interesting = []
correct_both = []
for ex in examples:
    row = {
        'sentence': ex['sentence'][:65] + ('...' if len(ex['sentence']) > 65 else ''),
        'gt': ex['ground_truth'],
        'base': ex['base_prediction'],
        'oft': ex['finetuned_prediction'],
        'base_correct': '✓' if ex['base_prediction'] == ex['ground_truth'] else '✗',
        'oft_correct': '✓' if ex['finetuned_prediction'] == ex['ground_truth'] else '✗',
    }
    if ex['base_prediction'] != ex['finetuned_prediction'] or ex['base_prediction'] != ex['ground_truth']:
        interesting.append(row)
    else:
        correct_both.append(row)

# Show mix of interesting + correct examples (max 12 rows)
display_rows = interesting[:6] + correct_both[:6]

table_data = []
for r in display_rows:
    table_data.append([r['sentence'], r['gt'], r['base'], r['base_correct'], r['oft'], r['oft_correct']])

col_labels = ['Review (truncated)', 'Ground\nTruth', 'Base\nModel', '', 'OFT\nModel', '']
table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='left')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

# Header styling
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(color='white', fontweight='bold')

# Row coloring
for i in range(1, len(display_rows) + 1):
    row_data = display_rows[i-1]
    bg_color = '#f8f9fa' if i % 2 == 0 else 'white'
    for j in range(len(col_labels)):
        table[i, j].set_facecolor(bg_color)

    # Color the check/cross marks
    if row_data['base_correct'] == '✗':
        table[i, 2].set_text_props(color='red')
        table[i, 3].set_text_props(color='red', fontweight='bold')
    else:
        table[i, 3].set_text_props(color='green', fontweight='bold')

    if row_data['oft_correct'] == '✗':
        table[i, 4].set_text_props(color='red')
        table[i, 5].set_text_props(color='red', fontweight='bold')
    else:
        table[i, 5].set_text_props(color='green', fontweight='bold')

# Column widths
col_widths = [0.50, 0.10, 0.12, 0.04, 0.12, 0.04]
for j, w in enumerate(col_widths):
    for i in range(len(display_rows) + 1):
        table[i, j].set_width(w)

ax.set_title("Qualitative Results: Before vs After OFT Fine-tuning", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_qualitative_results.png"), dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig3_qualitative_results.png")

print("\nAll figures generated successfully!")
