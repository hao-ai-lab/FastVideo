#!/usr/bin/env python3
"""Visualize FastVideo pipeline stage timing breakdown for 8 GPUs."""

import matplotlib.pyplot as plt
import numpy as np

# Data from data.txt - GPUs=8 row
stage_names = [
    "InputValidationStage",
    "LTX2TextEncodingStage", 
    "LTX2LatentPreparationStage",
    "LTX2DenoisingStage",
    "LTX2AudioDecodingStage",
    "DecodingStage",
]

# Times in ms from GPUs=8 row
times_ms = [0.05226, 92.02112, 105.87283, 5051.15117, 12.70169, 1887.34311]

# Convert to seconds
times_sec = [t / 1000 for t in times_ms]

# Shorter labels for display
short_names = [
    "Input Validation",
    "Text Encoding",
    "Latent Preparation", 
    "Denoising",
    "Audio Decoding",
    "Video Decoding",
]

# Pastel colors
pastel_colors = [
    '#FFB3BA',  # pastel pink
    '#BAFFC9',  # pastel green
    '#BAE1FF',  # pastel blue
    '#FFFFBA',  # pastel yellow
    '#FFD9BA',  # pastel orange
    '#E0BBE4',  # pastel purple
]

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("LTX2 Pipeline Timing (8 B200 GPUs)", fontsize=14, fontweight='bold')

# Left: Horizontal bar chart
y_pos = np.arange(len(short_names))
bars = ax1.barh(y_pos, times_sec, color=pastel_colors, edgecolor='gray', linewidth=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(short_names)
ax1.set_xlabel("Duration (seconds)")
ax1.set_title("Stage Duration Breakdown")
ax1.invert_yaxis()  # Largest at top
ax1.set_xlim(0, max(times_sec) * 1.25)  # Add 25% padding for labels

# Add value labels on bars
for bar, val in zip(bars, times_sec):
    ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
            f'{val:.3f}s', va='center', fontsize=9)

# Right: Pie chart - use legend instead of labels to avoid overlap
def autopct_func(pct):
    return f'{pct:.1f}%' if pct > 2 else ''

wedges, texts, autotexts = ax2.pie(
    times_sec, 
    colors=pastel_colors,
    autopct=autopct_func,
    startangle=90,
    pctdistance=0.75,
    wedgeprops={'edgecolor': 'gray', 'linewidth': 0.5},
)
ax2.set_title("Stage Time Distribution")

# Add legend to avoid label overlap
ax2.legend(wedges, short_names, title="Stages", loc="center left", 
           bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)

# Add total time annotation
total_sec = sum(times_sec)
fig.text(0.5, 0.02, f"Total: {total_sec:.3f}s ({total_sec*1000:.2f}ms)", 
         ha='center', fontsize=11, fontweight='bold')

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("timing_8gpus.png", dpi=150, bbox_inches='tight')
print(f"Saved figure to timing_8gpus.png")
plt.show()
