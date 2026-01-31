#!/usr/bin/env python3
"""Grouped bar chart comparing pipeline stage timings across GPU configurations."""

import matplotlib.pyplot as plt
import numpy as np

# Data from data.txt - all rows (times in ms)
gpu_configs = [1, 2, 4, 8]

# Stage names (short)
short_names = [
    "Input Validation",
    "Text Encoding",
    "Latent Preparation", 
    "Denoising",
    "Audio Decoding",
    "Video Decoding",
]

# Times in ms for each GPU config (rows from data.txt)
times_data = {
    1: [0.05677, 88.77044, 101.66176, 24276.15683, 12.70252, 3331.84156],
    2: [0.0638, 94.22693, 117.48425, 13778.04096, 15.30154, 3124.44973],
    4: [0.05684, 89.72921, 116.69089, 8158.99098, 13.71993, 2459.41875],
    8: [0.05226, 92.02112, 105.87283, 5051.15117, 12.70169, 1887.34311],
}

# Convert to seconds
times_sec = {k: [t / 1000 for t in v] for k, v in times_data.items()}

# Pastel colors for each GPU config
gpu_colors = {
    1: '#FFB3BA',  # pastel pink
    2: '#BAFFC9',  # pastel green
    4: '#BAE1FF',  # pastel blue
    8: '#E0BBE4',  # pastel purple
}

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Bar parameters
n_stages = len(short_names)
n_configs = len(gpu_configs)
bar_width = 0.18
y_pos = np.arange(n_stages)

# Draw grouped horizontal bars
for i, gpu in enumerate(gpu_configs):
    offset = (i - n_configs/2 + 0.5) * bar_width
    bars = ax.barh(y_pos + offset, times_sec[gpu], bar_width, 
                   label=f'{gpu} GPU{"s" if gpu > 1 else ""}',
                   color=gpu_colors[gpu], edgecolor='gray', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(short_names)
ax.set_xlabel("Duration (seconds)", fontsize=12)
ax.set_title("LTX2 Pipeline Stage Latency by B200 GPU Count", fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.legend(title="Configuration", loc='lower right')
ax.set_xlim(0, max(max(v) for v in times_sec.values()) * 1.15)
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("timing_all_gpus.png", dpi=150, bbox_inches='tight')
print(f"Saved figure to timing_all_gpus.png")
plt.show()
