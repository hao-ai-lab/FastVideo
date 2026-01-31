#!/usr/bin/env python3
"""Grouped bar chart comparing pipeline stage timings across GPU configurations on H200."""

import matplotlib.pyplot as plt
import numpy as np
import os

# Read data from data.txt
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, "data.txt")

with open(data_file, 'r') as f:
    lines = f.readlines()

# Parse header
header = lines[0].strip().split('\t')
stage_names = header[1:]  # Skip 'GPUs' column

# Parse all data rows
times_data = {}
gpu_configs = []
for line in lines[1:]:
    line = line.strip()
    if not line:
        continue
    parts = line.split('\t')
    gpu_count = int(parts[0])
    gpu_configs.append(gpu_count)
    times_data[gpu_count] = [float(x) for x in parts[1:]]

# Sort GPU configs
gpu_configs.sort()

# Stage names (short)
short_names = [
    "Input Validation",
    "Text Encoding",
    "Latent Preparation", 
    "Denoising",
    "Audio Decoding",
    "Video Decoding",
]

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
                   color=gpu_colors.get(gpu, '#CCCCCC'), edgecolor='gray', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(short_names)
ax.set_xlabel("Duration (seconds)", fontsize=12)
ax.set_title("LTX2 Pipeline Stage Latency by H200 GPU Count", fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.legend(title="Configuration", loc='lower right')
ax.set_xlim(0, max(max(v) for v in times_sec.values()) * 1.15)
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "timing_all_gpus_h200.png"), dpi=150, bbox_inches='tight')
print(f"Saved figure to timing_all_gpus_h200.png")
plt.show()
