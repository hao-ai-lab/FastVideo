#!/usr/bin/env python3
"""Line plot of DecodingStage timing across GPU configurations."""

import matplotlib.pyplot as plt
import numpy as np

# Data from data.txt - DecodingStage column (in ms)
gpus = [1, 2, 4, 8]
decoding_times_ms = [3331.84156, 3124.44973, 2459.41875, 1887.34311]

# Convert to seconds
decoding_times_sec = [t / 1000 for t in decoding_times_ms]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Pastel blue color
color = '#BAE1FF'

# Line plot with markers
ax.plot(gpus, decoding_times_sec, marker='o', markersize=10, linewidth=2.5,
        color='#6699CC', markerfacecolor=color, markeredgecolor='#336699', 
        markeredgewidth=2)

# Add value labels on points
for x, y in zip(gpus, decoding_times_sec):
    ax.annotate(f'{y:.2f}s', (x, y), textcoords="offset points", 
                xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')

ax.set_xlabel("Number of GPUs", fontsize=12)
ax.set_ylabel("Decoding Time (seconds)", fontsize=12)
ax.set_title("LTX2 DecodingStage (VAE) Latency vs GPU Count", fontsize=14, fontweight='bold')
ax.set_xticks(gpus)
ax.set_xticklabels([str(g) for g in gpus])
ax.grid(True, alpha=0.3)

# Set y-axis limits with padding
ax.set_ylim(0, max(decoding_times_sec) * 1.2)

plt.tight_layout()
plt.savefig("decoding_vs_gpus.png", dpi=150, bbox_inches='tight')
print(f"Saved figure to decoding_vs_gpus.png")
plt.show()
