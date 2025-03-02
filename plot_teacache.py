import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = "./teacache_stats.npy"
data = np.load(file_path)

# Ensure the data shape is as expected
if data.ndim != 2 or data.shape[0] != 4:
    raise ValueError("Unexpected data shape. Expected (4, N), but got {}".format(data.shape))

# Define labels and colors
labels = ["Noisy Input", "Modulated Noisy Input", "Timestep Embeddings", "Residual Model Output"]
colors = ["red", "green", "orange", "blue"]

# Plot the data
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.plot(data[i][1::2], label=labels[i], color=colors[i])

plt.xlabel("Diffusion timestep")
plt.ylabel("Difference (L1 rel)")
plt.title("Wan2.1 1.3B L1 Rel Difference")
plt.legend()
plt.grid(True)
plt.savefig("teacache_stats.png")