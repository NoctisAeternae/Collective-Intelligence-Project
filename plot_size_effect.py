import os

import matplotlib.pyplot as plt
import pandas as pd

frames_per_second = 60

folder = "tsv/clusters"

# One file per agent count (single seed each).
sizes = [25, 50, 100, 200, 400]
files = {
    s: folder + "/cluster_counts_per_frame_size_effect_%d_seed_0.csv" % s
    for s in sizes
}

# Light smoothing so the bold line is readable (raw counts are noisy integers).
smooth_window = 200

# A distinct colour per agent count, ordered light -> dark with more agents.
colors = plt.cm.viridis([i / (len(sizes) - 1) for i in range(len(sizes))])

plt.figure(figsize=(12, 6))

for size, color in zip(sizes, colors):
    # Read the file into a table.
    data = pd.read_csv(files[size])

    # Turn the frame number into seconds.
    time_in_seconds = data["frame"] / frames_per_second
    counts = data["num_unique_clusters"]

    # Smoothed version (rolling mean) for the bold line.
    smoothed = counts.rolling(smooth_window, center=True, min_periods=1).mean()

    # Draw the raw counts as a thin faint line in the background.
    plt.plot(time_in_seconds, counts, color=color, alpha=0.15, linewidth=0.7)

    # Bold smoothed line on top.
    plt.plot(time_in_seconds, smoothed, color=color, linewidth=2,
             label="%d agents" % size)

# Labels and styling.
plt.xlabel("Time (seconds)")
plt.ylabel("Number of opinion clusters")
plt.title("Number of opinion clusters over time (finite-size effect)")
plt.legend(title="Agent count")
plt.grid(True, alpha=0.3)

# Save the plot and show it.
plots_folder = os.path.join(folder, "plots")
os.makedirs(plots_folder, exist_ok=True)
save_path = os.path.join(plots_folder, "clusters_over_time_size_effect.png")
plt.savefig(save_path, dpi=150)

plt.show()
