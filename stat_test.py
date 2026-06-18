import os

import matplotlib.pyplot as plt
import pandas as pd

frames_per_second = 60

folder = "tsv/clusters"


files = [folder + "/cluster_counts_per_frame_post1_seed_%d.csv" % i for i in range(1, 31)]

plt.figure(figsize=(12, 6))


all_counts = []

for file in files:
    # Read the file into a table.
    data = pd.read_csv(file)

    # Turn the frame number into seconds.
    time_in_seconds = data["frame"] / frames_per_second
    counts = data["num_unique_clusters"]

    # Draw every seed as a thin faint grey line in the background.
    plt.plot(time_in_seconds, counts, color="grey", alpha=0.15, linewidth=0.7)
    all_counts.append(counts)

# Put all the seed counts side by side so we can do stats per frame.
combined = pd.concat(all_counts, axis=1)
mean_counts = combined.mean(axis=1)
std_counts = combined.std(axis=1)

# Time axis (same for every seed).
time = pd.read_csv(files[0])["frame"] / frames_per_second

# Shaded band showing the spread (mean plus/minus one standard deviation).
plt.fill_between(time, mean_counts - std_counts, mean_counts + std_counts,
                 color="steelblue", alpha=0.3, label="± 1 std dev")

# Bold mean line on top.
plt.plot(time, mean_counts, color="navy", linewidth=2,
         label="mean of %d seeds" % len(files))

# Labels and styling.
plt.xlabel("Time (seconds)")
plt.ylabel("Number of clusters")
plt.title("Number of clusters over time")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot and show it.
plots_folder = os.path.join(folder, "plots")
os.makedirs(plots_folder, exist_ok=True)
save_path = os.path.join(plots_folder, "clusters_over_time.png")
plt.savefig(save_path, dpi=150)


plt.show()
