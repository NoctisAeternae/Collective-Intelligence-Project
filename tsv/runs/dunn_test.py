import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikit_posthocs as sp

# 1. Load and combine the data
files = [
    "cluster_counts_post0.csv", "cluster_counts_post1.csv", "cluster_counts_post2.csv",
    "cluster_counts_post3.csv", "cluster_counts_post4.csv", "cluster_counts_post5.csv"
]
labels_map = {0: "centre-left", 1: "even split", 2: "centre-right", 3: "right", 4: "middle", 5: "left"}

df_list = []
for i, file in enumerate(files):
    df = pd.read_csv(file)
    df['Condition'] = labels_map[i]
    df_list.append(df)
all_data = pd.concat(df_list, ignore_index=True)

# 2. Define the correct order
desired_order = ['left', 'centre-left', 'middle', 'centre-right', 'right', 'even split']

# 3. Perform Dunn's Test
# p_adjust='bonferroni' handles the correction for multiple comparisons automatically
dunn_results = sp.posthoc_dunn(
    all_data, 
    val_col='num_clusters', 
    group_col='Condition', 
    p_adjust='bonferroni'
)

# 4. Reorder the result matrix to match your desired order
dunn_results = dunn_results.reindex(index=desired_order, columns=desired_order)

# 5. Plot the Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(dunn_results, annot=True, cmap="coolwarm_r", vmin=0, vmax=1, fmt=".3f", square=True, linewidths=.5)
plt.title('Dunn\'s Test Adjusted p-values')
plt.tight_layout()
plt.savefig('dunn_library_heatmap.png')
plt.show()