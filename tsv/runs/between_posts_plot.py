import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
files = [
    "cluster_counts_post0.csv",
    "cluster_counts_post1.csv",
    "cluster_counts_post2.csv",
    "cluster_counts_post3.csv",
    "cluster_counts_post4.csv",
    "cluster_counts_post5.csv"
]

# Map the file index to the designated label
labels_map = {
    0: "centre-left",
    1: "even split",
    2: "centre-right",
    3: "right",
    4: "middle",
    5: "left"
}

df_list = []
for i, file in enumerate(files):
    df = pd.read_csv(file)
    df['Condition'] = labels_map[i]
    df_list.append(df)

all_data = pd.concat(df_list, ignore_index=True)

# 2. Define your specific left-to-right spectrum order
desired_order = ['left', 'centre-left', 'middle', 'centre-right', 'right', 'even split']

# 3. Plot: Mean + Standard Deviation explicitly
plt.figure(figsize=(10, 6))

sns.pointplot(
    x='Condition', y='num_clusters', data=all_data, 
    order=desired_order, # Forces the X-axis to use your custom spectrum
    errorbar='sd', capsize=0.1, join=False, markers='o'
)

plt.title('Mean and Standard Deviation of Final Opinion Number of Clusters per Post')
plt.ylabel('Number of Opinion Clusters')
plt.xlabel('Initial Opinion Distribution')
plt.xticks(rotation=45, ha='right') # Tilts the text 45 degrees so the longer labels fit perfectly
plt.tight_layout()
plt.savefig('cluster_std_dev_ordered.png')
plt.show()