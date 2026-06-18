import os
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and aggregate all cluster count files
files = {
    'post0': 'cluster_counts_post0 (1).csv',
    'post1': 'cluster_counts_post1 (1).csv',
    'post2': 'cluster_counts_post2 (1).csv',
    'post3': 'cluster_counts_post3 (1).csv',
    'post4': 'cluster_counts_post4 (1).csv',
    'post5': 'cluster_counts_post5.csv'
}

df_list = []
for name, path in files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['post'] = name
        df_list.append(df)

df_all = pd.concat(df_list, ignore_index=True)

# 2. Perform global rank assignment for non-parametric evaluation
df_all['rank'] = stats.rankdata(df_all['num_clusters'])
mean_ranks = df_all.groupby('post')['rank'].mean().to_dict()

# 3. Global Omnibus Test: Kruskal-Wallis
groups = [df['num_clusters'].values for df in df_list]
h_stat, p_val = stats.kruskal(*groups)
print(f"Kruskal-Wallis Omnibus Test: H = {h_stat:.4f}, p = {p_val:.6f}")

# 4. Post-Hoc Analysis: Dunn's Test with Tie Corrections
N = len(df_all)
n_group = 30  # Each post condition contains exactly 30 seeds

# Calculate tie correction factor for variance
ties = df_all['num_clusters'].value_counts()
tie_sum = sum(t**3 - t for t in ties)
var_rank = (N * (N + 1) / 12) - (tie_sum / (12 * (N - 1)))

posts = sorted(list(mean_ranks.keys()))
dunn_results = []

for i in range(len(posts)):
    for j in range(i + 1, len(posts)):
        p1 = posts[i]
        p2 = posts[j]
        diff = abs(mean_ranks[p1] - mean_ranks[p2])
        
        # Standard error under the null hypothesis
        se = np.sqrt(var_rank * (1 / n_group + 1 / n_group))
        z_stat = diff / se
        p_unadj = 2 * (1 - stats.norm.cdf(z_stat))
        
        dunn_results.append({
            'Comparison': f"{p1} vs {p2}",
            'Rank Diff': diff,
            'Z-statistic': z_stat,
            'p_unadj': p_unadj
        })

df_dunn = pd.DataFrame(dunn_results)
# Bonferroni adjustment (15 unique pairwise comparisons)
df_dunn['p_bonf'] = (df_dunn['p_unadj'] * len(df_dunn)).clip(upper=1.0)
df_dunn['Significant'] = df_dunn['p_bonf'] < 0.05

# Save statistical tables
df_dunn.to_csv('dunn_test_results.csv', index=False)
print("\nPost-Hoc Dunn's Test Results saved to 'dunn_test_results.csv'")