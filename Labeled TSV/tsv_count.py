import pandas as pd

file_path = 'comments_ALBANIA_3mnulr45aok2q_labeled.tsv'
df = pd.read_csv(file_path, sep='\t')

counts = {}
for label in df['political_position']:
    counts[label] = counts.get(label, 0) + 1

print(counts)
        