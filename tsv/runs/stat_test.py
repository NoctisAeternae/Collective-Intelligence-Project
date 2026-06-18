import glob
import os

import pandas as pd
from scipy import stats

FILE_PATTERN = "cluster_counts_post*.csv"
VALUE_COLUMN = "num_clusters"
ALPHA = 0.05  # significance threshold: p-values below this count as "significant"

def load_groups():
    """
    Read every CSV file and return a dictionary like:
        {"post0": <numbers>, "post2": <numbers>, ...}

    The key is the post name (taken from the file name) and the value is the
    column of num_clusters numbers from that file.
    """
    file_paths = sorted(glob.glob(FILE_PATTERN))

    if len(file_paths) < 2:
        raise SystemExit(
            f"Need at least 2 files matching '{FILE_PATTERN}', "
            f"but found {len(file_paths)}."
        )

    groups = {}
    for path in file_paths:
        # Build a short label from the file name, e.g.
        # "cluster_counts_post2.csv" -> "post2"
        file_name = os.path.basename(path)
        label = file_name.replace("cluster_counts_", "").replace(".csv", "")

        # Read the file into a table (a pandas DataFrame).
        table = pd.read_csv(path)

        if VALUE_COLUMN not in table.columns:
            raise SystemExit(
                f"File '{path}' has no column called '{VALUE_COLUMN}'. "
                f"Columns found: {list(table.columns)}"
            )

        # Keep only the column we care about, and drop any blank rows.
        values = table[VALUE_COLUMN].dropna()
        groups[label] = values

    return groups


def print_summary(groups):
    """Print count, mean, median, min, and max for each post."""
    print("Summary of num_clusters per post:")
    print(f"{'post':<8}{'n':>5}{'mean':>9}{'median':>9}{'min':>6}{'max':>6}")

    for label, values in groups.items():
        print(
            f"{label:<8}"
            f"{values.size:>5}"
            f"{values.mean():>9.3f}"
            f"{values.median():>9.3f}"
            f"{values.min():>6}"
            f"{values.max():>6}"
        )
    print()


def run_kruskal(groups):
    """
    Run the Kruskal-Wallis test across all posts.
    Returns the p-value so the caller can decide what to do next.
    """
    # stats.kruskal needs each group as a separate argument. We collect the
    # number-lists into a list, then unpack it with the * symbol.
    list_of_value_arrays = []
    for values in groups.values():
        list_of_value_arrays.append(values.values)

    h_statistic, p_value = stats.kruskal(*list_of_value_arrays)

    post_names = ", ".join(groups.keys())
    print(f"Kruskal-Wallis test across {len(groups)} posts: {post_names}")
    print(f"  H statistic = {h_statistic:.4f}")
    print(f"  p-value     = {p_value:.4g}")

    if p_value < ALPHA:
        print(f"  Result: significant (p < {ALPHA}) -> the posts differ.")
    else:
        print(f"  Result: not significant (p >= {ALPHA}) -> no clear difference.")
    print()

    return p_value


def run_posthoc(groups):
    """
    Dunn's test: when the overall test is significant, this tells us which
    PAIRS of posts are different from each other. Each number in the table is
    a p-value for one pair (small value = that pair differs).
    """
    try:
        import scikit_posthocs as sp
    except ImportError:
        print("Skipping pairwise test. Install it with: pip install scikit-posthocs")
        return

    # Dunn's test wants the data in "long" format: one row per value, with a
    # column saying which post it came from. We build that table row-group by
    # row-group.
    pieces = []
    for label, values in groups.items():
        piece = pd.DataFrame({VALUE_COLUMN: values.values, "post": label})
        pieces.append(piece)
    long_table = pd.concat(pieces, ignore_index=True)

    result = sp.posthoc_dunn(
        long_table,
        val_col=VALUE_COLUMN,
        group_col="post",
        p_adjust="bonferroni",  # correction for testing many pairs at once
    )

    print("Pairwise comparison (Dunn's test, Bonferroni-adjusted p-values):")
    print(result.to_string())


def main():
    groups = load_groups()
    print_summary(groups)
    p_value = run_kruskal(groups)

    # Only bother with pairwise comparisons if the overall test found something.
    if p_value < ALPHA:
        run_posthoc(groups)


if __name__ == "__main__":
    main()