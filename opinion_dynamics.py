from dataclasses import dataclass

from vi import Agent, Simulation, HeadlessSimulation
from vi.config import Config
import random
import polars as pl
import matplotlib.pyplot as plt
import scipy.stats as ss
import scikit_posthocs as sp
import pandas as pd

@dataclass
class OpinionConfig(Config): ...

BELIEFS = {"left": 0, "center-left": 1, "middle": 2, "center-right": 3, "right": 4}
BELIEF_TO_PROB = {0: 0.9, 1: 0.833, 2: 0.5, 3: 0.833, 4: 0.9}

IMAGES = ["images/left.png", "images/center-left.png", "images/middle.png", "images/center-right.png", "images/right.png"]

SEEDS = [i for i in range(1, 31)]

CODE_TO_NAME = {0: "left", 1: "center-left", 2: "middle", 3: "center-right", 4: "right"}
COLORS = {0: "#2166ac", 1: "#92c5de", 2: "#999999", 3: "#f4a582", 4: "#b2182b"}

POSTS = []

class OpinionAgent(Agent[OpinionConfig]):

    def change_belief(self, agent):

        num = random.random()

        if num > self.change_belief_prob:
            self.belief = agent.belief
            self.change_belief_prob = BELIEF_TO_PROB[self.belief]
            self.change_image(self.belief)
        else:
            pass
    
    def update(self):

        current_agents = {}
        for agent, _ in self.in_proximity_accuracy():
            current_agents[agent.id] = agent
        
        just_entered = current_agents.keys() - self.in_range

        for agent_id in just_entered:
            agent = current_agents[agent_id]
            if agent.belief != self.belief:
                self.change_belief(agent)
            else:
                pass

        self.in_range = set(current_agents.keys())

def plot_belief_counts(counts_long, fps=60, every=10, x_unit="minutes", save_path=None):
    thinned = counts_long.filter(pl.col("frame") % every == 0)   # ~1080 pts/line, not 108k
    fig, ax = plt.subplots(figsize=(9, 5))
    for code in range(5):
        sub = thinned.filter(pl.col("image_index") == code).sort("frame")
        x = sub["frame"].to_numpy().astype(float)
        if x_unit == "minutes": x = x / fps / 60
        elif x_unit == "seconds": x = x / fps
        ax.plot(x, sub["count"].to_numpy(),
                label=CODE_TO_NAME[code], color=COLORS[code], linewidth=1.8)
    xlabel = {"minutes": "Time (minutes)", "seconds": "Time (seconds)", "frames": "Frame"}[x_unit]
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of agents (comments)")
    ax.set_title("Opinion population over time")
    ax.set_ylim(bottom=0)
    ax.legend(title="Political view", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig, ax

def read_post_counts(post):
    pdf = pl.read_csv(post, separator="\t")
    counts = {c: 0 for c in range(5)}
    for v in pdf["political_position"].drop_nulls().to_list():
        counts[BELIEFS[v.strip().lower()]] += 1
    return counts

def run_stats(post_csv_paths, p_adjust="bonferroni"):
    groups, long_rows = [], []
    for path in post_csv_paths:
        vals = pl.read_csv(path)["num_clusters"].to_list()
        groups.append(vals)
        long_rows += [{"post": post, "num_clusters": v} for v in vals]
    H, p = ss.kruskal(*groups)
    print(f"Kruskal-Wallis: H = {H:.4f}, p = {p:.4g}")
    dunn = sp.posthoc_dunn(pd.DataFrame(long_rows),
                           val_col="num_clusters", group_col="post", p_adjust=p_adjust)
    print(f"Dunn's test (p-values, {p_adjust}-adjusted):")
    print(dunn)
    return H, p, dunn

cluster_results = []

post_csv_paths = []

for post in POSTS:
    counts = read_post_counts(post)

    cluster_results = []

    for seed in SEEDS:

        df = (
            # Step 1: Create a new simulation.
            HeadlessSimulation(OpinionConfig(
                image_rotation=True, 
                movement_speed=1, 
                radius=50,
                fps_limit=60,
                duration=10*60*60,
                seed=seed
                ))
            # Step 2: Add 50 agents to the simulation.
            .batch_spawn_agents(counts[0], OpinionAgent, images=IMAGES, belief="left")
            .batch_spawn_agents(counts[1], OpinionAgent, images=IMAGES, belief="center-left")
            .batch_spawn_agents(counts[2], OpinionAgent, images=IMAGES, belief="middle")
            .batch_spawn_agents(counts[3], OpinionAgent, images=IMAGES, belief="center-right")
            .batch_spawn_agents(counts[4], OpinionAgent, images=IMAGES, belief="right")

            # Step 3: Profit! 🎉
            .run()
            .snapshots
        )

        last_frame = df["frame"].max()

        # Get the opinion cluster counts at the end of the simulation
        num_opinion_clusters = len(df.filter(pl.col("frame") == last_frame).select("image_index").unique())
        cluster_results.append({"seed": seed, "num_clusters": num_opinion_clusters})

        # Get the number of agents for  counts at the end of the simulation
        counts_long = df.group_by(["frame", "image_index"]).len().rename({"len": "count"}).sort(["frame", "image_index"])

        counts_long.write_parquet(f"counts_{post}_seed_{seed}.parquet")

        print(num_opinion_clusters)

        plot_belief_counts(counts_long, save_path=f"belief_trajectories_{post}_seed_{seed}.png")

        # save results
        df.write_parquet(f"results_{post}_seed_{seed}.parquet")

    csv_path = f"cluster_counts_{post}.csv"
    pl.DataFrame(cluster_results).write_csv(csv_path)
    post_csv_paths.append(csv_path)

run_stats(post_csv_paths)