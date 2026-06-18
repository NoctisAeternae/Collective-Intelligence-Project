from dataclasses import dataclass

from vi import Agent, Simulation, HeadlessSimulation
from vi.config import Config
import random
import polars as pl
import matplotlib.pyplot as plt
import scipy.stats as ss
import scikit_posthocs as sp
import pandas as pd
import time

@dataclass
class OpinionConfig(Config): ...

BELIEFS = {"left": 0, "center-left": 1, "middle": 2, "center-right": 3, "right": 4}
BELIEF_TO_PROB = {0: 0.915, 1: 0.58, 2: 0.5, 3: 0.58, 4: 0.915}

IMAGES = ["images/left.png", "images/center-left.png", "images/middle.png", "images/center-right.png", "images/right.png"]

SEEDS = [i for i in range(1, 31)]

CODE_TO_NAME = {0: "left", 1: "center-left", 2: "middle", 3: "center-right", 4: "right"}
COLORS = {0: "#2166ac", 1: "#92c5de", 2: "#999999", 3: "#f4a582", 4: "#b2182b"}

POSTS = []

# TEMP VARS
cluster_results = []

agent_counts = [20, 20, 20, 20, 20]

post_name = "post2"

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


# for i in len(POSTS):
#     post = POSTS[i]
#     post_name = f"post{i}"

#     cluster_results = []

#     agent_counts = read_tsv(post)

for seed in range(SEEDS):

    random.seed(seed)
    
    start = time.time()

    df = (
        # Step 1: Create a new simulation.
        HeadlessSimulation(OpinionConfig(
            image_rotation=False, 
            movement_speed=1, 
            radius=50,
            fps_limit=60,
            duration=10*60*60,
            seed=seed
            ))
        # Step 2: Add 100 agents to the simulation.
        .batch_spawn_agents(agent_counts[0], OpinionAgent, images=IMAGES, belief="left")
        .batch_spawn_agents(agent_counts[1], OpinionAgent, images=IMAGES, belief="center-left")
        .batch_spawn_agents(agent_counts[2], OpinionAgent, images=IMAGES, belief="middle")
        .batch_spawn_agents(agent_counts[3], OpinionAgent, images=IMAGES, belief="center-right")
        .batch_spawn_agents(agent_counts[4], OpinionAgent, images=IMAGES, belief="right")

        # Step 3: Profit! 🎉
        .run()
        .snapshots
    )

    last_frame = df["frame"].max()

    # Get the opinion cluster counts at the end of the simulation
    num_opinion_clusters = len(df.filter(pl.col("frame") == last_frame).select("image_index").unique())
    cluster_results.append({"seed": seed, "num_clusters": num_opinion_clusters})

    # Get the number of unique image_index for each frame of the simulation.
    clusters_per_frame = (
        df.group_by("frame")
          .agg(pl.col("image_index").n_unique().alias("num_unique_clusters"))
          .sort("frame")
    )
    
    pl.DataFrame(clusters_per_frame).write_csv(f"cluster_counts_per_frame_{post_name}_seed_{seed}.csv")

    # Number of agents holding each belief, per frame (for the trajectory plot)
    counts_long = (
        df.group_by(["frame", "image_index"])
          .len()
          .rename({"len": "count"})
          .sort(["frame", "image_index"])
    )

    plot_belief_counts(counts_long, save_path=f"belief_trajectories_{post_name}_seed_{seed}.png")
    
    print("seed:",seed, "\nclusters:",num_opinion_clusters,"\n"+"-"*10)
    end = time.time()
    print(end - start)

# Create CSV file with the final cluster amounts for current post -> 30 seeded runs
pl.DataFrame(cluster_results).write_csv(f"cluster_counts_{post_name}.csv")