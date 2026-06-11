from dataclasses import dataclass

from vi import Agent, Simulation
from vi.config import Config
import random
import polars as pl

@dataclass
class FlockingConfig(Config): ...

BELIEFS = {"left": 0, "center-left": 1, "middle": 2, "center-right": 3, "right": 4}

BELIEF_TO_PROB = {0: 0.9, 1: 0.75, 2: 0.5, 3: 0.75, 4: 0.9}

IMAGES = ["images/left.png", "images/center-left.png", "images/middle.png", "images/center-right.png", "images/right.png"]

random.seed(42)

class OpinionAgent(Agent[FlockingConfig]):

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



df = (
    # Step 1: Create a new simulation.
    Simulation(FlockingConfig(
        image_rotation=True, 
        movement_speed=1, 
        radius=50,
        fps_limit=60,
        duration=10*60,
        seed=42
        ))
    # Step 2: Add 50 agents to the simulation.

    .batch_spawn_agents(10, OpinionAgent, images=IMAGES, belief="left")
    .batch_spawn_agents(10, OpinionAgent, images=IMAGES, belief="center-left")
    .batch_spawn_agents(10, OpinionAgent, images=IMAGES, belief="middle")
    .batch_spawn_agents(10, OpinionAgent, images=IMAGES, belief="center-right")
    .batch_spawn_agents(10, OpinionAgent, images=IMAGES, belief="right")
    # Step 3: Profit! 🎉
    .run()
    .snapshots
)

num_opinion_clusters = len(df.filter(pl.col("frame") == 600).select("image_index").unique())

print(num_opinion_clusters)

# save results
df.write_parquet("results.parquet")