from dataclasses import dataclass

from vi import Agent, Simulation, HeadlessSimulation
from vi.config import Config


@dataclass
class FlockingConfig(Config): ...


class FlockingAgent(Agent[FlockingConfig]): ...

import time

start = time.time()

(
    # Step 1: Create a new simulation.
    HeadlessSimulation(FlockingConfig(image_rotation=True, movement_speed=1, radius=50, duration=60*60*5, fps_limit = 0))
    # Step 2: Add 50 agents to the simulation.
    .batch_spawn_agents(50, FlockingAgent, images=["images/triangle.png"], belief="left")
    # Step 3: Profit! 🎉
    .run()
)

end = time.time()
print(end - start)