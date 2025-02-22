from typing import Any, Dict

import gym

from dql_multirotor_landing.double_q_learning import DoubleQLearningAgent
from dql_multirotor_landing.landing_simulation_env import SimulationLandingEnv

agent_x = DoubleQLearningAgent.load()
agent_y = DoubleQLearningAgent.load()


env: SimulationLandingEnv = gym.make(
    "Landing-Simulation-v0",
    # Zero indexed, curriculum step 5
    initial_curriculum_step=4,
)  # type:ignore


def log(info: Dict[str, Any]):
    # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
    # Clear screen and return to the left corner
    print("\x1b[0;0f", end="")
    print("\x1b[J", end="")
    for k, v in info.items():
        print(f"{k}: {v}")
    print("\x1b[0;0f", end="")
    print("\x1b[J", end="")
    # Prepare for next print
    print("\x1b[0;0f", end="")
    print("\x1b[J", end="")
    print("\x1b[0m", end="")


for current_episode in range(30):
    current_state_x, current_state_y = env.reset()

    done = False
    while not done:
        action_x = agent_x.predict(current_state_x)
        action_y = agent_y.predict(current_state_y)

        next_state_x, next_state_y, done, info = env.step(action_x, action_y)

        if done:
            info["current_episode"] = current_episode
            info["remaining_episodes"] = 400 - current_episode
            for k, v in info.items():
                print(f"{k}: {v}")
            break
        current_state_x = next_state_x
        current_state_y = next_state_y
