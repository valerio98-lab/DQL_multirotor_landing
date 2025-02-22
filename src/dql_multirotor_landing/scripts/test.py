from typing import Any, Dict

import gym
import rospy

from dql_multirotor_landing.double_q_learning import DoubleQLearningAgent
from dql_multirotor_landing.landing_simulation_env import SimulationLandingEnv

if __name__ == "__main__":
    rospy.init_node("landing_simulation_gym_node")
    agent_x = DoubleQLearningAgent.load()
    agent_y = DoubleQLearningAgent.load()

    env: SimulationLandingEnv = gym.make(
        "Landing-Simulation-v0",
        # Zero indexed, curriculum step 5
        initial_curriculum_step=4,
    )  # type:ignore

    def log(info: Dict[str, Any], clean=True):
        if clean:
            print("\x1b[0;0f", end="")
            print("\x1b[J", end="")
        else:
            print("=" * 80)
        info["Termination condition"] = info["Termination condition"].replace(
            "SUCCESS", "\x1b[1;32mSUCCESS\x1b[0m"
        )
        info["Termination condition"] = info["Termination condition"].replace(
            "FAILURE", "\x1b[1;31mFAILURE\x1b[0m"
        )
        for k, v in info.items():
            print(f"{k}: {v}")
        if clean:
            # Prepare for next print
            print("\x1b[0;0f", end="")
            print("\x1b[J", end="")
            print("\x1b[0m", end="")
        else:
            print("=" * 80)

    for current_episode in range(30):
        current_state_x, current_state_y = env.reset()

        done = False
        while not done:
            action_x = agent_x.predict(current_state_x)
            action_y = agent_y.predict(current_state_y)

            next_state_x, next_state_y, done, info = env.step(action_x, action_y)

            if done:
                info["current_episode"] = current_episode
                print("=" * 80)
                for k, v in info.items():
                    print(f"{k}: {v}")
                print("=" * 80)
                env.pause_sim()
                break
            current_state_x = next_state_x
            current_state_y = next_state_y
    rospy.signal_shutdown("Training ended sucessfully")
