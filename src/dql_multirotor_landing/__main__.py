import argparse

from omni.isaac.lab.app import AppLauncher

from dql_multirotor_landing.utils import Actions

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--task", type=str, default="Isaac-Quadrotor-Landing-V0", help="Task to run.")
# apend AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = False
# launch omniverse app

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import omni.isaac.lab_tasks  # isort: skip  # noqa: F401
from dataclasses import dataclass

from omni.isaac.lab_tasks.utils import parse_env_cfg

from dql_multirotor_landing.environment.moving_platform import MovingPlatform
from dql_multirotor_landing.pid import PIDController


def main():
    """Random actions agent with Isaac Lab environment."""

    # create environment configuration
    env_cfg = parse_env_cfg(
        task_name="Isaac-Quadrotor-Landing-V0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    # create environment"Isaac-Quadrotor-Landing-V0"
    env = gym.make(id="Isaac-Quadrotor-Landing-V0", cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    observation, info = env.reset()

    print(f"[INFO]: Observation: {observation}")
    height_set_point = observation["target_position"][0, 2]

    # Controllers
    pid_controller = PIDController(set_point=[height_set_point, 0])
    target_controller = MovingPlatform()

    # simulate environment
    iteration = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            height = observation["relative_position"][0, 2]
            yaw = observation["relative_acceleration"][0, 2]

            _, v_mp, _, w_mp = target_controller.compute_wheel_velocity(dt=0.02)
            if -height < 0:
                thrust = torch.tensor(0.0)
            else:
                thrust, yaw = pid_controller.output([-height, yaw])
            actions = Actions(
                thrust.item(),
                0.0,
                0.0,
                0.0,
                v_mp,
                v_mp,
            ).to_tensor(device=env.unwrapped.device)

            observation, _reward, _terminated, _truncated, _info = env.step(actions)
            print(observation)
            exit()
            # print("Height: ", height)

            if _terminated:
                pid_controller.reset()

            # height = observation["observation"].agent_position[0, 2]
            # yaw = observation["observation"].agent_angular_velocity[0, 2]
            # thrust, yaw = pid_controller.output([height, yaw])
            # print(f"height: {height}, thrust: {thrust.item()}, yaw: {yaw.item()}")
            # print("v_mp: ", height)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
