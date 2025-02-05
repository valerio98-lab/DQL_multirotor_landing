import argparse

from omni.isaac.lab.app import AppLauncher

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

# Register the environment
import environment  # noqa: F401
from dql_multirotor_landing.pid import PIDController
from dql_multirotor_landing.environment.moving_platform import MovingPlatform


@dataclass
class Actions:
    thrust: float
    roll: float
    pitch: float
    yaw: float
    left_wheel: float
    right_wheel: float

    def to_tensor(self, device):
        return torch.tensor(
            [
                [
                    self.thrust,
                    self.roll,
                    self.pitch,
                    self.yaw,
                    self.left_wheel,
                    self.right_wheel,
                ]
            ],
            device=device,
        )


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
    # prit info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    observation, info = env.reset()
    # print(observation)
    # print(observation["observation"].agent_position.shape)
    pid_controller = PIDController(set_point=[5, 0])
    target_controller = MovingPlatform()

    # simulate environment
    iteration = 0
    last_setpoint = 5
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            # apply actions
            iteration += 1
            # if iteration % 50 == 0:
            #     if last_setpoint > 1:
            #         last_setpoint = last_setpoint - 0.1
            #         pid_controller.set_setpoint([last_setpoint, 0])
            #         iteration = 0
            height = observation["relative_position"][0, 2]
            yaw = observation["agent_angular_velocity"]
            print(yaw)
            if iteration % 300 == 0:
                thrust, yaw_i = pid_controller.output([abs(height), yaw])
                yaw_i = 0.005
            else:
                thrust, yaw_i = pid_controller.output([abs(height), yaw])
                yaw_i = yaw_i.item()

            _, v_mp, _, w_mp = target_controller.compute_wheel_velocity(dt=0.02)
            actions = Actions(
                # ths[iteration],
                thrust.item(),
                0.0,
                0.0,
                yaw_i,
                w_mp,
                w_mp,
            ).to_tensor(
                device=env.unwrapped.device
            )  # type: ignore

            observation, _reward, _terminated, _truncated, _info = env.step(actions)

            if _terminated:
                pid_controller.reset()

            print(f"height: {height}, thrust: {thrust.item()}, yaw: {yaw_i}, w_mp: {w_mp}")
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
