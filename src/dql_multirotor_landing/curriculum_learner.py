# region: IsaacLab prelude
import argparse

from omni.isaac.lab.app import AppLauncher

import dql_multirotor_landing.environment  # noqa: F401

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
# endregion

from abc import ABC
from dataclasses import dataclass
from typing import Optional, TypeAlias

import gymnasium as gym
import numpy as np
import torch
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

from dql_multirotor_landing import logger
from dql_multirotor_landing.environment import mdp
from dql_multirotor_landing.environment.moving_platform import MovingPlatform
from dql_multirotor_landing.pid import PIDController
from dql_multirotor_landing.utils import Action


# region: Utils
@dataclass
class Limits:
    """Definition of the limits as described in equation 38,39,40"""

    p: float
    """defines the maximum allowable deviation of the UAV from the target landing position at a given curriculum step. It get shrinked as training progresses to force the landing to be more precise"""
    v: float
    """Limit velocity, define the maximum allowable velocity and acceleration at each curriculum step."""
    a: float
    """acceleration limit define the maximum allowable velocity and acceleration"""
    sigma: float
    """State action scaling factor that shrinks (refine the state space) at each curriculum step."""
    t_0: float
    """Time needed fo the platform to each $x_{max}$ using $a_{mpmax}$"""
    p_max: float
    """Maximum initial position deviation. This represents the UAV s maximum allowed starting distance from the platform at the first curriculum step."""
    v_max: float
    """Maximum initial velocity. This represents the largest velocity allowed at the first curriculum step. It is used for normalization."""
    a_mpmax: float
    """Maximum acceleration of the moving platform"""

    def contraction(self, curriculum_step: int):
        t_i = (self.sigma**curriculum_step) * self.t_0
        self.p = (0.5 * self.a_mpmax * (t_i**2)) / self.p_max
        self.v = (self.a_mpmax * t_i) / self.v_max
        # Ensure that this is consistent, should not be necessary
        self.a = 1


@dataclass
class Goals:
    p: float
    """Goal position"""

    v: float
    """Goal velocity"""

    a: float
    """Goal acceleration"""
    beta_p: float
    """Position shrink factor"""
    beta_v: float
    """Velocity shrink factor"""
    beta_a: float
    """Acceleration shrink factor"""
    sigma_a: float
    """Acceleration contraction factor"""

    def contraction(self, limits: Limits):
        self.p = self.beta_p * limits.p
        self.v = self.beta_v * limits.v
        self.a = self.beta_a * self.sigma_a


class Idx(ABC):
    position_idx: int
    velocity_idx: int
    acceleration_idx: int
    pitch_idx: int
    action_idx: Optional[int] = None

    def __init__(self, discrete_state: mdp.DiscreteState) -> None:
        self.position_idx = discrete_state.position
        self.velocity_idx = discrete_state.velocity
        self.acceleration_idx = discrete_state.acceleration
        self.pitch_idx = discrete_state.pitch


class QTableIdx(Idx):
    def __init__(self, discrete_state: mdp.DiscreteState) -> None:
        super().__init__(discrete_state)

    def set_action_idx(self, action_idx: int):
        self.action_idx = action_idx


StateCountPairIdx: TypeAlias = QTableIdx


class Table(ABC):
    data: np.ndarray

    def __init__(
        self, num_positions: int, num_velocities: int, num_accelerations: int, num_pitches: int, num_actions: int
    ) -> None:
        self.data = np.zeros((num_positions, num_velocities, num_accelerations, num_pitches, num_actions))

    def __setitem__(self, index: QTableIdx, key):
        if index.action_idx is not None:
            self.data[index.position_idx, index.velocity_idx, index.acceleration_idx, index.pitch_idx] = key
        else:
            self.data[
                index.position_idx, index.velocity_idx, index.acceleration_idx, index.pitch_idx, index.action_idx
            ] = key

    def __getitem__(self, index: QTableIdx):
        result = self.data[index.position_idx, index.velocity_idx, index.acceleration_idx, index.pitch_idx]
        if index.action_idx is not None:
            result = result[index.action_idx]
        return result

    def __repr__(self) -> str:
        return self.data.__repr__()


class QTable(Table):
    def __init__(
        self, num_positions: int, num_velocities: int, num_accelerations: int, num_pitches: int, num_actions: int
    ) -> None:
        super().__init__(num_positions, num_velocities, num_accelerations, num_pitches, num_actions)

    def transfer_learning(self, previous_Qtable: "QTable", current_r_max: float, previous_r_max: float) -> None:
        self.data = (current_r_max / previous_r_max) * previous_Qtable.data


class StateCountPairTable(Table):
    def __init__(
        self, num_positions: int, num_velocities: int, num_accelerations: int, num_pitches: int, num_actions: int
    ) -> None:
        super().__init__(num_positions, num_velocities, num_accelerations, num_pitches, num_actions)

    def update(self, index: StateCountPairIdx) -> None:
        assert index.action_idx is not None, "Cannot update the state count pair table without an action"
        self[index] += 1


@dataclass
class CurriculumLearnerParameters:
    omega: float = 0.51
    alpha_min: float = 0.02949
    total_curiculum_steps: int = 10
    total_episodes: int = 10
    max_timesteps_per_episode: int = 1000
    num_positions: int = 3
    num_velocities: int = 3
    num_accelerations: int = 3
    num_pitches: int = 7
    num_actions: int = 3
    sigma: float = 0.1
    initial_exploration_rate: float = 1.0
    gamma: float = 0.99
    x_max: float = 4.5
    p_max: float = 4.5
    v_max: float = 3.39
    a_mpmax: float = 1.28
    beta_p: float = 1 / 3
    beta_v: float = 1 / 3
    beta_a: float = 1 / 3
    sigma_a: float = 0.416


# endregion


class CurriculumLearner:
    def __init__(self, parameters: CurriculumLearnerParameters = CurriculumLearnerParameters()) -> None:
        env_cfg = parse_env_cfg(
            task_name="Isaac-Quadrotor-Landing-V0",
            device=args_cli.device,
            num_envs=args_cli.num_envs,
        )
        self.env = gym.make(id="Isaac-Quadrotor-Landing-V0", cfg=env_cfg)
        """Simulation environment"""
        self.gamma = parameters.gamma
        """Discount factor used for the Q table update"""

        self.omega = parameters.omega
        """Decay factor for the leaning rate"""

        self.alpha_min = parameters.alpha_min
        """Minimal leaning ability threshold. Once the leaning rate arrives here it stops decreasing."""

        self.total_curiculum_steps = parameters.total_curiculum_steps
        """Total number of curriculum steps"""

        self.num_actions = parameters.num_actions
        """Numbe of possible discrete actions"""

        self.num_positions = parameters.num_positions
        """Number of possible discrete positions"""

        self.num_velocities = parameters.num_velocities
        """Number of possible discrete velocities"""

        self.num_accelerations = parameters.num_accelerations
        """Number of possible discrete accelerations"""

        self.num_pitches = parameters.num_pitches
        """Number of possible discrete pitch angles"""

        # Double Q learning assumes we use two different Q tables
        self.Q_tables1 = [
            QTable(
                parameters.num_positions,
                parameters.num_velocities,
                parameters.num_accelerations,
                parameters.num_pitches,
                parameters.num_actions,
            )
            for _ in range(self.total_curiculum_steps)
        ]
        self.Q_tables2 = [
            QTable(
                parameters.num_positions,
                parameters.num_velocities,
                parameters.num_accelerations,
                parameters.num_pitches,
                parameters.num_actions,
            )
            for _ in range(self.total_curiculum_steps)
        ]

        self.state_action_pair_count = StateCountPairTable(
            parameters.num_positions,
            parameters.num_velocities,
            parameters.num_accelerations,
            parameters.num_pitches,
            parameters.num_actions,
        )
        """Count of how many times a state action pair has been visited"""

        self.limits = Limits(
            0,
            0,
            1,
            parameters.sigma,
            np.sqrt(2 * parameters.x_max / parameters.a_mpmax),
            parameters.p_max,
            parameters.v_max,
            parameters.a_mpmax,
        )
        """Limits define, at each curriculum step, the maximum allowable normalized actions.  Initially they are very coarse and get refined as the curriculum sequence advances."""

        self.goals = Goals(
            0,
            0,
            1,
            parameters.beta_p,
            parameters.beta_v,
            parameters.beta_a,
            parameters.sigma_a,
        )
        """Goals define, at each curriculum step, the current goals. Initially they are very coarse and get refined as the curriculum sequence advances."""

        self.total_episodes = parameters.total_episodes
        """Total number of episodes for each curiculum step"""

        self.max_timesteps_per_episode = parameters.max_timesteps_per_episode
        """Total number of episodes for each episode"""

        self.exploration_rate = parameters.initial_exploration_rate
        """Exploration rate that exponentially decays."""

        self.pid_controller = PIDController(set_point=[5, 0])
        self.moving_platform = MovingPlatform()

    def multi_resolution_train(self):
        previous_r_max: Optional[float] = None
        for curriculum_step in range(self.total_curiculum_steps):
            # Calculate contraction for limits using: Eq. 38-40
            self.limits.contraction(curriculum_step)
            logger.debug(f"{self.limits=}")

            # Calculate contraction for goals using: Eq. 41-43
            self.goals.contraction(self.limits)
            logger.debug(f"{self.goals=}")

            # Initialize the state space with the limits and the goals
            state_space = mdp.StateSpace(
                self.goals.p,
                self.limits.p,
                self.goals.v,
                self.limits.v,
                self.goals.a,
                self.limits.a,
            )

            if curriculum_step > 0:
                assert previous_r_max is not None
                # Get the curent known maximum reward using Eq: 28
                current_r_max = state_space.get_max_reward()
                # Apply transfer learning from the previous curriculum step using Eq: 31
                self.Q_tables1[curriculum_step].transfer_learning(
                    self.Q_tables1[curriculum_step - 1],
                    current_r_max,
                    previous_r_max,
                )
                self.Q_tables2[curriculum_step].transfer_learning(
                    self.Q_tables2[curriculum_step - 1],
                    current_r_max,
                    previous_r_max,
                )

            self._double_q_learning(
                self.Q_tables1[curriculum_step],
                self.Q_tables2[curriculum_step],
                state_space,
            )

            logger.debug(f"{self.Q_tables1[curriculum_step]=}")
            logger.debug(f"{self.Q_tables2[curriculum_step]=}")
            self._decay_exploration_rate()
            previous_r_max = state_space.get_max_reward()

    def _learning_rate(self, index: StateCountPairIdx) -> float:
        """Returns the curent learning rate using Eq. 30"""
        n_c = self.state_action_pair_count[index]
        return max(((n_c + 1) ** (-self.omega)), self.alpha_min)

    def _decay_exploration_rate(
        self,
    ):
        self.exploration_rate *= 0.995
        self.exploration_rate = max(self.exploration_rate, 0.01)

    def _update_q_table(
        self,
        decision_policy: QTable,
        behaviour_policy: QTable,
        current_idx: QTableIdx,
        next_idx: QTableIdx,
        reward: float,
    ):
        best_action = np.argmax(
            decision_policy.data[
                next_idx.position_idx,
                next_idx.velocity_idx,
                next_idx.acceleration_idx,
                next_idx.pitch_idx,
            ]
        )
        next_idx.set_action_idx(int(best_action))

        decision_policy[current_idx] += self._learning_rate(current_idx) * (
            reward + self.gamma * behaviour_policy[next_idx] - decision_policy[current_idx]
        )

    def _double_q_learning(self, Q_table1: QTable, Q_table2: QTable, state_space: mdp.StateSpace):
        for _episode in range(1, self.total_episodes):
            observation, _info = self.env.reset()
            current_continuous_state = mdp.ContinuousState(
                torch.abs(observation["relative_position"][0]),
                torch.abs(observation["relative_velocity"][0]),
                torch.abs(observation["relative_acceleration"][0]),
                torch.abs(observation["relative_orientation"][0, 0]).item(),
            )
            # Discretize the current state

            current_discrete_state = state_space.get_discretized_state(current_continuous_state)
            current_idx = QTableIdx(current_discrete_state)

            for _timestep in range(1, self.max_timesteps_per_episode):
                # Explore
                if np.random.rand() < self.exploration_rate:
                    action = state_space.sample()
                # Commit
                else:
                    # Get the action based on both Q table1 and Q table2
                    action = np.argmax((Q_table1[current_idx] + Q_table2[current_idx]) / 2)
                current_idx.set_action_idx(int(action))

                # Update the state action pair count
                self.state_action_pair_count.update(current_idx)
                thrust, yaw = self.pid_controller.output(
                    [
                        torch.abs(observation["relative_position"][0, 2]),
                        observation["relative_acceleration"][0, 2],
                    ]
                )
                platform_v = self.moving_platform.compute_wheel_velocity(0.02)[1]  # noqa: F841
                # Do nothing
                pitch = observation["relative_orientation"][0, 0]
                # Decrease
                if action == 1:
                    pitch = observation["relative_orientation"][0, 0] - state_space.delta_angle
                # Increase
                elif action == 2:
                    pitch = observation["relative_orientation"][0, 0] + state_space.delta_angle

                action = Action(
                    thrust.item(),
                    # TODO: What do we do ?
                    0.0,
                    float(pitch),
                    yaw.item(),
                    # platform_v.item(),
                    # platform_v.item(),
                    0,
                    0,
                )

                observation, _reward, terminated, truncated, _info = self.env.step(
                    action.to_tensor(self.env.unwrapped.device)  # type: ignore
                )
                next_continuous_state = mdp.ContinuousState(
                    torch.abs(observation["relative_position"][0]),
                    torch.abs(observation["relative_velocity"][0]),
                    torch.abs(observation["relative_acceleration"][0]),
                    torch.abs(observation["relative_orientation"][0, 0]).item(),
                )
                state_space._set_last_state(current_continuous_state)
                next_discrete_state = state_space.get_discretized_state(next_continuous_state)
                next_idx = QTableIdx(next_discrete_state)

                reward = state_space.get_reward(next_continuous_state)

                # Update either Q table1 or Q table2
                if np.random.rand() < 0.5:
                    self._update_q_table(
                        Q_table1,
                        Q_table2,
                        current_idx,
                        next_idx,
                        reward,
                    )

                else:
                    self._update_q_table(
                        Q_table2,
                        Q_table1,
                        current_idx,
                        next_idx,
                        reward,
                    )

                if terminated or truncated:
                    break
                current_idx = next_idx
                current_continuous_state = next_continuous_state


CurriculumLearner().multi_resolution_train()
