from dataclasses import dataclass
from typing import Optional

import torch

from dql_multirotor_landing.parameters import Parameters

INCREASING = 2
DECREASING = 1
NOTHING = 0


@dataclass
class ContinuousState:
    relative_position: torch.Tensor
    relative_velocity: torch.Tensor
    relative_acceleration: torch.Tensor
    pitch_angle: float

    def __repr__(self) -> str:
        return f"{self.relative_position}, {self.relative_velocity}, {self.relative_acceleration}, {self.pitch_angle}"


@dataclass
class DiscreteState:
    position: int
    velocity: int
    acceleration: int
    pitch: int


class StateSpace:
    """
    Represents the discretized state space for reinforcement learning.

    Attributes:
        parameters (Parameters): The set of global parameters.
        p_max, v_max, a_max (float): Maximum values for position, velocity, and acceleration.
        norm_goal_pos, norm_lim_pos, norm_goal_vel, norm_lim_vel, norm_goal_acc, norm_lim_acc (torch.Tensor):
            Normalized limits and goal values for position, velocity, and acceleration.
        r_success, r_failure (float): Reward values for success and failure.
        discretized_goal_state (torch.Tensor): The target state that defines success.
    """

    def __init__(
        self,
        goal_pos: float,
        p_lim: float,
        goal_vel: float,
        v_lim: float,
        goal_acc: float,
        a_lim: float,
        device="cpu",
    ):
        self.parameters = Parameters()
        self.device = device
        self.p_max = self.parameters.p_max
        self.v_max = self.parameters.v_max
        self.a_max = self.parameters.a_max
        self.w_p = self.parameters.w_p
        self.w_v = self.parameters.w_v
        self.w_theta = self.parameters.w_theta
        self.w_dur = self.parameters.w_dur
        self.delta_t = 1 / self.parameters.fag
        self.delta_angle = self.parameters.delta_angle
        self.last_state: Optional[ContinuousState] = None

        if not isinstance(self.last_state, ContinuousState) and self.last_state is not None:
            print(not isinstance(self.last_state, ContinuousState))
            print(self.last_state is not None)
            raise TypeError(
                "last_state should be a ContinuousState (@dataclass) or None if no state is available at the current time step."
            )

        self.goal_pos = goal_pos
        self.p_lim = p_lim
        self.goal_vel = goal_vel
        self.v_lim = v_lim
        self.goal_acc = goal_acc
        self.a_lim = a_lim

        self.r_term = None
        self.r_success = self.parameters.r_success
        self.r_failure = self.parameters.r_failure
        self.discretized_goal_state = torch.tensor([1, 1], device=self.device)

    def sample(self):
        """
        Samples a random action from the action space using a random choice.

        Returns:
            int: The index of the sampled action.
        """

        ## Randomly select an action between [0, 1, 2]. 0: NOTHING, 1: DECREASING, 2: INCREASING
        action = torch.randint(0, 3, (1,)).item()

        return action

    def _get_discrete_angle(self, current_angle: float, idx: int):
        """
        Updates the current angle based on the selected discrete action idx.

        Args:
            current_angle (float): The current angle of the drone.
            idx (int): The action index (INCREASING, DECREASING, or NOTHING).

        Returns:
            tuple: (new_angle_idx (int), new_angle (float))

            new_angle_idx (int): The index of the closest discrete action in `angle_values`,
            which represents the set of 7 possible pitch positions the drone can assume.

            new_angle (float): The updated pitch angle of the drone as a float.

        """

        if idx == INCREASING:
            current_angle += self.parameters.delta_angle
        elif idx == DECREASING:
            current_angle -= self.parameters.delta_angle
        elif idx == NOTHING:
            current_angle = current_angle

        new_angle = torch.clamp(current_angle, -self.parameters.angle_max, self.parameters.angle_max)
        new_angle_idx = torch.abs(self.parameters.angle_values - new_angle).argmin().item()

        return new_angle_idx, new_angle

    def _set_last_state(self, last_state: ContinuousState):
        """
        Sets the last state (`last_state`) for the class instance.

        Args:
            last_state (ContinuousState): The new state to assign. Must be an instance of ContinuousState.

        Raises:
            TypeError: If `last_state` is not an instance of ContinuousState.

        """

        if not isinstance(last_state, ContinuousState):
            raise TypeError("last_state should be a ContinuousState (@dataclass)")
        self.last_state = last_state

    def d_f(self, continuos_state: torch.Tensor, x1: float, x2: float):
        """
        Discretizes a continuous state value into one of three categories:
        - 0: Far from the goal state
        - 1: Close to the goal state
        - 2: Middle distance from the goal state

        Args:
            continuous_state (torch.Tensor): The continuous state value.
            x1 float: The goal state boundary.
            x2 float: The maximum state boundary.

        Returns:
            int: The discretized state index.
        """

        discretized_state_x = continuos_state[0]

        if x2 < x1:
            raise ValueError(f"max_value {x2} should be greater than or equal to min_value {x1}")

        if discretized_state_x > x2:
            raise ValueError(
                f"The x_dimension of the state {discretized_state_x} should be less than or equal to max_value {x2}"
            )

        if discretized_state_x < x1:
            raise ValueError(
                f"The x_dimension of the state {discretized_state_x} should be greater than or equal to min_value {x1}"
            )

        if discretized_state_x >= -x2 and discretized_state_x < -x1:
            state = 0  ##Far distance wrt the goal state

        if discretized_state_x >= -x1 and discretized_state_x <= x1:
            state = 1  ##Close distance wrt the goal state

        if discretized_state_x > x1 and discretized_state_x <= x2:
            state = 2  ##Middle distance wrt the goal state

        return state

    def get_discretized_state(self, state: ContinuousState):
        """
        Converts a continuous state into a discretized state representation.

        Args:
            state ContinuousState(): an observation from the environment filled up with a relative position,
            a relative velocity and a relative acc.

        Returns:
            DiscreteState: The discretized state.
        """

        relative_pos = state.relative_position
        relative_vel = state.relative_velocity
        relative_acc = state.relative_acceleration
        pitch = state.pitch_angle

        normalized_position = self.parameters._normalized_state(relative_pos, self.p_max)
        normalized_velocity = self.parameters._normalized_state(relative_vel, self.v_max)

        if relative_acc is not None:
            normalized_acc = self.parameters._normalized_state(relative_acc, self.a_max)

        return DiscreteState(
            position=self.d_f(normalized_position, self.goal_pos, self.p_lim),
            velocity=self.d_f(normalized_velocity, self.goal_vel, self.v_lim),
            acceleration=self.d_f(normalized_acc, self.goal_acc, self.a_lim),
            pitch=torch.abs(self.parameters.angle_values - pitch).argmin().item(),
        )

    def get_max_reward(self):
        """
        Returns the maximum possible reward.

        Returns:
            float: The maximum reward.
        """
        r_p_max = abs(self.parameters.w_p) * self.v_lim * self.delta_t
        r_v_max = abs(self.parameters.w_v) * self.a_lim * self.delta_t
        r_theta_max = abs(self.parameters.w_theta) * self.v_lim * (self.delta_angle / self.parameters.angle_max)
        r_dur_max = self.parameters.w_dur * self.v_lim * self.delta_t

        return r_p_max + r_v_max + r_theta_max + r_dur_max

    def get_reward(
        self,
        current_continuous_state: ContinuousState,
    ):
        """
        Computes the reward for a given state transition leveraging the last state transition performed in the environment
        For details look for section 3.3.6 of the paper.

        Args:
            current_continuous_state (ContinuousState): The current observation from the environment.

        Returns:
            float: The computed reward.
        """
        current_relative_pos = current_continuous_state.relative_position
        current_relative_vel = current_continuous_state.relative_velocity
        last_relative_pos = self.last_state.relative_position  # type: ignore
        last_relative_vel = self.last_state.relative_velocity  # type:ignore

        current_relative_pos = self.parameters._normalized_state(current_relative_pos, self.parameters.p_max)
        last_relative_pos = self.parameters._normalized_state(last_relative_pos, self.parameters.p_max)
        discrete_pos = self.d_f(current_relative_pos, self.goal_pos, self.p_lim)

        current_relative_vel = self.parameters._normalized_state(current_relative_vel, self.parameters.v_max)
        last_relative_vel = self.parameters._normalized_state(last_relative_vel, self.parameters.v_max)
        discrete_vel = self.d_f(current_relative_vel, self.goal_vel, self.v_lim)

        actual_state = torch.tensor([discrete_pos, discrete_vel], device=self.device)

        # terminal term r_term
        if torch.equal(actual_state, self.discretized_goal_state):
            r_term = self.r_success
        elif torch.abs(current_relative_pos[0]) > self.p_lim:
            r_term = self.r_failure
        else:
            r_term = 0

        # position term r_p
        relative_pos_reduction = torch.abs(current_relative_pos[0]) - torch.abs(last_relative_pos[0])
        r_p = torch.clip(self.w_p * relative_pos_reduction, -self.parameters.r_p_max, self.parameters.r_p_max)

        # velocity term r_v
        relative_vel_reduction = torch.abs(current_relative_vel[0]) - torch.abs(last_relative_vel[0])
        r_v = torch.clip(self.w_v * relative_vel_reduction, -self.parameters.r_v_max, self.parameters.r_v_max)

        # orientation term r_theta and duration term r_dur
        relative_theta_reduction = abs(current_continuous_state.pitch_angle) - abs(
            self.last_state.pitch_angle  # type: ignore
        )
        r_theta = (self.w_theta * relative_theta_reduction) / self.parameters.angle_max / self.v_lim
        r_dur = self.parameters.w_dur * self.v_lim * self.delta_t

        return (r_p + r_v + r_theta + r_dur + r_term).item()
