import torch
from dataclasses import dataclass
from typing import Optional
from dql_multirotor_landing.parameters import Parameters


INCREASING = 1
DECREASING = -1
NOTHING = 0


class ActionSpace:
    """
    Represents the action space for the reinforcement learning agent,
    where actions correspond to angle adjustments.

    Attributes:
        action_space_dim (int): The number of discrete actions available.
        device (str): The computing device ('cpu' or 'cuda').
        angles_set (torch.Tensor): The set of possible angle values.
        delta_angle (float): The increment/decrement step for angle adjustments.
        parameters (Parameters): The set of global parameters.
    """

    def __init__(
        self,
        action_space_dim: int = 3,
        device="cpu",
    ):
        self.action_space_dim = action_space_dim
        self.device = device
        self.parameters = Parameters()
        self.delta_angle = self.parameters.delta_angle
        self.angles_set = self._discretize_action_space()

    def _discretize_action_space(self):
        """
        Discretizes the angle space based on the max angle and the number of the hyperparameter n_theta.

        Returns:
            torch.Tensor: A tensor containing the set of discrete angles.
        """

        angle_values = torch.tensor(
            [-self.parameters.angle_max + i * self.delta_angle for i in range(2 * self.action_space_dim + 1)],
            device=self.device,
        )
        return angle_values

    def get_discrete_action(self, current_angle: torch.Tensor, idx: int):
        """
        Updates the current angle based on the selected discrete action.

        Args:
            current_angle (torch.Tensor): The current angle of the drone.
            idx (int): The action index (INCREASING, DECREASING, or NOTHING).

        Returns:
            int: The index of the closest discrete action in `angles_set`.
        """
        if idx == INCREASING:
            current_angle += self.delta_angle
        elif idx == DECREASING:
            current_angle -= self.delta_angle
        elif idx == NOTHING:
            current_angle = current_angle

        current_angle = torch.clamp(current_angle, -self.parameters.angle_max, self.parameters.angle_max)

        return torch.abs(self.angles_set - current_angle).argmin().item()


@dataclass
class DiscreteState:
    position: int
    velocity: int
    acceleration: int
    action_index: int


@dataclass
class ContinuousState:
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    pitch_angle: float


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
            print(not isinstance(self.last_state, None))
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

    def d_f(self, continuos_state: torch.tensor, x1: float, x2: float):
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

    def get_discretized_state(self, state: ContinuousState, discrete_action: int = None):
        """
        Converts a continuous state into a discretized state representation.

        Args:
            state ContinuousState(): an observation from the environment filled up with a relative position,
            a relative velocity and a relative acc.

        Returns:
            DiscreteState: The discretized state.
        """

        relative_pos = state.position
        relative_vel = state.velocity
        relative_acc = state.acceleration
        theta_index = None

        normalized_position = self.parameters._normalized_state(relative_pos, self.p_max)
        normalized_velocity = self.parameters._normalized_state(relative_vel, self.v_max)
        if relative_acc is not None:
            normalized_acc = self.parameters._normalized_state(relative_acc, self.a_max)
        if discrete_action is not None:
            theta_index = discrete_action
        return DiscreteState(
            position=self.d_f(normalized_position, self.goal_pos, self.p_lim),
            velocity=self.d_f(normalized_velocity, self.goal_vel, self.v_lim),
            acceleration=self.d_f(normalized_acc, self.goal_acc, self.a_lim),
            action_index=theta_index,
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
        current_relative_pos = current_continuous_state.position
        current_relative_vel = current_continuous_state.velocity
        last_relative_pos = self.last_state.position
        last_relative_vel = self.last_state.velocity

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
        relative_theta_reduction = abs(current_continuous_state.pitch_angle) - abs(self.last_state.pitch_angle)
        r_theta = (self.w_theta * relative_theta_reduction) / self.parameters.angle_max / self.v_lim
        r_dur = self.parameters.w_dur * self.v_lim * self.delta_t

        return (r_p + r_v + r_theta + r_dur + r_term).item()


# if __name__ == "__main__":
#     # Creiamo un esempio di spazio di stato
#     state_space = StateSpace(
#         p_max=torch.tensor(0.8),
#         v_max=torch.tensor(0.5),
#         a_max=torch.tensor(0.5),
#         goal_pos=torch.randint(0, 10, (3,)).float(),
#         lim_pos=torch.randint(0, 10, (3,)).float(),
#         goal_vel=torch.randint(0, 10, (3,)).float(),
#         lim_vel=torch.randint(0, 10, (3,)).float(),
#         goal_acc=torch.randint(0, 10, (3,)).float(),
#         lim_acc=torch.randint(0, 10, (3,)).float(),
#         device="cpu",
#     )

#     # Stato attuale del drone
#     while True:
#         relative_pos = (torch.randn(3),)
#         relative_vel = (torch.randn(3),)
#         relative_acc = (torch.randn(3),)
#         theta_index = 2  # Supponiamo un indice qualsiasi dell'angolo

#         # Otteniamo lo stato discretizzato
#         discrete_state = state_space.get_discretized_state(relative_pos, relative_vel, relative_acc, theta_index)
#         if discrete_state.position != 1 or discrete_state.velocity != 1 or discrete_state.acceleration != 1:
#             print("Stato Discretizzato:", discrete_state)
#         # print("Stato Discretizzato:", discrete_state)
