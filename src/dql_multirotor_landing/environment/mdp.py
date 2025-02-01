import torch
from dataclasses import dataclass
from dql_multirotor_landing.parameters import Parameters as prms


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
        self.angles_set = self._discretize_action_space()
        self.delta_angle = self.angle_max / self.action_space_dim
        self.parameters = prms()

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
    index: int


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
        device="cpu",
    ):
        self.parameters = prms()
        self.p_max = self.parameters.p_max
        self.v_max = self.parameters.v_max
        self.a_max = self.parameters.a_max
        self.w_p = self.parameters.w_p
        self.w_v = self.parameters.w_v
        self.w_theta = self.parameters.w_theta
        self.w_dur = self.parameters.w_dur
        self.device = device
        self.norm_goal_pos = self.parameters.norm_goal_pos
        self.norm_lim_pos = self.parameters.norm_lim_pos
        self.norm_goal_vel = self.parameters.norm_goal_vel
        self.norm_lim_vel = self.parameters.norm_lim_vel
        self.norm_goal_acc = self.parameters.norm_goal_acc
        self.norm_lim_acc = self.parameters.norm_lim_acc

        self.r_term = None
        self.r_success = self.parameters.r_success
        self.r_failure = self.parameters.r_failure
        self.discretized_goal_state = torch.tensor([1, 1], device=self.device)

    def d_f(self, continuos_state: torch.Tensor, x1, x2):
        """
        Discretizes a continuous state value into one of three categories:
        - 0: Far from the goal state
        - 1: Close to the goal state
        - 2: Middle distance from the goal state

        Args:
            continuous_state (torch.Tensor): The continuous state value.
            x1 (torch.Tensor): The goal state boundary.
            x2 (torch.Tensor): The maximum state boundary.

        Returns:
            int: The discretized state index.
        """
        print(continuos_state, x1, x2)
        if continuos_state >= -x2 and continuos_state < -x1:
            discretized_state = 0  ##Far distance wrt the goal state

        if continuos_state >= -x1 and continuos_state <= x1:
            discretized_state = 1  ##Close distance wrt the goal state

        if continuos_state > x1 and continuos_state <= x2:
            discretized_state = 2  ##Middle distance wrt the goal state

        return discretized_state

    # fmt: off
    def get_discretized_state(
        self, relative_pos: torch.Tensor, 
        relative_vel: torch.Tensor, 
        relative_acc: torch.Tensor, 
        angle_index: int
    ):  # fmt: on

        """
        Converts a continuous state into a discretized state representation.

        Args:
            relative_pos (torch.Tensor): Relative position.
            relative_vel (torch.Tensor): Relative velocity.
            relative_acc (torch.Tensor): Relative acceleration.
            angle_index (int): The action index corresponding to the new angle assumes by the drone.

        Returns:
            DiscreteState: The discretized state.
        """

        normalized_position = prms._normalized_state(relative_pos, self.p_max)
        normalized_velocity = prms._normalized_state(relative_vel, self.v_max)
        normalized_acc = prms._normalized_state(relative_acc, self.a_max)

        return DiscreteState(
            position=self.d_f(normalized_position[0], self.norm_goal_pos, self.norm_lim_pos),
            velocity=self.d_f(normalized_velocity[0], self.norm_goal_vel, self.norm_lim_vel),
            acceleration=self.d_f(normalized_acc[0], self.norm_goal_acc, self.norm_lim_acc),
            index=angle_index,
        )

    def get_reward(
        self,
        state: DiscreteState,
        current_relative_pos: torch.Tensor,
        last_relative_pos: torch.Tensor,
    ):
        """
        Computes the reward for a given state transition.

        Args:
            state (DiscreteState): The current discretized state.
            current_relative_pos (torch.Tensor): The current relative position.
            last_relative_pos (torch.Tensor): The previous relative position.

        Returns:
            float: The computed reward.
        """

        actual_state = torch.tensor([state.position, state.velocity], device=self.device)
        current_relative_pos = self.normalized_state(current_relative_pos, self.p_max)
        last_relative_pos = self.normalized_state(last_relative_pos, self.p_max)

        # terminal term r_term
        if torch.equal(actual_state, self.discretized_goal_state):
            r_term = self.r_success
        elif torch.abs(current_relative_pos[0]) > self.parameters.norm_lim_pos[0]:
            r_term = self.r_failure
        else:
            r_term = 0

        # position term r_p
        relative_pos_reduction = torch.abs(current_relative_pos[0]) - torch.abs(last_relative_pos[0])
        r_p = torch.clip(self.w_p * relative_pos_reduction, -self.parameters.r_p_max, self.parameters.r_p_max)

        # velocity term r_v
        relative_vel_reduction = torch.abs(current_relative_pos[1]) - torch.abs(last_relative_pos[1])
        r_v = torch.clip(self.w_v * relative_vel_reduction, -self.parameters.r_v_max, self.parameters.r_v_max)

        # orientation term r_theta and duration term r_dur
        r_theta = ...
        r_dur = self.parameters.w_dur * self.parameters.norm_lim_vel * (1 / self.parameters.fag)

        return r_p + r_v + r_theta + r_dur + r_term


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
