import torch
from dataclasses import dataclass

import enum


INCREASING = 1
DECREASING = -1
NOTHING = 0


class ActionSpace:
    def __init__(
        self,
        action_space_dim: int = 3,
        ka: float = 0.5,
        platform_max_acc: torch.Tensor = None,
        device="cpu",
        gravity: float = 9.81,
    ):
        self.action_space_dim = action_space_dim
        self.device = device
        self.ka = ka
        self.platform_max_acc = platform_max_acc
        self.gravity = gravity
        self.angle_max = self._get_angle_max()
        self.angles_set = self._discretize_action_space()
        self.delta_angle = self.angle_max / self.action_space_dim

    def _discretize_action_space(self):
        # self.angle_max = 10
        angle_values = torch.tensor(
            [-self.angle_max + i * self.delta_angle for i in range(2 * self.action_space_dim + 1)], device=self.device
        )
        return angle_values

    def _get_angle_max(self):
        return torch.atan((self.ka * torch.tensor(self.platform_max_acc[0])) / self.gravity, device=self.device)

    def get_discrete_action(self, current_angle: torch.Tensor, idx: int):
        if idx == INCREASING:
            current_angle += self.delta_angle
        elif idx == DECREASING:
            current_angle -= self.delta_angle
        elif idx == NOTHING:
            current_angle = current_angle

        current_angle = torch.clamp(current_angle, -self.angle_max, self.angle_max)

        return torch.abs(self.angles_set - current_angle).argmin().item()


@dataclass
class DiscreteState:
    position: int
    velocity: int
    acceleration: int
    index: int


class StateSpace:
    def __init__(
        self,
        p_max,
        v_max,
        a_max,
        goal_pos: torch.Tensor,
        lim_pos: torch.Tensor,
        goal_vel: torch.Tensor,
        lim_vel: torch.Tensor,
        goal_acc: torch.Tensor,
        lim_acc: torch.Tensor,
        r_success: float,
        r_failure: float,
        device="cpu",
    ):
        self.p_max = p_max
        self.v_max = v_max
        self.a_max = a_max
        self.device = device
        self.lim_pos = lim_pos
        self.normalized_gps = self.normalized_state(goal_pos, self.p_max)
        self.normalized_lps = self.normalized_state(lim_pos, self.p_max)
        self.normalized_gvl = self.normalized_state(goal_vel, self.v_max)
        self.normalized_lvl = self.normalized_state(lim_vel, self.v_max)
        self.normalized_gacc = self.normalized_state(goal_acc, self.a_max)
        self.normalized_lacc = self.normalized_state(lim_acc, self.a_max)

        self.r_term = None
        self.r_success = r_success
        self.r_failure = r_failure
        self.discretized_goal_state = torch.tensor([1, 1], device=self.device)

    def normalized_state(self, state: torch.Tensor, max_value):
        return torch.clip(state[0] / max_value, -1, 1).to(self.device)

    def d_f(self, continuos_state: torch.Tensor, x1, x2):
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

        normalized_position = self.normalized_state(relative_pos, self.p_max)
        normalized_velocity = self.normalized_state(relative_vel, self.v_max)
        normalized_acc = self.normalized_state(relative_acc, self.a_max)

        return DiscreteState(
            position=self.d_f(normalized_position[0], self.normalized_gps, self.normalized_lps),
            velocity=self.d_f(normalized_velocity[0], self.normalized_gvl, self.normalized_lvl),
            acceleration=self.d_f(normalized_acc[0], self.normalized_gacc, self.normalized_lacc),
            index=angle_index,
        )

    def get_reward(self):
        # rt = rp + rv + r_theta + rdur + rterm.
        pass

    def _get_terminal_term(self, state: DiscreteState, relative_pos: torch.Tensor, lim_pos: torch.Tensor):
        actual_state = torch.tensor([state.position, state.velocity], device=self.device)
        relative_pos = self.normalized_state(relative_pos, self.p_max)
        lim_pos = self.normalized_state(lim_pos, self.p_max)

        if torch.equal(actual_state, self.discretized_goal_state):
            self.r_term = self.r_success
        elif torch.abs(relative_pos[0]) > lim_pos[0]:
            self.r_term = self.r_failure
        else:
            self.r_term = 0


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
