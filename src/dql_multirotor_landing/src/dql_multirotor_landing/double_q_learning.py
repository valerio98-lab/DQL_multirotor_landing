from typing import Tuple, Union

import numpy as np

from dql_multirotor_landing import ASSETS_PATH  # type: ignore

CurriculumIdx = int
PositionIdx = int
VelocityIdx = int
AccelerationIdx = int
AngleIdx = int
ActionIdx = int
State = Tuple[
    Union[CurriculumIdx, int],
    Union[PositionIdx, int],
    Union[VelocityIdx, int],
    Union[AccelerationIdx, int],
    Union[AngleIdx, int],
]

StateAction = Tuple[
    Union[CurriculumIdx, int],
    Union[PositionIdx, int],
    Union[VelocityIdx, int],
    Union[AccelerationIdx, int],
    Union[AngleIdx, int],
    Union[ActionIdx, int],
]


class DoubleQLearningAgent:
    def __init__(self, curriculum_steps: int = 1) -> None:
        self.curriculum_steps = curriculum_steps
        self.Q_table_a = np.zeros((curriculum_steps, 3, 3, 3, 7, 3))
        self.Q_table_b = np.zeros((curriculum_steps, 3, 3, 3, 7, 3))
        self.state_action_counter = np.zeros((curriculum_steps, 3, 3, 3, 7, 3))

    def save(self, curriculum_step: int):
        with open(
            ASSETS_PATH / f"curriculum_steps{curriculum_step}" / "Q_table_a.npy",
            "wb",
        ) as f:
            np.save(f, self.Q_table_a)
        with open(
            ASSETS_PATH / f"curriculum_steps{curriculum_step}" / "Q_table_a.npy",
            "wb",
        ) as f:
            np.save(f, self.Q_table_b)
        with open(
            ASSETS_PATH
            / f"curriculum_steps{curriculum_step}"
            / "state_action_counter.npy",
            "wb",
        ) as f:
            np.save(f, self.state_action_counter)

    def load(self, curriculum_step: int):
        try:
            with open(
                ASSETS_PATH / f"curriculum_steps{curriculum_step}" / "Q_table_a.npy",
                "rb",
            ) as f:
                np.load(f, self.Q_table_a)
            with open(
                ASSETS_PATH / f"curriculum_steps{curriculum_step}" / "Q_table_b.npy",
                "rb",
            ) as f:
                np.load(f, self.Q_table_b)
            with open(
                ASSETS_PATH
                / f"curriculum_steps{curriculum_step}"
                / "state_action_counter.npy",
                "rb",
            ) as f:
                np.load(f, self.state_action_counter)
        except FileNotFoundError:
            print("The requested load file were not found.")

    def insert_curriculum_step(self, curriculum_step: int):
        self.Q_table_a = np.insert(self.Q_table_a, curriculum_step, 0.0, 0)
        self.Q_table_b = np.insert(self.Q_table_a, curriculum_step, 0.0, 0)
        self.state_action_counter = np.insert(self.Q_table_a, curriculum_step, 0.0, 0)

    def transfer_learning(
        self,
        current_curriculum_step: int,
        transfer_learning_ratio: float,
    ):
        self.Q_table_a[current_curriculum_step] = (
            self.Q_table_a[current_curriculum_step - 1] * transfer_learning_ratio
        )
        self.Q_table_b[current_curriculum_step] = (
            self.Q_table_b[current_curriculum_step - 1] * transfer_learning_ratio
        )

    def update(
        self,
        current_state_action: StateAction,
        next_state: State,
        alpha: float,
        gamma: float,
        reward,
    ):
        # Choose the action to follow
        self._update_q_table(
            self.Q_table_a if np.random.uniform(0, 1) < 0.5 else self.Q_table_a,
            current_state_action,
            next_state,
            alpha,
            gamma,
            reward,
        )

    def guess(
        self,
        state: State,
        exploration_rate: float,
    ):
        explore = np.random.uniform(0, 1) < exploration_rate
        return np.where(explore, np.random.randint(3), self.predict(state))

    def predict(
        self,
        state: State,
    ):
        """Function samples actions based on the current Q-values."""
        return int(np.argmax(np.add(self.Q_table_a[state], self.Q_table_b[state]) / 2))

    def _update_q_table(
        self,
        q_table: np.ndarray,
        current_state_action: StateAction,
        next_state: State,
        alpha: float,
        gamma: float,
        reward: float,
    ):
        # Determine the learning rate
        best_action = np.argmax(q_table[next_state])
        current_state_best_action = next_state + (best_action,)
        # Non-terminal state
        loss = alpha * (
            reward
            + (gamma * q_table[current_state_best_action])
            # Check for terminal condition, the position has not changed
            * (int(current_state_action[1] != next_state[1]))
            - q_table[current_state_action]
        )
        q_table[current_state_action] += loss
