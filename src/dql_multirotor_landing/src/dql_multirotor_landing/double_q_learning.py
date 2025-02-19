from typing import Tuple

import numpy as np

from dql_multirotor_landing import ASSETS_PATH  # type: ignore

CurriculumIdx = int
PositionIdx = int
VelocityIdx = int
AccelerationIdx = int
AngleIdx = int
ActionIdx = int
StateAction = Tuple[
    CurriculumIdx,
    PositionIdx,
    VelocityIdx,
    AccelerationIdx,
    AngleIdx,
    ActionIdx,
]
print(ASSETS_PATH)


class DoubleQLearningAgent:
    def __init__(
        self,
        curriculum_steps: int,
    ) -> None:
        self.curriculum_steps = curriculum_steps
        self.Q_table = np.zeros((curriculum_steps, 3, 3, 3, 7, 3))
        self.Q_table_double = np.zeros((curriculum_steps, 3, 3, 3, 7, 3))
        self.state_action_counter = np.zeros((curriculum_steps, 3, 3, 3, 7, 3))

    def _update_q_table(
        self,
        q_table: np.ndarray,
        current_state_action: Tuple[
            CurriculumIdx,
            PositionIdx,
            VelocityIdx,
            AccelerationIdx,
            AngleIdx,
            ActionIdx,
        ],
        next_state: Tuple[
            CurriculumIdx,
            PositionIdx,
            VelocityIdx,
            AccelerationIdx,
            AngleIdx,
        ],
        alpha: float,
        gamma: float,
        reward: float,
    ):
        # Determine the learning rate
        best_action = np.argmax(self.Q_table[next_state])
        current_state_best_action = next_state + (best_action,)
        # Non-terminal state
        loss = alpha * (
            reward
            + (gamma * self.Q_table_double[current_state_best_action])
            # Check for terminal condition, the position has not changed
            * (int(current_state_action[1] != next_state[1]))
            - self.Q_table[current_state_action]
        )
        q_table[current_state_action] += loss

    def save(self):
        with open(
            ASSETS_PATH / f"curriculum_steps{self.curriculum_steps}" / "Q_table.npy",
            "wb",
        ) as f:
            np.save(f, self.Q_table)
        with open(
            ASSETS_PATH
            / f"curriculum_steps{self.curriculum_steps}"
            / "Q_table_double.npy",
            "wb",
        ) as f:
            np.save(f, self.Q_table_double)
        with open(
            ASSETS_PATH
            / f"curriculum_steps{self.curriculum_steps}"
            / "state_action_counter.npy",
            "wb",
        ) as f:
            np.save(f, self.state_action_counter)

    def load(
        self,
    ):
        try:
            with open(
                ASSETS_PATH
                / f"curriculum_steps{self.curriculum_steps}"
                / "Q_table.npy",
                "rb",
            ) as f:
                np.load(f, self.Q_table)
            with open(
                ASSETS_PATH
                / f"curriculum_steps{self.curriculum_steps}"
                / "Q_table_double.npy",
                "rb",
            ) as f:
                np.load(f, self.Q_table_double)
            with open(
                ASSETS_PATH
                / f"curriculum_steps{self.curriculum_steps}"
                / "state_action_counter.npy",
                "rb",
            ) as f:
                np.load(f, self.state_action_counter)
        except FileNotFoundError:
            print("The requested load file were not found.")

    def transfer_learning(
        self, current_curriculum_step: int, transfer_learning_ratio: float
    ):
        if current_curriculum_step >= 1:
            self.Q_table[current_curriculum_step] = (
                self.Q_table[current_curriculum_step - 1] * transfer_learning_ratio
            )
            self.Q_table_double[current_curriculum_step] = (
                self.Q_table_double[current_curriculum_step - 1]
                * transfer_learning_ratio
            )

    def update(
        self,
        current_state_action: Tuple[
            CurriculumIdx,
            PositionIdx,
            VelocityIdx,
            AccelerationIdx,
            AngleIdx,
            ActionIdx,
        ],
        next_state: Tuple[
            CurriculumIdx,
            PositionIdx,
            VelocityIdx,
            AccelerationIdx,
            AngleIdx,
        ],
        alpha: float,
        gamma: float,
        reward,
    ):
        # Choose the action to follow

        self.state_action_counter[current_state_action] += 1
        self._update_q_table(
            self.Q_table if np.random.uniform(0, 1) < 0.5 else self.Q_table_double,
            current_state_action,
            next_state,
            alpha,
            gamma,
            reward,
        )

    def guess(
        self,
        state: Tuple[
            CurriculumIdx,
            PositionIdx,
            VelocityIdx,
            AccelerationIdx,
            AngleIdx,
        ],
        exploration_rate: float,
    ):
        explore = np.random.uniform(0, 1) < exploration_rate
        return np.where(explore, np.random.randint(3), self.predict(state))

    def predict(
        self,
        state: Tuple[
            CurriculumIdx,
            PositionIdx,
            VelocityIdx,
            AccelerationIdx,
            AngleIdx,
        ],
    ):
        """Function samples actions based on the current Q-values."""
        return np.argmax(np.add(self.Q_table[state], self.Q_table_double[state]) / 2)
