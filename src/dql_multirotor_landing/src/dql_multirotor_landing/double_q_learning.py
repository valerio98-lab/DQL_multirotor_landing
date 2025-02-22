from pathlib import Path
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
    def __init__(self, curriculum_steps: int = 5) -> None:
        self.curriculum_steps = curriculum_steps
        self.Q_table_a = np.zeros((curriculum_steps, 3, 3, 3, 7, 3))
        self.Q_table_b = np.zeros((curriculum_steps, 3, 3, 3, 7, 3))
        self.state_action_counter = np.zeros((curriculum_steps, 3, 3, 3, 7, 3))
        self.check = True

    def save(self, save_path: Path):
        # Avoid file ovewrite
        qa_path = save_path / "Q_table_a.npy"
        qb_path = save_path / "Q_table_b.npy"
        sac_path = save_path / "state_action_count.npy"
        if self.check:
            with open(qa_path, "wb") as f:
                np.save(f, self.Q_table_a)
            with open(qb_path, "wb") as f:
                np.save(f, self.Q_table_b)
            with open(sac_path, "wb") as f:
                np.save(f, self.state_action_counter)

    @staticmethod
    def load(save_path: Path = ASSETS_PATH):
        # Allow to fail gracefully
        with open(save_path / "Q_table_a.npy", "rb") as f:
            qa = np.load(f)
        with open(save_path / "Q_table_b.npy", "rb") as f:
            qb = np.load(f)
        with open(save_path / "state_action_count.npy", "rb") as f:
            sac = np.load(f)
        if qa.shape != qb.shape != sac.shape:
            raise ValueError(
                f"The shapes of Q table a {qa.shape}, Q table b {qb.shape}"
                + f"and State action count {sac.shape} cannot be different"
            )
        agent = DoubleQLearningAgent(len(qa))
        agent.Q_table_a = qa
        agent.Q_table_b = qb
        agent.state_action_counter = sac
        return agent

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
        self.state_action_counter[current_state_action] += 1
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
        return int(np.where(explore, np.random.randint(3), self.predict(state)))

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
