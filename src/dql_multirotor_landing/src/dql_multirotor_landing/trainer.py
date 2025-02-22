"""Module containing the definition for the trainer."""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import gym
import numpy as np

from dql_multirotor_landing import ASSETS_PATH  # type: ignore
from dql_multirotor_landing.double_q_learning import DoubleQLearningAgent, StateAction
from dql_multirotor_landing.landing_simulation_env import TrainingLandingEnv


class Trainer:
    _alpha_min: float = 0.02949
    _omega: float = 0.51
    _gamma: float = 0.99
    _scale_modification_value = [
        0.8172650252856599,
        0.8211253690681617,
        0.8257273369742982,
        0.8311571820651724,
    ]
    # Section 4.5 training:
    # The training is ended as as soon as the agent manages
    # to reach the goal state, associated with the latest
    # step in the sequential curriculum in 96% of the last 100 episodes.
    _success_rate = 0.96
    _t_max = 20  # seconds

    def __init__(
        self,
        curriculum_steps: int = 5,
        double_q_learning_agent: Optional[DoubleQLearningAgent] = None,
        number_of_successful_episodes: int = 100,
        training_successful_fraction: float = 0.96,
        max_num_episodes: int = 50000,
        *,
        initial_curriculum_step: int = 0,
        seed: int = 42,
    ) -> None:
        np.random.seed(seed)
        if not double_q_learning_agent:
            double_q_learning_agent = DoubleQLearningAgent()
        self.double_q_learning_agent = double_q_learning_agent
        self._curriculum_steps = self.double_q_learning_agent.curriculum_steps

        self._number_of_successful_episodes = number_of_successful_episodes
        self._training_successful_fraction = training_successful_fraction
        self._max_num_episodes = max_num_episodes

        self._seed = seed
        self._current_episode = 0
        self._working_curriculun_step = initial_curriculum_step
        self._exploration_rate = 0.0
        self._alpha = self._alpha_min

    def alpha(self, current_state_action: StateAction):
        counter = self.double_q_learning_agent.state_action_counter[
            current_state_action
        ]
        self._alpha = float(
            np.max(
                [
                    (1 / (counter) if counter != 0 else 0) ** self._omega,
                    self._alpha_min,
                ]
            )
        )
        return self._alpha

    def exploration_rate(self, current_episode: int, current_curriculum_step: int):
        """For the first curriculum step, the exploration rate schedule is
        empirically set to ε = 1 (episode 0-800) before it is linearly
        reduced to ε = 0.01 (episode 800-2000).
        For all later curriculum steps, it is ε = 0."""
        if current_curriculum_step > 0:
            self._exploration_rate = 0.0
        elif 0 <= current_episode <= 800:
            self._exploration_rate = 1.0
        else:
            self._exploration_rate = min(
                0.01, 1 + (0.01 - 1) * (current_episode - 800) / (2000 - 800)
            )
        return self._exploration_rate

    def transfer_learning_ratio(self, curriculum_step: int) -> float:
        if curriculum_step < 1:
            return 1.0
        # For the first 5 curiculum steps the value is known, taken from the paper
        elif curriculum_step < len(self._scale_modification_value):
            return self._scale_modification_value[curriculum_step]
        # In any other case `Eq. 31` is applied
        raise ValueError(
            f"Transfer learning can be done up to he 4th curiculum_step, {curriculum_step} is invalid"
        )

    def save(
        self,
    ) -> None:
        """Saves the trainer object to a file using pickle."""
        save_path: Path = ASSETS_PATH
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "tainer.pickle", "wb") as f:
            pickle.dump(self, f)
        self.double_q_learning_agent.save(save_path)

    @staticmethod
    def load() -> "Trainer":
        """Loads a trainer object from a pickle file."""
        save_path = ASSETS_PATH
        with open(save_path / "tainer.pickle", "rb") as f:
            trainer = pickle.load(f)
        agent = DoubleQLearningAgent.load(save_path)
        trainer.double_q_learning_agent = agent
        return trainer

    def curriculum_training(
        self,
    ):
        env: TrainingLandingEnv = gym.make("Landing-Training-v0")  # type:ignore
        for self._working_curriculum_step in range(self._curriculum_steps):
            self._working_curriculum_step = self._working_curriculum_step
            env.set_curriculum_step(self._working_curriculum_step)
            for current_episode in range(self._max_num_episodes):
                self._current_episode = current_episode
                current_state = env.reset()
                done = False
                while not done:
                    action = self.double_q_learning_agent.guess(
                        current_state,
                        self.exploration_rate(
                            current_episode, self._working_curriculum_step
                        ),
                    )
                    next_state, reward, done, info = env.step(action)
                    current_state_action = current_state + (action,)
                    self.double_q_learning_agent.state_action_counter[
                        current_state_action
                    ] += 1
                    self.double_q_learning_agent.update(
                        current_state_action,
                        next_state,
                        self.alpha(current_state_action),
                        self._gamma,
                        reward,
                    )
                    print(info)
                    if done:
                        info["Curent episode"] = self._current_episode
                        info["Remaining episodes"] = (
                            self._max_num_episodes - current_episode
                        )
                        # TODO add success rate
                        # self.log(info)
                        print(info)
                        self.save()
                        break
                    current_state = next_state
            self.double_q_learning_agent.insert_curriculum_step(
                self._working_curriculum_step
            )
            self.double_q_learning_agent.transfer_learning(
                self._working_curriculum_step,
                self.transfer_learning_ratio(
                    self._working_curriculum_step,
                ),
            )

            self.save()

    def log(self, info: Dict[str, Any]):
        # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
        # Clear screen and return to the left corner
        print("\x1b[0;0f", end="")
        print("\x1b[J", end="")
        print(f"Curiculum step: {self._working_curriculum_step}")
        for k, v in info.items():
            print(f"{k}: {v}")
        print("\x1b[0;0f", end="")
        print("\x1b[J", end="")
        # Prepare for next print
        print("\x1b[0;0f", end="")
        print("\x1b[J", end="")
        print("\x1b[0m", end="")
