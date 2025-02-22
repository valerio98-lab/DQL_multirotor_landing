"""Module containing the definition for the trainer."""

import math
import pickle
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

from dql_multirotor_landing import ASSETS_PATH  # type: ignore
from dql_multirotor_landing.double_q_learning import DoubleQLearningAgent, StateAction
from dql_multirotor_landing.landing_simulation_env import TrainingLandingEnv


class Trainer:
    # Section 4.5 training:
    # The training is ended as as soon as the agent manages
    # to reach the goal state, associated with the latest
    # step in the sequential curriculum in 96% of the last 100 episodes.
    def __init__(
        self,
        curriculum_steps: int = 5,
        double_q_learning_agent: Optional[DoubleQLearningAgent] = None,
        successive_successful_episodes: int = 100,
        success_rate: float = 0.96,
        max_num_episodes: int = 50000,
        initial_curriculum_step: int = 0,
        seed: int = 42,
        save_path=ASSETS_PATH / datetime.now().strftime(r"%d-%m-%Y %H:%M:%S"),
        *,
        alpha_min: float = 0.02949,
        omega: float = 0.51,
        gamma: float = 0.99,
        scale_modification_value=[
            0.8172650252856599,
            0.8211253690681617,
            0.8257273369742982,
            0.8311571820651724,
        ],
        t_max=20,
        z_init=4.0,
    ) -> None:
        np.random.seed(seed)
        if not double_q_learning_agent:
            double_q_learning_agent = DoubleQLearningAgent(curriculum_steps)
        self.double_q_learning_agent = double_q_learning_agent
        self._curriculum_steps = self.double_q_learning_agent.curriculum_steps
        self._alpha_min = alpha_min
        self._omega = omega
        self._gamma = gamma
        self._scale_modification_value = scale_modification_value
        self._successive_successful_episodes = successive_successful_episodes
        self._success_rate = success_rate
        self._max_num_episodes = max_num_episodes
        self.save_path: Path = save_path

        self._seed = seed
        self._current_episode = 0
        self._working_curriculum_step = initial_curriculum_step
        self._exploration_rate = 0.0
        self._curriculum_episode_count = 0
        self._successes = deque([], maxlen=successive_successful_episodes)
        self._z_init = z_init
        self._alpha = self._alpha_min
        self._t_max = t_max

    def alpha(self, current_state_action: StateAction):
        counter = self.double_q_learning_agent.state_action_counter[
            current_state_action
        ]
        # avoid division by 0
        if counter == 0:
            self._alpha = self._alpha_min
        else:
            self._alpha = float(
                np.max(
                    [
                        np.float_power(1 / (counter), self._omega),
                        self._alpha_min,
                    ]
                )
            )
        if math.isnan(self._alpha):
            raise ValueError(
                f"Leaning rate cannot be NaN, {counter}, {self._omega}, {self._alpha_min}"
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
            self._exploration_rate = max(
                1 + (0.01 - 1) * (current_episode - 800) / (2000 - 800), 0.01
            )
        return self._exploration_rate

    def transfer_learning_ratio(self, curriculum_step: int) -> float:
        if curriculum_step < 1:
            return 1.0
        # For the first 5 curiculum steps the value is known, taken from the paper
        elif curriculum_step < (len(self._scale_modification_value) + 1):
            return self._scale_modification_value[curriculum_step - 1]
        # In any other case `Eq. 31` is applied
        raise ValueError(
            f"Transfer learning can be done up to he 5th curiculum_step, {curriculum_step} is invalid"
        )

    def save(
        self,
    ) -> None:
        """Saves the trainer object to a file using pickle."""
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)
        with open(self.save_path / "tainer.pickle", "wb") as f:
            pickle.dump(self, f)
        self.double_q_learning_agent.save(self.save_path)
        # Make available for training purposes
        # A copy of the latest weights is made available
        # So that is direcly available to the load method
        self.double_q_learning_agent.save(self.save_path / "..")

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
        for self._working_curriculum_step in range(
            self._working_curriculum_step, self._curriculum_steps
        ):
            env: TrainingLandingEnv = gym.make(
                "Landing-Training-v0",
                t_max=self._t_max,
                initial_curriculum_step=self._working_curriculum_step,
                z_init=self._z_init,
            )  # type:ignore
            info = {}
            self._working_curriculum_step = self._working_curriculum_step

            for self._current_episode in range(self._max_num_episodes):
                self._curriculum_episode_count += 1
                current_state = env.reset()
                done = False
                while not done:
                    action = self.double_q_learning_agent.guess(
                        current_state,
                        self.exploration_rate(
                            self._current_episode, self._working_curriculum_step
                        ),
                    )
                    next_state, reward, done, info = env.step(action)
                    current_state_action = current_state + (action,)
                    self.double_q_learning_agent.update(
                        current_state_action,
                        next_state,
                        self.alpha(current_state_action),
                        self._gamma,
                        reward,
                    )
                    if done:
                        break
                    current_state = next_state
                info["Curent episode"] = self._current_episode
                info["Remaining episodes"] = (
                    self._max_num_episodes - self._current_episode + 1
                )
                info["Exploration rate"] = self._exploration_rate
                info["Learning rate"] = self._alpha
                self._successes.append(
                    int("Goal state reached" in info["Termination condition"])
                )
                info["Success rate"] = (
                    sum(self._successes) / self._successive_successful_episodes
                )
                self.save()
                self.log(info)
                # Curiculum promotion
                if info["Success rate"] > self._success_rate:
                    break

            self.double_q_learning_agent.transfer_learning(
                self._working_curriculum_step,
                self.transfer_learning_ratio(
                    self._working_curriculum_step,
                ),
            )
            env.close()
            self.save()

    def log(self, info: Dict[str, Any], clean=True):
        # https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
        # Clear screen and return to the left corner
        # TERMINAL_CONTACT = "\x1b[1;32mSUCCESS\x1b[0m: Touched platform"
        # "\x1b[1;31mFAILURE\x1b[0m: Drone moved too far from platform in x direction"

        writer = SummaryWriter(log_dir=self.save_path / "logs")
        writer.add_scalar(
            "Episode/Success Rate", info["Success rate"], self._curriculum_episode_count
        )
        writer.add_scalar(
            "Episode/Cumulative Reward",
            info["Cumulative reward"],
            self._curriculum_episode_count,
        )
        writer.add_scalar(
            "Episode/Exploration Rate",
            info["Exploration rate"],
            self._curriculum_episode_count,
        )
        writer.add_scalar(
            "Episode/Learning Rate",
            info["Learning rate"],
            self._curriculum_episode_count,
        )
        writer.add_scalar(
            "Episode/Mean reward",
            info["Mean reward"],
            self._curriculum_episode_count,
        )
        writer.add_text(
            "Episode/Termination Condition",
            info["Termination condition"],
            self._curriculum_episode_count,
        )
        if clean:
            print("\x1b[0;0f", end="")
            print("\x1b[J", end="")
        else:
            print("=" * 80)
        print(f"Curiculum step: {self._working_curriculum_step}")
        print(f"Current episode: {self._current_episode}")
        info["Termination condition"] = info["Termination condition"].replace(
            "SUCCESS", "\x1b[1;32mSUCCESS\x1b[0m"
        )
        info["Termination condition"] = info["Termination condition"].replace(
            "FAILURE", "\x1b[1;31mFAILURE\x1b[0m"
        )
        for k, v in info.items():
            print(f"{k}: {v}")
        if clean:
            # Prepare for next print
            print("\x1b[0;0f", end="")
            print("\x1b[J", end="")
            print("\x1b[0m", end="")
        else:
            print("=" * 80)
        writer.close()
