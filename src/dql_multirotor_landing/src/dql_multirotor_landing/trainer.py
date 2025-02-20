"""Module containing the definition for the trainer."""

from time import sleep

import gym
import numpy as np

from dql_multirotor_landing.double_q_learning import DoubleQLearningAgent, StateAction
from dql_multirotor_landing.landing_simulation_env import (
    LandingSimulationEnv,  # noqa: F401
)


class Trainer:
    __alpha_min: float = 0.02949
    __omega: float = 0.51
    __gamma: float = 0.99
    __scale_modification_value = [
        0.8172650252856599,
        0.8211253690681617,
        0.8257273369742982,
        0.8311571820651724,
    ]

    def __init__(
        self,
        double_q_learning_agent: DoubleQLearningAgent,
        # TODO: Should be 100
        number_of_successful_episodes: int = 1,
        training_successful_fraction=0.96,
        max_num_episodes: int = 50000,
        *,
        log_freq: int = 1,
        log_latest: int = 20,
        initial_curriculum_step: int = 0,
        seed: int = 42,
    ) -> None:
        np.random.seed(seed)
        self.__double_q_learning_agent = double_q_learning_agent
        self.__curriculum_steps = self.__double_q_learning_agent.curriculum_steps
        self.__initial_curiculum_step = (
            initial_curriculum_step
            if initial_curriculum_step <= self.__curriculum_steps
            else self.__curriculum_steps
        )
        self.__number_of_successful_episodes = number_of_successful_episodes
        self.__training_successful_fraction = training_successful_fraction
        self.__max_num_episodes = max_num_episodes

        # Logging
        self.__seed = seed

    def alpha(self, current_state_action: StateAction):
        counter = self.__double_q_learning_agent.state_action_counter[
            current_state_action
        ]
        return np.max(
            [
                (1 / (counter) if counter != 0 else 0) ** self.__omega,
                self.__alpha_min,
            ]
        )

    def exploration_rate(self, current_episode: int, current_curriculum_step: int):
        """For the first curriculum step, the exploration rate schedule is
        empirically set to ε = 1 (episode 0-800) before it is linearly
        reduced to ε = 0.01 (episode 800-2000).
        For all later curriculum steps, it is ε = 0."""
        if current_curriculum_step > 0:
            return 0
        if 0 <= current_episode <= 800:
            return 1
        return min(0.01, 1 + (0.01 - 1) * (current_episode - 800) / (2000 - 800))

    def _transfer_learning(self, q_table, idx):
        # If we're at the first curriculum step, skip
        if self.current_curriculum_step < 1:
            return
        # For the first 4 curiculum steps the value is known, taken from the paper
        if idx < len(self.__scale_modification_value):
            q_table[len(q_table) - 1] = (
                q_table[len(q_table) - 2] * self.__scale_modification_value[idx]
            )
            return
        # In any other case `Eq. 31` is applied
        # TODO: Inserire eq 31

    def transfer_learning_ratio(self, curriculum_step: int) -> float:
        if curriculum_step < 1:
            return 1.0
        # For the first 4 curiculum steps the value is known, taken from the paper
        elif curriculum_step < len(self.__scale_modification_value):
            return self.__scale_modification_value[curriculum_step]
        else:
            # TODO: We should implement the equation (?)
            return 1.0

    def curriculum_training(
        self,
    ):
        env: LandingSimulationEnv = gym.make("landing_simulation-v0")  # type:ignore
        # TODO: This should be ideally just removed
        # -----
        env.reset()
        env.unpause_sim()
        sleep(2)
        env.pause_sim()
        env.reset()
        env.unpause_sim()
        sleep(2)
        env.pause_sim()
        # -----
        for current_curriculum_step in range(self.__curriculum_steps):
            self.current_curriculum_step = current_curriculum_step
            for current_episode in range(self.__max_num_episodes):
                current_state = env.reset()
                for _ in range(current_episode):
                    action = self.__double_q_learning_agent.guess(
                        current_state,
                        self.exploration_rate(current_episode, current_curriculum_step),
                    )
                    next_state, reward, done, info = env.step(action)
                    current_state_action = current_state + (action,)
                    self.__double_q_learning_agent.state_action_counter[
                        current_state_action
                    ] += 1
                    self.__double_q_learning_agent.update(
                        current_state_action,
                        next_state,
                        self.alpha(current_state_action),
                        self.__gamma,
                        reward,
                    )
                    if done:
                        break
            self.__double_q_learning_agent.insert_curriculum_step(
                current_curriculum_step
            )
            self.__double_q_learning_agent.transfer_learning(
                current_curriculum_step,
                self.transfer_learning_ratio(
                    current_curriculum_step,
                ),
            )
        return
