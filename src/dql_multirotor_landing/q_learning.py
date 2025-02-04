from dataclasses import astuple, dataclass
from typing import Optional, TypeAlias

import numpy as np
from gymnasium import Env

from dql_multirotor_landing.environment import mdp


@dataclass
class Limits:
    """Definition of the limits as described in equation 38,39,40"""

    p: float
    """defines the maximum allowable deviation of the UAV from the target landing position at a given curriculum step. It get shrinked as training progresses to force the landing to be more precise"""
    v: float
    """Limit velocity, define the maximum allowable velocity and acceleration at each curriculum step."""
    a: float
    """acceleration limit define the maximum allowable velocity and acceleration"""
    sigma: float
    """State action scaling factor that shrinks (refine the state space) at each curriculum step."""
    t_0: float
    """Time needed fo the platform to each $x_{max}$ using $a_{mpmax}$"""
    p_max: float
    """Maximum initial position deviation. This represents the UAV s maximum allowed starting distance from the platform at the first curriculum step."""
    v_max: float
    """Maximum initial velocity. This represents the largest velocity allowed at the first curriculum step. It is used for normalization."""
    a_mpmax: float
    """Maximum acceleration of the moving platform"""

    def contraction(self, curriculum_step: int):
        t_i = (self.sigma**curriculum_step) * self.t_0
        self.p = (0.5 * self.a_mpmax * (t_i**2)) / self.p_max
        self.v = (self.a_mpmax * t_i) / self.v_max
        # Ensure that this is consistent, should not be necessary
        self.a = 1


@dataclass
class Goals:
    p: float
    """Goal position"""

    v: float
    """Goal velocity"""

    a: float
    """Goal acceleration"""
    beta_p: float
    """Position shrink factor"""
    beta_v: float
    """Velocity shrink factor"""
    beta_a: float
    """Acceleration shrink factor"""
    # TODO: Understand how to get it
    sigma_a: float
    """Acceleration contraction factor"""

    def contraction(self, limits: Limits):
        self.p = self.beta_p * limits.p
        self.v = self.beta_v * limits.v
        self.a = self.beta_a


class QTableIdx:
    position_idx: int
    velocity_idx: int
    acceleration_idx: int
    pitch_idx: int
    action_idx: Optional[int] = None

    def __init__(self, discrete_state: mdp.DiscreteState) -> None:
        self.position_idx = discrete_state.position
        self.velocity_idx = discrete_state.velocity
        self.acceleration_idx = discrete_state.acceleration
        self.pitch_idx = discrete_state.pitch


StateCountPairIdx: TypeAlias = QTableIdx


class QTable:
    data: np.ndarray

    def __init__(self, num_positions, num_velocities, num_accelerations, num_pitches, num_actions) -> None:
        self.data = np.zeros((num_positions, num_velocities, num_accelerations, num_pitches, num_actions))

    def transfer_learning(self, current_r_max: float, previous_r_max: float) -> None:
        self.data = (current_r_max / previous_r_max) * self.data

    def __setitem__(self, index: QTableIdx, key):
        if index.action_idx is not None:
            self.data[index.position_idx, index.velocity_idx, index.acceleration_idx, index.pitch_idx] = key
        else:
            self.data[
                index.position_idx, index.velocity_idx, index.acceleration_idx, index.pitch_idx, index.action_idx
            ] = key

    def __getitem__(self, index: QTableIdx):
        result = self.data[index.position_idx, index.velocity_idx, index.acceleration_idx, index.pitch_idx]
        if index.action_idx is not None:
            result = result[index.action_idx]
        return result


StateCountPair: TypeAlias = QTable


class CurriculumLearner:
    def __init__(
        self,
        env: Env,
        omega: float,
        alpha_min: float,
        total_curiculum_steps: int,
        num_states: int,
        num_actions: int,
        num_positions: int,
        num_velocities: int,
        num_pitches: int,
        num_accelerations: int,
        sigma: float,
        initial_exploration_rate: float,
        gamma: float,
        t_0: float,
        p_max: float,
        v_max: float,
        a_mpmax: float,
        beta_p: float,
        beta_v: float,
        beta_a: float,
        sigma_a: float,
        total_episodes: int,
        total_steps: int,
    ) -> None:
        self.env = env
        """Simulation environment"""
        self.gamma = gamma
        """Discount factor used for the Q table update"""

        self.omega = omega
        """Decay factor for the leaning rate"""

        self.alpha_min = alpha_min
        """Minimal leaning ability threshold. Once the leaning rate arrives here it stops decreasing."""

        self.total_curiculum_steps = total_curiculum_steps
        """Total number of curriculum steps"""

        # self.num_states = num_states
        # """Number of possstates"""

        self.num_actions = num_actions
        """Numbe of possible discrete actions"""

        self.num_positions = num_positions
        """Number of possible discrete positions"""

        self.num_velocities = num_velocities
        """Number of possible discrete velocities"""

        self.num_accelerations = num_accelerations
        """Number of possible discrete accelerations"""

        self.num_pitches = num_pitches
        """Number of possible discrete pitch angles"""

        # Double Q learning assumes we use two different Q tables
        self.Q_table1 = np.zeros((num_positions, num_velocities, num_accelerations, num_pitches, num_actions))
        self.Q_table2 = np.zeros((num_positions, num_velocities, num_accelerations, num_pitches, num_actions))

        self.state_action_pair_count = np.zeros(
            (num_positions, num_velocities, num_accelerations, num_pitches, num_actions)
        )
        """Count of how many times a state action pair has been visited"""

        self.limits = Limits(0, 0, 1, sigma, t_0, p_max, v_max, a_mpmax)
        """Limits define, at each curriculum step, the maximum allowable normalized actions.  Initially they are very coarse and get refined as the curriculum sequence advances."""

        self.goals = Goals(0, 0, 1, beta_p, beta_v, beta_a, sigma_a)
        """Goals define, at each curriculum step, the current goals. Initially they are very coarse and get refined as the curriculum sequence advances."""

        self.total_episodes = total_episodes
        """Total number of episodes for each curiculum step"""

        self.total_steps = total_steps
        """Total number of episodes for each episode"""
        # self.action_space = mdp.ActionSpace(num_actions)
        # """The possible angle adjustements the agent perform"""

        self.initial_exploration_rate = initial_exploration_rate
        """Initial value of the exploration rate. This should decay as ..."""
        # TODO: It should not be this
        self.exploration_rate = initial_exploration_rate

    def learning_rate(self, index: StateCountPairIdx) -> float:
        """Returns the curent learning rate using Eq. 30"""
        n_c = self.state_action_pair_count[
            index.position_idx, index.velocity_idx, index.acceleration_idx, index.pitch_idx, index.action_idx
        ]
        return max([((n_c + 1) ** (-self.omega)), self.alpha_min])  # type:ignore

    def decay_exploration_rate(
        self,
    ):
        self.exploration_rate *= 0.995
        self.exploration_rate = max(self.exploration_rate, 0.01)
        return self.initial_exploration_rate

    def multi_resolution_train(self):
        previous_r_max: Optional[float] = None
        for curriculum_step in range(self.total_curiculum_steps):
            # Calculate contraction for limits using: Eq. 38-40
            self.limits.contraction(curriculum_step)

            # Calculate contraction for goals using: Eq. 41-43
            self.goals.contraction(self.limits)

            # Initialize the state space with the limits and the goals
            state_space = mdp.StateSpace(
                self.goals.p,
                self.limits.p,
                self.goals.v,
                self.limits.v,
                self.goals.a,
                self.limits.a,
            )

            if curriculum_step > 0:
                assert previous_r_max is not None
                # Get the curent known maximum reward using Eq: 28
                current_r_max = state_space.get_max_reward()
                # Apply transfer learning from the previous curriculum step using Eq: 31
                self.Q_table1 = (current_r_max / previous_r_max) * self.Q_table1
                self.Q_table2 = (current_r_max / previous_r_max) * self.Q_table2

            self.double_q_learning(state_space)
            previous_r_max = current_r_max

    def update_q_table(
        self,
        decision_policy: np.ndarray,
        behaviour_policy: np.ndarray,
        current_idx: QTableIdx,
        next_idx: QTableIdx,
        reward: float,
    ):
        best_action = np.argmax(
            decision_policy[
                current_idx.position_idx, current_idx.velocity_idx, current_idx.acceleration_idx, current_idx.pitch_idx
            ]
        )

        decision_policy[
            current_idx.position_idx,
            current_idx.velocity_idx,
            current_idx.acceleration_idx,
            current_idx.pitch_idx,
            current_idx.action_idx,
        ] += self.learning_rate(current_idx) * (
            reward
            + self.gamma
            * behaviour_policy[
                next_idx.position_idx,
                next_idx.velocity_idx,
                next_idx.acceleration_idx,
                next_idx.pitch_idx,
                best_action,
            ]
            - decision_policy[
                current_idx.position_idx,
                current_idx.velocity_idx,
                current_idx.acceleration_idx,
                current_idx.pitch_idx,
                current_idx.action_idx,
            ]
        )

    def double_q_learning(self, state_space: mdp.StateSpace):
        for _episode in range(1, self.total_episodes):
            observation, _info = self.env.reset()
            current_continuous_state = mdp.ContinuousState(
                observation["relative_position"],
                observation["relative_velocity"],
                observation["relative_acceleration"],
                observation["relative_orientation"][2],
            )
            # Discretize the current state
            current_discrete_state = state_space.get_discretized_state(current_continuous_state)
            for _episode in range(1, self.total_steps):
                current_idx = QTableIdx(current_discrete_state)  # type:ignore
                # Explore
                if np.random.rand() < self.exploration_rate:
                    action = state_space.sample()
                # Commit
                else:
                    # Get the action based on both Q table1 and Q table2
                    action = np.argmax(
                        (
                            self.Q_table1[
                                current_idx.position_idx,
                                current_idx.velocity_idx,
                                current_idx.acceleration_idx,
                                current_idx.pitch_idx,
                            ]
                            + self.Q_table2[
                                current_idx.position_idx,
                                current_idx.velocity_idx,
                                current_idx.acceleration_idx,
                                current_idx.pitch_idx,
                            ]
                        )
                        / 2
                    )

                current_idx.action_idx = int(action)
                # Update the state action pair count
                self.state_action_pair_count[
                    current_idx.position_idx,
                    current_idx.velocity_idx,
                    current_idx.acceleration_idx,
                    current_idx.pitch_idx,
                    current_idx.action_idx,
                ] += 1
                observation, _reward, terminated, truncated, _info = self.env.step(action)
                new_continuous_state = mdp.ContinuousState(
                    observation["relative_position"],
                    observation["relative_velocity"],
                    observation["relative_acceleration"],
                    observation["relative_orientation"][2],
                )
                reward = state_space.get_reward(new_continuous_state)
                new_discrete_state = state_space.get_discretized_state(new_continuous_state)

                next_idx = QTableIdx(new_discrete_state)

                # Update either Q table1 or Q table2
                if np.random.rand() < 0.5:
                    self.update_q_table(
                        self.Q_table1,
                        self.Q_table2,
                        current_idx,
                        next_idx,
                        reward,
                    )

                else:
                    self.update_q_table(
                        self.Q_table2,
                        self.Q_table1,
                        current_idx,
                        next_idx,
                        reward,
                    )
                current_discrete_state = new_discrete_state
                if terminated or truncated:
                    break
        self.decay_exploration_rate()


"""
For each curriculum_step in total_curriculum_steps:

    Calculate contraction for limits using:
        p_lim = sigma^(2 * step) * p_max  (Eq. 38)
        v_lim = sigma^step * v_max  (Eq. 39)
        a_lim = a_max  (Eq. 40)

    Calculate contraction for goals using:
        p_goal = β_p * p_lim  (Eq. 41)
        v_goal = β_v * v_lim  (Eq. 42)
        a_goal = β_a * a_lim  (Eq. 43)

    `Pass the contractions to Valerio`: 
        1. Initialize an object StateSpace
        2. Pass to the object the contractions (p_lim, p_goal, etc...)
    
    `Recieve from Valerio the max reward values`:
        reward_max = StateSpace.get_max_reward()

    If curriculum_step > 0:
        Scale Q-tables using Eq. (31) to transfer knowledge from previous step

    # Q-Learning with Double Q-Learning Algorithm
    For each episode in total_episodes:
        
        Initialize state based on current curriculum step 
        Initialize the environment: 
            1. obs = env.reset() ##Isaac env
            2. continuous_state = ContinuousState(position=obs.position, etc...)
            3. StateSpace.set_last_state(continuous_state) ##obs has to be a ContinuousState dataclass
            4. discrete_state = StateSpace.get_discrete_state(obs) 

        
        For each step in episode_steps:
            With probability ε select a random action (exploration)
            Otherwise, select action with max Q-value (exploitation)
            
            (TODO: talk about how to select the action)

            Execute action: 
                1. obs = env.step(action) ##Isaac env
                2. continuous_state = ContinuousState(position=obs.position, etc...)
                3. reward = StateSpace.get_reward(continuous_state)
                4. StateSpace.set_last_state(continuous_state) 
                5. discrete_state = StateSpace.get_discretized_state(continuous_state)
         
            
            Update learning rate using Eq. (30)
            
            With 50% probability, update Q1:
                Q1[state, action] = ...
            Else, update Q2:
                Q2[state, action] = ...
            
            # Check if goal state is reached or episode ends
            If goal_state_reached or episode_ends:
                Break   (TODO: talk about this condition for isaac and reward terms)
            
        Decay exploration rate ε as episode progresses
"""
