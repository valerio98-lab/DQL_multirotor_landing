from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class Limits:
    """Definition of the limits as described in equation 38,39,40"""

    p: np.ndarray
    """defines the maximum allowable deviation of the UAV from the target landing position at a given curriculum step. It get shrinked as training progresses to force the landing to be more precise"""
    v: np.ndarray
    """Limit velocity, define the maximum allowable velocity and acceleration at each curriculum step."""
    a: np.ndarray
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

    def shrink(self, i):
        t_i = (self.sigma**i) * self.t_0
        self.p[i] = (0.5 * self.a_mpmax * (t_i**2)) / self.p_max
        self.v[i] = (self.a_mpmax * t_i) / self.v_max
        # Ensue that this is consistent, but not necessary
        self.a[i] = 1


@dataclass
class Goals:
    p: np.ndarray
    """Goal position"""

    v: np.ndarray
    """Goal velocity"""

    a: np.ndarray
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

    def shrink(self, i, p_lim, v_lim, a_lim):
        self.p[i] = self.beta_p * p_lim
        self.v[i] = self.beta_v * v_lim
        self.a[i] = self.beta_a * a_lim


class CurriculumLearner:
    def __init__(
        self,
        env,
        learning_rate,
        omega,
        alpha_min,
        gamma,
        n_theta,
        n_cs,
        num_states,
        num_actions,
        sigma,
        t_0,
        p_max,
        v_max,
        a_mpmax,
        beta_p,
        beta_v,
        beta_a,
        sigma_a,
        x_max,
    ) -> None:
        self.env = env
        """Simulation environment"""
        self.learning_rate = learning_rate
        r"""$\alpha$ in equation 30"""
        self.omega = omega
        r"""$\omega$ in equation 30"""
        self.alpha_min = alpha_min
        r"""$\alpha_{min}$ in equation 30"""

        self.gamma = gamma
        r"""Discount factor"""
        self.x_max = x_max
        self.n_theta = n_theta
        """Number of intervals used to discretize an observation of the environment. $n_{\theta}$ in equation 9"""
        self.n_cs = n_cs
        """Number of curriculum steps"""
        self.Q_table1 = np.zeros((num_states, num_actions))
        self.Q_table2 = np.zeros((num_states, num_actions))
        self.sigma = sigma
        self.limits = Limits(
            np.zeros(self.n_cs), np.zeros(self.n_cs), np.ones(self.n_cs), sigma, t_0, p_max, v_max, a_mpmax
        )
        """Definition of the limits as described in equation 38,39,40"""

        self.goals = Goals(
            np.zeros(self.n_cs), np.zeros(self.n_cs), np.zeros(self.n_cs), beta_p, beta_v, beta_a, sigma_a
        )
        self.a_mpmax = a_mpmax

    def exponential_goal_contraction(self, i):
        """Exponential contraction of the goal position boundary, ensuring smooth curriculum learning."""
        return (self.sigma ** (2 * (i + 1))) * self.x_max

    def compute_worst_case_motion(self, i):
        """Compute worst-case motion values for position, velocity, and acceleration."""
        a_wc = self.a_mpmax  # Worst-case acceleration remains constant
        v_wc = self.a_mpmax * i  # Worst-case velocity (Equation 36)
        p_wc = 0.5 * self.a_mpmax * i**2  # Worst-case position (Equation 37)
        return p_wc, v_wc, a_wc

    def multi_resolution_train(self): ...
        r_pmax = ...
        """Maximum reward function for reducing the relative position"""
        r_vmax = ...
        """Maximum reward function for reducing the relative velocity"""
        r_thetamax = ...
        """Maximum reward function for reducing the relative pitch angle"""
        r_durmax = ...
        """This tries to descourage staying in place. (?)"""
        # TODO: might need to add the termination.
        r_max = r_pmax + r_vmax + r_thetamax + r_durmax
        """Maximum attainable reward, is an papplication of equation 19."""
    def double_q_learning(self):...


"""
For each curriculum_step in total_curriculum_steps:

    Calculate contraction for relative position using Eqs. (32-34)
    Calculate contraction for limits using:
        p_lim = sigma^(2 * step) * p_max  (Eq. 38)
        v_lim = sigma^step * v_max  (Eq. 39)
        a_lim = a_max  (Eq. 40)

    Calculate contraction for goals using:
        p_goal = β_p * p_lim  (Eq. 41)
        v_goal = β_v * v_lim  (Eq. 42)
        a_goal = β_a * a_lim  (Eq. 43)

    `Pass the contractions to Valerio`
    `Recieve from Valerio the max reward values`

    If curriculum_step > 0:
        Scale Q-tables using Eq. (31) to transfer knowledge from previous step

    # Q-Learning with Double Q-Learning Algorithm
    For each episode in total_episodes:
        Initialize state based on current curriculum step
        Discretize initial state
        
        For each step in episode_steps:
            With probability ε select a random action (exploration)
            Otherwise, select action with max Q-value (exploitation)
            
            Execute action, observe reward and next state
        
            Discretize next state
            
            Update learning rate using Eq. (30)
            
            With 50% probability, update Q1:
                Q1[state, action] = ...
            Else, update Q2:
                Q2[state, action] = ...
            
            # Check if goal state is reached or episode ends
            If goal_state_reached or episode_ends:
                Break
            
        Decay exploration rate ε as episode progresses
"""