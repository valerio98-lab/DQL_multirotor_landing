from copy import deepcopy
from typing import Dict, List, Literal, Tuple

import numpy as np

from dql_multirotor_landing.msg import ObservationRelativeState


class Mdp:
    # Definiti in Tabella 3 reward function
    w_p: float = -100.0
    w_v: float = -10.0
    w_theta: float = -1.55
    w_dur: float = -6.0
    w_fail: float = -2.6
    w_succ: float = 2.6
    n_theta: int = 3
    p_max: float = 4.5
    v_max: float = 3.39411
    a_max: float = 1.28
    theta_max: float = np.deg2rad(21.37723)
    delta_theta: float = np.deg2rad(7.12574)
    discrete_pitches = np.linspace(-theta_max, theta_max, (n_theta * 2) + 1)
    beta: float = 1 / 3
    sigma_a: float = 0.416
    initial_action_values: Dict[str, float] = {
        "pitch": 0,
        "roll": 0,
        "v_z": -0.1,
        "yaw": 0,
    }
    f_ag: float = 22.92
    delta_t: float = 1 / f_ag

    def __init__(
        self,
        limits: Dict[str, List[float]] = {
            "position": [1.0],
            "velocity": [1.0],
            "acceleration": [1.0],
        },
        curriculum_steps=1,
    ) -> None:
        self.action_values = self.initial_action_values
        """Latest performed action"""
        # TODO: Capire il primo stato in cui si trova un mdp
        self.current_discrete_state: Tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
        self.previous_discrete_state: Tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
        self.current_continuos_observation = ObservationRelativeState()
        self.limits = limits
        self.curiculum_steps = (
            curriculum_steps  # min(len(values) for values in self.limits.values())
        )
        # TODO: For now we pass the current limits, this might change in the future
        self.current_shaping_value = {
            "position": np.zeros(self.curiculum_steps),
            "velocity": np.zeros(self.curiculum_steps),
            "pitch": np.zeros(self.curiculum_steps),
        }
        self.previous_shaping_value = {
            "position": np.zeros(self.curiculum_steps),
            "velocity": np.zeros(self.curiculum_steps),
            "pitch": np.zeros(self.curiculum_steps),
        }
        """Stores the current shaping value for each curriculum timestep"""
        self.max_possible_reward_for_one_timestep = 0.0

    def _latest_valid_curriculum_step_for_state(
        self,
        state_limits: "list[float]",
        value: float,
    ):
        for idx in range(1, len(state_limits)):
            limit = state_limits[idx]
            if not (-limit <= value <= limit):
                return idx
        return 0

    def _discretiazion_function(
        self, continuous_value: float, goal: float, limit: float
    ):
        # print(f"Discretization_func, {continuous_value=}, {goal=} ,{limit=}")
        if -limit <= continuous_value < -goal:
            return 0
        elif -goal <= continuous_value <= goal:
            return 1
        else:
            return 2

    def discrete_state(
        self,
        msg_rel_state: ObservationRelativeState,
    ):
        self.current_continuos_observation = msg_rel_state
        continuous_position = np.clip(msg_rel_state.rel_p_x / self.p_max, -1, 1)
        continuous_velocity = np.clip(msg_rel_state.rel_v_x / self.v_max, -1, 1)
        continuous_acceleration = np.clip(msg_rel_state.rel_a_x / self.a_max, -1, 1)

        latest_valid_curriculum_step = min(
            [
                self._latest_valid_curriculum_step_for_state(
                    self.limits["position"], continuous_position
                ),
                self._latest_valid_curriculum_step_for_state(
                    self.limits["velocity"], continuous_velocity
                ),
                self._latest_valid_curriculum_step_for_state(
                    self.limits["acceleration"], continuous_acceleration
                ),
            ]
        )

        position_contraction = self.beta
        if latest_valid_curriculum_step < (self.curiculum_steps - 1):
            position_contraction = (
                self.limits["position"][latest_valid_curriculum_step + 1]
                / self.limits["position"][latest_valid_curriculum_step]
            )
        discrete_position = self._discretiazion_function(
            continuous_position,
            self.limits["position"][latest_valid_curriculum_step]
            * position_contraction,
            self.limits["position"][latest_valid_curriculum_step],
        )

        velocity_contraction = self.beta
        if latest_valid_curriculum_step < (self.curiculum_steps - 1):
            velocity_contraction = (
                self.limits["velocity"][latest_valid_curriculum_step + 1]
                / self.limits["velocity"][latest_valid_curriculum_step]
            )
        discrete_velocity = self._discretiazion_function(
            continuous_velocity,
            self.limits["velocity"][latest_valid_curriculum_step]
            * velocity_contraction,
            self.limits["velocity"][latest_valid_curriculum_step],
        )

        acceleration_contraction = self.sigma_a
        if latest_valid_curriculum_step == (self.curiculum_steps - 1):
            acceleration_contraction *= self.beta
        discrete_acceleration = self._discretiazion_function(
            continuous_acceleration,
            self.limits["acceleration"][latest_valid_curriculum_step]
            * continuous_acceleration,
            self.limits["acceleration"][latest_valid_curriculum_step],
        )

        discrete_pitch = np.argmin(np.abs(self.discrete_pitches - msg_rel_state.pitch))
        # Store for future use
        self.previous_discrete_state_discrete_state = self.current_discrete_state
        self.current_discrete_state = (
            int(latest_valid_curriculum_step),
            int(discrete_position),
            int(discrete_velocity),
            int(discrete_acceleration),
            int(discrete_pitch),
        )
        return self.current_discrete_state

    def reward(self):
        continuous_position = self.current_continuos_observation.rel_p_x
        continuous_velocity = self.current_continuos_observation.rel_v_x

        # Normalize
        normalized_continuous_position = np.clip(
            continuous_position / self.p_max, -1, 1
        )
        normalized_continuous_velocity = np.clip(
            continuous_velocity / self.v_max, -1, 1
        )
        normalized_pitch = self.action_values["pitch"] / self.theta_max

        # Dobbiamo sapere a che curriculum step stiamo lavorando pe fare il curriculum shaping
        current_working_curriculum_step = self.current_discrete_state[0]
        previous_working_curriculum_step = self.previous_discrete_state[0]

        # Il paper non e' chiaro su questo punto, leggendo il paper infatti mai viene fatto riferimento ai curriculum step precedenti
        # Tutavia nella loro repo originale la eward shaping viene fatta in questo modo.
        for curriculum_step in range(self.curiculum_steps):
            self.current_shaping_value["position"][curriculum_step] = self.w_p * np.abs(
                normalized_continuous_position
            )
            self.current_shaping_value["velocity"][curriculum_step] = self.w_v * np.abs(
                normalized_continuous_velocity
            )
            self.current_shaping_value["pitch"][curriculum_step] = (
                self.w_theta * np.abs(normalized_pitch)
            )
        # Eq 24
        r_p_max = (
            np.abs(self.w_p)
            * self.limits["velocity"][current_working_curriculum_step]
            * self.delta_t
        )
        # Eq 25
        r_v_max = (
            np.abs(self.w_v)
            * self.limits["acceleration"][current_working_curriculum_step]
            * self.delta_t
        )
        # Eq 26
        r_theta_max = (
            np.abs(self.w_theta)
            * (self.delta_theta / self.theta_max)
            * self.limits["velocity"][current_working_curriculum_step]
        )

        # Eq 27
        r_dur_max = (
            self.w_dur
            * self.limits["velocity"][current_working_curriculum_step]
            * self.delta_t
        )
        # Eq 28
        r_max = r_p_max + r_v_max + r_theta_max + r_dur_max

        # Kinda equation 20
        r_p = np.clip(
            self.current_shaping_value["position"][current_working_curriculum_step]
            - self.previous_shaping_value["position"][current_working_curriculum_step],
            -r_p_max,
            r_p_max,
        )
        r_v = np.clip(
            self.current_shaping_value["velocity"][current_working_curriculum_step]
            - self.previous_shaping_value["velocity"][current_working_curriculum_step],
            -r_v_max,
            r_v_max,
        )
        r_theta = (
            self.w_theta
            * (
                np.abs(
                    self.current_shaping_value["pitch"][current_working_curriculum_step]
                )
                - np.abs(
                    self.previous_shaping_value["pitch"][
                        current_working_curriculum_step
                    ]
                )
            )
            / self.theta_max
            * self.limits["velocity"][current_working_curriculum_step]
        )
        r_dur = (
            self.w_dur
            * self.limits["velocity"][current_working_curriculum_step]
            * self.delta_t
        )
        # Condiion |p_x| > p_lim e' iconducibile al caso curiculum step precedente a quello attuale.
        if current_working_curriculum_step < previous_working_curriculum_step:
            r_term = self.w_fail * r_max
        elif current_working_curriculum_step > previous_working_curriculum_step:
            r_term = self.w_succ * r_max
        else:
            r_term = 0.0
        # Update curiculum shaping terms

        self.previous_shaping_value["position"] = deepcopy(
            self.current_shaping_value["position"]
        )
        self.previous_shaping_value["velocity"] = deepcopy(
            self.current_shaping_value["velocity"]
        )

        self.previous_shaping_value["pitch"] = deepcopy(
            self.current_shaping_value["pitch"]
        )

        r_t = r_p + r_v + r_theta + r_dur + r_term
        # TODO: look if max_reward should be kept or returned
        self.max_possible_reward_for_one_timestep = r_max
        return r_t

    def update_action_values(
        self, action: int, parameter: Literal["pitch", "roll"] = "pitch"
    ):
        """Updates the setpoints for the attitude controller of the multi-rotor vehicle."""

        if action == 0:  # Increase
            self.action_values[parameter] = np.min(
                (
                    self.action_values[parameter] + self.delta_theta,
                    self.theta_max,
                )
            )
        elif action == 1:  # Decrease
            self.action_values[parameter] = np.max(
                (
                    self.action_values[parameter] - self.delta_theta,
                    -self.theta_max,
                )
            )
        # If action == 2 (do nothing)

    def reset(self):
        self.action_values = self.initial_action_values
        # TODO: Capire il primo stato in cui si trova un mdp
        self.current_discrete_state: Tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
        self.previous_discrete_state: Tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
        # TODO: For now we pass the currnt limits, this might change in the future
        self.current_shaping_value = {
            "position": np.zeros(self.curiculum_steps),
            "velocity": np.zeros(self.curiculum_steps),
            "pitch": np.zeros(self.curiculum_steps),
        }
        self.previous_shaping_value = {
            "position": np.zeros(self.curiculum_steps),
            "velocity": np.zeros(self.curiculum_steps),
            "pitch": np.zeros(self.curiculum_steps),
        }
        self.max_possible_reward_for_one_timestep = 0.0
