from dataclasses import dataclass
from typing import Literal, Union

import numpy as np


@dataclass
class ObservationRelativeState:
    rel_p_x: float
    rel_v_x: float
    rel_a_x: float
    pitch: float


class Mdp:
    # Definiti in Tabella 3 reward function
    # TODO: ritrovare questi valori nel paper e metterli in fila
    w_p: float = -100.0
    w_v: float = -10.0
    w_theta: float = -1.55
    w_dur: float = -6.0
    w_fail: float = -2.6
    w_suc: float = 2.6
    n_theta: int = 3

    p_max: float = 4.5
    v_max: float = 3.39411
    a_max: float = 1.28
    theta_max: float = np.deg2rad(21.37723)
    discrete_pitches = np.linspace(-theta_max, theta_max, (n_theta * 2) + 1)
    beta: float = 1 / 3
    sigma_a: float = 0.416
    initial_action_values: dict = {
        "pitch": 0,  # [rad]
        "roll": 0,  # [rad]
        "v_z": -0.1,  # [m/s]
        "yaw": 0,  # [rad]
    }
    delta_values: dict = {"pitch": theta_max / n_theta, "yaw": theta_max / n_theta}

    def __init__(
        self,
    ) -> None:
        self.action_values = self.initial_action_values
        """Latest performed action"""
        # self.limits = limits
        self.current_discrete_state = (0, 0, 0, 0)

    def _latest_valid_curriculum_step_for_state(
        self, state_limits: "list[float]", value: float
    ):
        for idx in range(1, len(state_limits)):
            limit = state_limits[idx]
            # TODO Non dovrebbe mai ritornare -1, but you never know
            if not (-limit <= value <= limit):
                return idx - 1
        return len(state_limits) - 1

    def _discretiazion_function(self, continuous_value, goal, limit):
        if -limit <= continuous_value < -goal:
            return 0
        elif -goal <= continuous_value <= goal:
            return 1
        else:
            return 2

    def discrete_state(
        self, msg_rel_state: ObservationRelativeState, limits: "dict[str,list[float]]"
    ):
        continuous_position = msg_rel_state.rel_p_x
        continuous_velocity = msg_rel_state.rel_v_x
        continuous_acceleration = msg_rel_state.rel_a_x

        latest_valid_curriculum_step = min(
            [
                self._latest_valid_curriculum_step_for_state(
                    limits["position"], continuous_position
                ),
                self._latest_valid_curriculum_step_for_state(
                    limits["velocity"], continuous_velocity
                ),
                self._latest_valid_curriculum_step_for_state(
                    limits["acceleration"], continuous_acceleration
                ),
            ]
        )

        position_contraction = self.beta
        if latest_valid_curriculum_step != (len(limits["position"]) - 1):
            position_contraction = (
                limits["position"][latest_valid_curriculum_step + 1]
                / limits["position"][latest_valid_curriculum_step]
            )
        discrete_position = self._discretiazion_function(
            continuous_position,
            continuous_position * position_contraction,
            limits["position"],
        )

        velocity_contraction = self.beta
        if latest_valid_curriculum_step != (len(limits["velocity"]) - 1):
            velocity_contraction = (
                limits["velocity"][latest_valid_curriculum_step + 1]
                / limits["velocity"][latest_valid_curriculum_step]
            )
        discrete_velocity = self._discretiazion_function(
            continuous_velocity,
            continuous_velocity * velocity_contraction,
            limits["velocity"],
        )

        acceleration_contraction = self.sigma_a
        if latest_valid_curriculum_step == (len(limits["velocity"]) - 1):
            acceleration_contraction *= self.beta
        discrete_acceleration = self._discretiazion_function(
            continuous_acceleration,
            continuous_velocity * velocity_contraction,
            limits["acceleration"],
        )

        discrete_pitch = np.argmin(np.abs(self.discrete_pitches - msg_rel_state.pitch))
        # Store for future use
        self.current_discrete_state = (
            latest_valid_curriculum_step,
            discrete_position,
            discrete_velocity,
            discrete_acceleration,
            discrete_pitch,
        )
        return self.current_discrete_state

    def reward(self, msg_rel_state: ObservationRelativeState):
        """
        - Ottenere i valori di posizione e velocita' relativa. Velocità e posizione sono sufficienti per determinare un buon atterraggio.
        - Normalizzare i valori, clippandoli tra -1 e 1. I valori sono ancora continui.
        - Ottenere dalla azione corrente/ l'ultima azione eseguita di pitch e normlizzala
        - Per ogni step nel curiculum learning fino ad ora, consideriamo il valore corrente di shaping come il valore normalizzato e pesato per il w_i relatvo

        """
        continuous_position = msg_rel_state.rel_p_x
        continuous_velocity = msg_rel_state.rel_v_x

        # Normalize
        normalized_continuous_position = np.clip(
            continuous_position / self.p_max, -1, 1
        )
        normalized_continuous_velocity = np.clip(
            continuous_velocity / self.v_max, -1, 1
        )
        normalized_continuous_acceleration = np.clip(
            continuous_velocity / self.a_max, -1, 1
        )

        normalized_pitch = self.action_values["pitch"] / self.theta_max
        # Dobbiamo sapere a che curriculum step stiamo lavorando pe fare il curriculum shaping
        working_curriculum_step = self.current_discrete_state[0]

    def update_action_values(
        self, action: int, parameter: Literal["pitch", "roll"] = "pitch"
    ):
        """Updates the setpoints for the attitude controller of the multi-rotor vehicle."""

        if action == 0:  # Increase
            self.action_values[parameter] = np.min(
                (
                    self.action_values[parameter] + self.delta_values[parameter],
                    self.delta_values[parameter],
                )
            )
        elif action == 1:  # Decrease
            self.action_values[parameter] = np.max(
                (
                    self.action_values[parameter] - self.delta_values[parameter],
                    -self.delta_values[parameter],
                )
            )
        # If action == 2 (do nothing), function exits naturally
