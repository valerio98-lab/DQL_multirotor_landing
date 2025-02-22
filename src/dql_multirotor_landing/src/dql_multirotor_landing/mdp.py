import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dql_multirotor_landing.msg import Action, Observation


class ContinuousObservation:
    def __init__(
        self,
        observation: Observation = Observation(),
        pitch: float = 0.0,
        roll: float = 0.0,
        abs_p_z: float = 0.0,
        contact: bool = False,
    ) -> None:
        self.rel_p_x = observation.rel_p_x
        self.rel_p_y = observation.rel_p_y
        self.rel_p_z = observation.rel_p_z
        self.rel_v_x = observation.rel_v_x
        self.rel_v_y = observation.rel_v_y
        self.rel_v_z = observation.rel_v_z
        self.rel_a_x = observation.rel_a_x
        self.rel_a_y = observation.rel_a_y
        self.rel_a_z = observation.rel_a_z
        self.pitch = pitch
        self.roll = roll
        self.abs_p_z = abs_p_z
        self.contact = contact


@dataclass
class RewardShapingValue:
    position: float = field(default_factory=lambda: 0)
    velocity: float = field(default_factory=lambda: 0)
    angle: float = field(default_factory=lambda: 0)


@dataclass
class Limits:
    _working_curriculum_step: int
    _position: List[float] = field(
        default_factory=lambda: [1.0, 0.64, 0.4096, 0.262144, 0.16777216]
    )
    _velocity: List[float] = field(
        default_factory=lambda: [1.0, 0.8, 0.64, 0.512, 0.4096]
    )
    _acceleration: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0]
    )

    @property
    def position(self) -> List[float]:
        return self._position[: self._working_curriculum_step + 1]

    @property
    def velocity(self) -> List[float]:
        return self._velocity[: self._working_curriculum_step + 1]

    @property
    def acceleration(self) -> List[float]:
        return self._acceleration[: self._working_curriculum_step + 1]


class CheckResult(enum.Enum):
    TERMINAL_CONTACT = "SUCCESS: Touched platform"
    TERMINAL_SUCCESS = "SUCCESS: Goal state reached"
    TERMINAL_FLYZONE_X = "FAILURE: Drone moved too far from platform in x direction"
    TERMINAL_FLYZONE_Y = "FAILURE: Drone moved too far from platform in y direction"
    TERMINAL_FLYZONE_Z = "FAILURE: Drone moved too far from platform in z direction"
    TERMINAL_MINIMUM_ALTITUDE = "FAILURE: Reached minimum altitude"
    TERMINAL_TIMEOUT = "FAILURE: Maximum episode duration"
    NON_TERMINAL_SUCCESS = enum.auto()
    NON_TERMINAL = enum.auto()


class AbstractMdp(ABC):
    current_shaping_value = RewardShapingValue()
    previous_shaping_value = RewardShapingValue()
    info: Dict[str, Any] = {}
    _step_count = 0
    _check_result = CheckResult.NON_TERMINAL

    def __init__(
        self,
        working_curriculum_step: int,
        f_ag: float,
        t_max: int,
        p_max: float = 4.5,
        *,
        w_p: float = -100.0,
        w_v: float = -10.0,
        w_theta: float = -1.55,
        w_dur: float = -6.0,
        w_fail: float = -2.6,
        w_succ: float = 2.6,
        n_theta: int = 3,
        v_max: float = 3.39411,
        a_max: float = 1.28,
        theta_max: float = np.deg2rad(21.37723),
        delta_theta: float = np.deg2rad(7.12574),
        beta: float = 1 / 3,
        sigma_a: float = 0.416,
        minimum_altitude: float = 0.1,
    ) -> None:
        self._working_curriculum_step = working_curriculum_step
        self._f_ag = f_ag
        self._t_max = t_max
        self._p_max = p_max
        self._flyzone_x = (-p_max, p_max)
        self._flyzone_y = (-p_max, p_max)
        self._flyzone_z = (0.0, p_max)

        self._w_p = w_p
        """Table 3: Relative position reward factor"""
        self._w_v = w_v
        """Table 3: Relative velociy reward factor"""
        self._w_theta = w_theta
        """Table 3: Relative theta reward factor"""
        self._w_dur = w_dur
        """Table 3: Negative factor to incentivize the drone to move"""
        self._w_fail = w_fail
        """Table 3: Negative factor for failing an episode"""
        self._w_succ = w_succ
        """Table 3: Positive factor for sucessfuly completing an episode"""
        self._n_theta = n_theta
        """Table 3: Number of actions in which the action space will be discretized"""
        self._theta_max = theta_max
        """Section 3.5: Determined using formula 44"""
        self._delta_theta = delta_theta
        """Section 3.3.3: Determined using formula 3"""
        self._v_max = v_max
        """Section 5.2: Maximum velocity for the drone"""
        self._a_max = a_max
        """Section 5.2: Maximum acceleration for the drone"""
        self._beta = beta
        """Section 3.4: Contraction term for the i-th curriculum step"""
        self._sigma_a = sigma_a
        """Section 3.4: Contraction term for the acceleration i-th curriculum step"""

        self._minimum_altitude = minimum_altitude
        self._discrete_angles = np.linspace(-theta_max, theta_max, (n_theta * 2) + 1)
        self._limits = Limits(working_curriculum_step)
        self._delta_t = 1 / self._f_ag

    def _latest_valid_curriculum_step_for_state(
        self,
        state_limits: "list[float]",
        value: float,
    ):
        for idx in range(1, len(state_limits)):
            limit = state_limits[idx]
            if value < -limit or value > limit:
                return idx - 1
        return len(state_limits) - 1

    def _discretiazion_function(
        self, continuous_value: float, goal: float, limit: float
    ):
        if -limit <= continuous_value < -goal:
            return 0
        elif -goal <= continuous_value <= goal:
            return 1
        elif continuous_value <= limit:
            return 2
        else:
            raise ValueError(f"Unexpected discretization case: {continuous_value}")

    @abstractmethod
    def discrete_state(
        self, current_continuous_observation: ContinuousObservation
    ) -> Union[
        Tuple[int, int, int, int, int],
        Tuple[
            Tuple[int, int, int, int, int],
            Tuple[int, int, int, int, int],
        ],
    ]: ...
    @abstractmethod
    def continuous_action(
        self,
        action_x: int,
        action_y: int,
    ) -> Action: ...

    @abstractmethod
    def check(
        self,
    ) -> Dict[str, Any]: ...

    @abstractmethod
    def reset(self):
        self._current_shaping_value = RewardShapingValue()
        self._previous_shaping_value = RewardShapingValue()
        self._info: Dict[str, Any] = {}
        self._step_count = 0
        self._check_result = CheckResult.NON_TERMINAL

    def reward(self) -> float:
        return 0.0


class TrainingMdp(AbstractMdp):
    _current_continuous_observation = ContinuousObservation()
    _current_discrete_state: Optional[Tuple[int, int, int, int, int]] = None
    _previous_discrete_state: Optional[Tuple[int, int, int, int, int]] = None
    _curriculum_check = 0
    _cumulative_reward = 0
    _current_continuous_action = Action(pitch=0, roll=0, yaw=0, v_z=-0.1)

    def __init__(
        self,
        working_curriculum_step: int,
        f_ag: float,
        t_max: int,
        p_max: float = 4.5,
        *,
        w_p: float = -100,
        w_v: float = -10,
        w_theta: float = -1.55,
        w_dur: float = -6,
        w_fail: float = -2.6,
        w_succ: float = 2.6,
        n_theta: int = 3,
        v_max: float = 3.39411,
        a_max: float = 1.28,
        theta_max: float = np.deg2rad(21.37723),
        delta_theta: float = np.deg2rad(7.12574),
        beta: float = 1 / 3,
        sigma_a: float = 0.416,
        minimum_altitude: float = 0.2,
    ) -> None:
        super().__init__(
            working_curriculum_step,
            f_ag,
            t_max,
            w_p=w_p,
            w_v=w_v,
            w_theta=w_theta,
            w_dur=w_dur,
            w_fail=w_fail,
            w_succ=w_succ,
            n_theta=n_theta,
            p_max=p_max,
            v_max=v_max,
            a_max=a_max,
            theta_max=theta_max,
            delta_theta=delta_theta,
            beta=beta,
            sigma_a=sigma_a,
            minimum_altitude=minimum_altitude,
        )

    def discrete_state(
        self,
        current_continuous_observation: ContinuousObservation,
    ) -> Tuple[int, int, int, int, int]:
        self._previous_discrete_state = self._current_discrete_state
        self._current_continuous_observation = current_continuous_observation
        continuous_position = np.clip(
            current_continuous_observation.rel_p_x / self._p_max, -1, 1
        )
        continuous_velocity = np.clip(
            current_continuous_observation.rel_v_x / self._v_max, -1, 1
        )
        continuous_acceleration = np.clip(
            current_continuous_observation.rel_a_x / self._a_max, -1, 1
        )
        latest_valid_curriculum_step = min(
            [
                self._latest_valid_curriculum_step_for_state(
                    self._limits.position, continuous_position
                ),
                self._latest_valid_curriculum_step_for_state(
                    self._limits.velocity, continuous_velocity
                ),
                self._latest_valid_curriculum_step_for_state(
                    self._limits.acceleration, continuous_acceleration
                ),
            ]
        )
        position_contraction = self._beta
        if latest_valid_curriculum_step < self._working_curriculum_step:
            position_contraction = (
                self._limits.position[latest_valid_curriculum_step + 1]
                / self._limits.position[latest_valid_curriculum_step]
            )
        discrete_position = self._discretiazion_function(
            continuous_position,
            self._limits.position[latest_valid_curriculum_step] * position_contraction,
            self._limits.position[latest_valid_curriculum_step],
        )

        velocity_contraction = self._beta
        if latest_valid_curriculum_step < self._working_curriculum_step:
            velocity_contraction = (
                self._limits.velocity[latest_valid_curriculum_step + 1]
                / self._limits.velocity[latest_valid_curriculum_step]
            )
        discrete_velocity = self._discretiazion_function(
            continuous_velocity,
            self._limits.velocity[latest_valid_curriculum_step] * velocity_contraction,
            self._limits.velocity[latest_valid_curriculum_step],
        )

        acceleration_contraction = self._sigma_a
        if latest_valid_curriculum_step == self._working_curriculum_step:
            acceleration_contraction *= self._beta
        discrete_acceleration = self._discretiazion_function(
            continuous_acceleration,
            self._limits.acceleration[latest_valid_curriculum_step]
            * acceleration_contraction,
            self._limits.acceleration[latest_valid_curriculum_step],
        )
        clipped_pitch = np.clip(
            self._current_continuous_observation.pitch,
            -self._theta_max,
            self._theta_max,
        )
        discrete_angle = np.argmin(np.abs(self._discrete_angles - clipped_pitch))

        self._current_discrete_state = (
            int(latest_valid_curriculum_step),
            int(discrete_position),
            int(discrete_velocity),
            int(discrete_acceleration),
            int(discrete_angle),
        )

        return self._current_discrete_state

    def check(self):
        # Section 3.3.6, Section 3.4
        # "During the training of the different curriculum steps the fol
        # lowing episodic terminal criteria have been applied. On the
        # one hand, the episode terminates with success and the suc-
        # cess reward rsuc is received if the goal state s ∗ of the latest
        # curriculum step is reached if the agent has been in that cur-
        # riculum step’s discrete states for at least one second without
        # interruption. This is different to all previous curriculum steps
        # where the success reward rsuc is received immediately after
        # reaching the goal state of the respective curriculum step"
        # Hinting that we have to differentiate between:
        #   - Goal condition eached non terminal.
        #   - Goal condition reached terminal.
        #   - Success contact
        #   - Failure
        # Check that you are in limits currently
        if not self._current_discrete_state:
            raise ValueError(
                "Cannot check an empty state\n"
                + "You must call `discrete_state` before calling check."
            )

        # update other variables to perform checks
        self._step_count += 1
        # I guess touch contact has priority over everything
        # A landing trial is considered successful
        # if the UAV touches down
        if self._current_continuous_observation.contact:
            self._check_result = CheckResult.TERMINAL_CONTACT
        elif (
            self._current_continuous_observation.rel_p_x < self._flyzone_x[0]
            or self._current_continuous_observation.rel_p_x > self._flyzone_x[1]
        ):
            self._check_result = CheckResult.TERMINAL_FLYZONE_X
            self._info["Relative x"] = (
                f"{self._current_continuous_observation.rel_p_x=}"
            )
            self._info["Fly zone x"] = f"{self._flyzone_x=}"
        elif (
            self._current_continuous_observation.rel_p_y < self._flyzone_y[0]
            or self._current_continuous_observation.rel_p_y > self._flyzone_y[1]
        ):
            self._check_result = CheckResult.TERMINAL_FLYZONE_Y
            self._info["Relative y"] = (
                f"{self._current_continuous_observation.rel_p_y=}"
            )
            self._info["Fly zone y"] = f"{self._flyzone_y=}"
        elif self._current_continuous_observation.abs_p_z < self._minimum_altitude:
            self._check_result = CheckResult.TERMINAL_MINIMUM_ALTITUDE
            self._info["Relative z"] = (
                f"{self._current_continuous_observation.abs_p_z=}"
            )
            self._info["Fly zone z"] = f"{self._flyzone_z=}"
        elif self._current_continuous_observation.abs_p_z > self._flyzone_z[1]:
            self._check_result = CheckResult.TERMINAL_FLYZONE_Z
            self._info["Relative z"] = (
                f"{self._current_continuous_observation.rel_p_y=}"
            )
            self._info["Fly zone z"] = f"{self._flyzone_y}"
        elif self._step_count >= (self._t_max * self._f_ag):
            self._check_result = CheckResult.TERMINAL_TIMEOUT
            self._info["Timeout"] = f"{self._t_max * self._f_ag =}"

        # WARN: The edge case for the first state is purposefully ignored
        # due to the check being already super verbose.
        # Goal state reached, this also must be the last for how it is defined
        elif (
            self._previous_discrete_state
            # If we can map the previou curriculum step to the current
            and self._current_discrete_state[1] == 1
            and self._current_discrete_state[2] == 1
        ):
            if (
                # You are actually at the correct curriculum step resolution level
                self._previous_discrete_state[0] == self._working_curriculum_step
                and self._current_discrete_state[0] == self._working_curriculum_step
                # And you've been consistent
            ):
                self._curriculum_check += 1
                if self._curriculum_check >= self._f_ag:
                    # If you are consistent for a whole second, then a terminal success is reached
                    self._check_result = CheckResult.TERMINAL_SUCCESS
                else:
                    self._check_result = CheckResult.NON_TERMINAL_SUCCESS
            else:
                # We lose all the progress made for
                # terminal success
                self._curriculum_check = 0
                # However you still get a reward for being in goal
                # self._check_result = CheckResult.NON_TERMINAL_SUCCESS
        if (
            self._check_result == CheckResult.TERMINAL_CONTACT
            or self._check_result == CheckResult.TERMINAL_SUCCESS
            or self._check_result == CheckResult.TERMINAL_FLYZONE_X
            or self._check_result == CheckResult.TERMINAL_FLYZONE_Y
            or self._check_result == CheckResult.TERMINAL_FLYZONE_Z
            or self._check_result == CheckResult.TERMINAL_MINIMUM_ALTITUDE
            or self._check_result == CheckResult.TERMINAL_TIMEOUT
        ):
            self._info["Termination condition"] = self._check_result.value
            self._info["Number of steps"] = self._step_count
            self._info["Cumulative reward"] = self._cumulative_reward
            self._info["Mean reward"] = self._cumulative_reward / self._step_count
        return self._info

    def reward(self) -> float:
        if not self._previous_discrete_state:
            raise ValueError(
                "Previous state missing.\n"
                + "You must call `reset` and `discrete_state`"
                + "and then `step`before calling check."
            )
        if not self._current_discrete_state:
            raise ValueError(
                "Cannot check an empty state.\n"
                + "You must call `discrete_state` before calling check."
            )
        continuous_position = self._current_continuous_observation.rel_p_x
        continuous_velocity = self._current_continuous_observation.rel_v_x

        # Normalize
        normalized_continuous_position = np.clip(
            continuous_position / self._p_max, -1, 1
        )
        normalized_continuous_velocity = np.clip(
            continuous_velocity / self._v_max, -1, 1
        )

        normalized_pitch = self._current_continuous_action.pitch / self._theta_max

        # We need to _shape_ the reward based on the curiculum step that we ae working on.
        current_working_curriculum_step = self._current_discrete_state[0]
        # Update curiculum shaping terms
        self.previous_shaping_value = replace(self.current_shaping_value)
        self.current_shaping_value = RewardShapingValue(
            self._w_p * np.abs(normalized_continuous_position),
            self._w_v * np.abs(normalized_continuous_velocity),
            self._w_theta * np.abs(normalized_pitch),
        )
        # Eq 24
        r_p_max = (
            np.abs(self._w_p)
            * self._limits.velocity[current_working_curriculum_step]
            * self._delta_t
        )
        # Eq 25
        r_v_max = (
            np.abs(self._w_v)
            * self._limits.acceleration[current_working_curriculum_step]
            * self._delta_t
        )
        # Eq 26
        r_theta_max = (
            np.abs(self._w_theta)
            * (self._delta_theta / self._theta_max)
            * self._limits.velocity[current_working_curriculum_step]
        )

        # Eq 27
        r_dur_max = (
            self._w_dur
            * self._limits.velocity[current_working_curriculum_step]
            * self._delta_t
        )
        # Eq 28
        r_max = r_p_max + r_v_max + r_theta_max + r_dur_max

        # Kinda equation 20
        r_p = np.clip(
            self.current_shaping_value.position - self.previous_shaping_value.position,
            -r_p_max,
            r_p_max,
        )
        r_v = np.clip(
            self.current_shaping_value.velocity - self.previous_shaping_value.velocity,
            -r_v_max,
            r_v_max,
        )
        r_theta = (
            self._w_theta
            * (
                np.abs(self.current_shaping_value.angle)
                - np.abs(self.previous_shaping_value.angle)
            )
            / self._theta_max
            * self._limits.velocity[current_working_curriculum_step]
        )
        r_dur = (
            self._w_dur
            * self._limits.velocity[current_working_curriculum_step]
            * self._delta_t
        )
        if self._check_result == CheckResult.NON_TERMINAL:
            r_term = 0.0
        if (
            self._check_result == CheckResult.NON_TERMINAL_SUCCESS
            or self._check_result == CheckResult.TERMINAL_SUCCESS
        ):
            r_term = self._w_succ * r_max
        else:
            r_term = self._w_fail * r_max

        r_t = r_p + r_v + r_theta + r_dur + r_term
        # log
        self._cumulative_reward += r_t
        return float(r_t)

    def continuous_action(self, action_x: int, action_y: int = 2):
        if action_y != 2:
            raise ValueError("Cannot move in the y direction while training")
        if action_x == 0:  # Increase
            self._current_continuous_action.pitch = min(
                (
                    self._current_continuous_action.pitch + self._delta_theta,
                    self._theta_max,
                )
            )
        elif action_x == 1:  # Decrease
            self._current_continuous_action.pitch = max(
                (
                    self._current_continuous_action.pitch - self._delta_theta,
                    -self._theta_max,
                )
            )
        return self._current_continuous_action

    def reset(self):
        super().reset()
        self._current_continuous_observation = ContinuousObservation()
        self._current_discrete_state: Optional[Tuple[int, int, int, int, int]] = None
        self._previous_discrete_state: Optional[Tuple[int, int, int, int, int]] = None
        self._curriculum_check = 0
        self._cumulative_reward = 0
        self._current_continuous_action = Action(pitch=0, roll=0, yaw=0, v_z=-0.1)


class SimulationMdp(AbstractMdp):
    _current_continuous_observation = ContinuousObservation()

    _current_discrete_state_x: Optional[Tuple[int, int, int, int, int]] = None
    _previous_discrete_state_x: Optional[Tuple[int, int, int, int, int]] = None

    _current_discrete_state_y: Optional[Tuple[int, int, int, int, int]] = None
    _previous_discrete_state_y: Optional[Tuple[int, int, int, int, int]] = None
    _current_continuous_action = Action(pitch=0, roll=0, yaw=np.pi / 4, v_z=-0.1)

    def __init__(
        self,
        working_curriculum_step: int,
        f_ag: float,
        t_max: int,
        *,
        w_p: float = -100,
        w_v: float = -10,
        w_theta: float = -1.55,
        w_dur: float = -6,
        w_fail: float = -2.6,
        w_succ: float = 2.6,
        n_theta: int = 3,
        p_max: float = 4.5,
        v_max: float = 3.39411,
        a_max: float = 1.28,
        theta_max: float = np.deg2rad(21.37723),
        delta_theta: float = np.deg2rad(7.12574),
        beta: float = 1 / 3,
        sigma_a: float = 0.416,
        minimum_altitude: float = 0.0,
    ) -> None:
        super().__init__(
            working_curriculum_step,
            f_ag,
            t_max,
            w_p=w_p,
            w_v=w_v,
            w_theta=w_theta,
            w_dur=w_dur,
            w_fail=w_fail,
            w_succ=w_succ,
            n_theta=n_theta,
            p_max=p_max,
            v_max=v_max,
            a_max=a_max,
            theta_max=theta_max,
            delta_theta=delta_theta,
            beta=beta,
            sigma_a=sigma_a,
            minimum_altitude=minimum_altitude,
        )

    def discrete_state(
        self,
        current_continuous_observation: ContinuousObservation,
    ) -> Tuple[Tuple[int, int, int, int, int], Tuple[int, int, int, int, int]]:
        self._previous_discrete_state_x = self._current_discrete_state_x
        self._previous_discrete_state_y = self._current_discrete_state_y
        self._current_continuous_observation = current_continuous_observation
        return self.discrete_state_x(), self.discrete_state_y()

    def discrete_state_x(
        self,
    ) -> Tuple[int, int, int, int, int]:
        continuous_position = np.clip(
            self._current_continuous_observation.rel_p_x / self._p_max, -1, 1
        )
        continuous_velocity = np.clip(
            self._current_continuous_observation.rel_v_x / self._v_max, -1, 1
        )
        continuous_acceleration = np.clip(
            self._current_continuous_observation.rel_a_x / self._a_max, -1, 1
        )
        latest_valid_curriculum_step = min(
            [
                self._latest_valid_curriculum_step_for_state(
                    self._limits.position, continuous_position
                ),
                self._latest_valid_curriculum_step_for_state(
                    self._limits.velocity, continuous_velocity
                ),
                self._latest_valid_curriculum_step_for_state(
                    self._limits.acceleration, continuous_acceleration
                ),
            ]
        )
        position_contraction = self._beta
        if latest_valid_curriculum_step < self._working_curriculum_step:
            position_contraction = (
                self._limits.position[latest_valid_curriculum_step + 1]
                / self._limits.position[latest_valid_curriculum_step]
            )
        discrete_position = self._discretiazion_function(
            continuous_position,
            self._limits.position[latest_valid_curriculum_step] * position_contraction,
            self._limits.position[latest_valid_curriculum_step],
        )

        velocity_contraction = self._beta
        if latest_valid_curriculum_step < self._working_curriculum_step:
            velocity_contraction = (
                self._limits.velocity[latest_valid_curriculum_step + 1]
                / self._limits.velocity[latest_valid_curriculum_step]
            )
        discrete_velocity = self._discretiazion_function(
            continuous_velocity,
            self._limits.velocity[latest_valid_curriculum_step] * velocity_contraction,
            self._limits.velocity[latest_valid_curriculum_step],
        )

        acceleration_contraction = self._sigma_a
        if latest_valid_curriculum_step == self._working_curriculum_step:
            acceleration_contraction *= self._beta
        discrete_acceleration = self._discretiazion_function(
            continuous_acceleration,
            self._limits.acceleration[latest_valid_curriculum_step]
            * acceleration_contraction,
            self._limits.acceleration[latest_valid_curriculum_step],
        )
        clipped_pitch = np.clip(
            self._current_continuous_observation.pitch,
            -self._theta_max,
            self._theta_max,
        )
        discrete_angle = np.argmin(np.abs(self._discrete_angles - clipped_pitch))

        self._current_discrete_state_x = (
            int(latest_valid_curriculum_step),
            int(discrete_position),
            int(discrete_velocity),
            int(discrete_acceleration),
            int(discrete_angle),
        )

        return self._current_discrete_state_x

    def discrete_state_y(
        self,
    ) -> Tuple[int, int, int, int, int]:
        continuous_position = np.clip(
            self._current_continuous_observation.rel_p_y / self._p_max, -1, 1
        )
        continuous_velocity = np.clip(
            self._current_continuous_observation.rel_v_y / self._v_max, -1, 1
        )
        continuous_acceleration = np.clip(
            self._current_continuous_observation.rel_a_y / self._a_max, -1, 1
        )
        latest_valid_curriculum_step = min(
            [
                self._latest_valid_curriculum_step_for_state(
                    self._limits.position, continuous_position
                ),
                self._latest_valid_curriculum_step_for_state(
                    self._limits.velocity, continuous_velocity
                ),
                self._latest_valid_curriculum_step_for_state(
                    self._limits.acceleration, continuous_acceleration
                ),
            ]
        )
        position_contraction = self._beta
        if latest_valid_curriculum_step < self._working_curriculum_step:
            position_contraction = (
                self._limits.position[latest_valid_curriculum_step + 1]
                / self._limits.position[latest_valid_curriculum_step]
            )
        discrete_position = self._discretiazion_function(
            continuous_position,
            self._limits.position[latest_valid_curriculum_step] * position_contraction,
            self._limits.position[latest_valid_curriculum_step],
        )

        velocity_contraction = self._beta
        if latest_valid_curriculum_step < self._working_curriculum_step:
            velocity_contraction = (
                self._limits.velocity[latest_valid_curriculum_step + 1]
                / self._limits.velocity[latest_valid_curriculum_step]
            )
        discrete_velocity = self._discretiazion_function(
            continuous_velocity,
            self._limits.velocity[latest_valid_curriculum_step] * velocity_contraction,
            self._limits.velocity[latest_valid_curriculum_step],
        )

        acceleration_contraction = self._sigma_a
        if latest_valid_curriculum_step == self._working_curriculum_step:
            acceleration_contraction *= self._beta
        discrete_acceleration = self._discretiazion_function(
            continuous_acceleration,
            self._limits.acceleration[latest_valid_curriculum_step]
            * acceleration_contraction,
            self._limits.acceleration[latest_valid_curriculum_step],
        )
        clipped_roll = np.clip(
            self._current_continuous_observation.roll,
            -self._theta_max,
            self._theta_max,
        )
        discrete_angle = np.argmin(np.abs(self._discrete_angles - clipped_roll))

        self._current_discrete_state_y = (
            int(latest_valid_curriculum_step),
            int(discrete_position),
            int(discrete_velocity),
            int(discrete_acceleration),
            int(discrete_angle),
        )

        return self._current_discrete_state_y

    def check(self):
        if not self._current_discrete_state_x or not self._current_discrete_state_y:
            raise ValueError(
                "Cannot check an empty state\n"
                + "You must call `discrete_state` before calling check."
            )

        # update other variables to perform checks
        self._step_count += 1
        # I guess touch contact has priority over everything
        if self._current_continuous_observation.contact:
            self._check_result = CheckResult.TERMINAL_CONTACT
        # Section 3.3.6
        # Discussed briefly when explaining the rewards.
        elif (
            self._current_continuous_observation.rel_p_x < self._flyzone_x[0]
            or self._current_continuous_observation.rel_p_x > self._flyzone_x[1]
        ):
            self._check_result = CheckResult.TERMINAL_FLYZONE_X
            self._info["Relative x"] = (
                f"{self._current_continuous_observation.rel_p_x=}"
            )
            self._info["Fly zone x"] = f"{self._flyzone_x=}"

        elif (
            self._current_continuous_observation.rel_p_y < self._flyzone_y[0]
            or self._current_continuous_observation.rel_p_y > self._flyzone_y[1]
        ):
            self._check_result = CheckResult.TERMINAL_FLYZONE_Y
            self._info["Relative y"] = (
                f"{self._current_continuous_observation.rel_p_y=}"
            )
            self._info["Fly zone y"] = f"{self._flyzone_y=}"
        elif self._current_continuous_observation.abs_p_z < self._minimum_altitude:
            self._check_result = CheckResult.TERMINAL_MINIMUM_ALTITUDE
            self._info["Relative z"] = (
                f"{self._current_continuous_observation.abs_p_z=}"
            )
            self._info["Fly zone z"] = f"{self._flyzone_z=}"

        elif self._current_continuous_observation.abs_p_z > self._flyzone_z[1]:
            self._check_result = CheckResult.TERMINAL_FLYZONE_Z
            self._info["Relative z"] = (
                f"{self._current_continuous_observation.rel_p_y=}"
            )
            self._info["Fly zone z"] = f"{self._flyzone_y}"

        elif self._step_count >= (self._t_max * self._f_ag):
            self._check_result = CheckResult.TERMINAL_TIMEOUT
            self._info["Timeout"] = f"{self._t_max * self._f_ag =}"

        if (
            self._check_result == CheckResult.TERMINAL_CONTACT
            or self._check_result == CheckResult.TERMINAL_FLYZONE_X
            or self._check_result == CheckResult.TERMINAL_FLYZONE_Y
            or self._check_result == CheckResult.TERMINAL_FLYZONE_Z
            or self._check_result == CheckResult.TERMINAL_MINIMUM_ALTITUDE
            or self._check_result == CheckResult.TERMINAL_TIMEOUT
        ):
            self._info["Termination condition"] = self._check_result.value
            self._info["Number of steps"] = self._step_count
        return self._info

    def continuous_action(self, action_x: int, action_y: int):
        if action_x == 0:  # Increase
            self._current_continuous_action.pitch = min(
                (
                    self._current_continuous_action.pitch + self._delta_theta,
                    self._theta_max,
                )
            )
        elif action_x == 1:  # Decrease
            self._current_continuous_action.pitch = max(
                (
                    self._current_continuous_action.pitch - self._delta_theta,
                    -self._theta_max,
                )
            )
        if action_y == 0:  # Increase
            self._current_continuous_action.roll = min(
                (
                    self._current_continuous_action.roll + self._delta_theta,
                    self._theta_max,
                )
            )
        elif action_y == 1:  # Decrease
            self._current_continuous_action.roll = max(
                (
                    self._current_continuous_action.roll - self._delta_theta,
                    -self._theta_max,
                )
            )
        return self._current_continuous_action

    def reset(self):
        super().reset()
        self._current_continuous_observation = ContinuousObservation()
        self._current_discrete_state_x: Optional[Tuple[int, int, int, int, int]] = None
        self._previous_discrete_state_x: Optional[Tuple[int, int, int, int, int]] = None
        self._current_discrete_state_y: Optional[Tuple[int, int, int, int, int]] = None
        self._previous_discrete_state_y: Optional[Tuple[int, int, int, int, int]] = None
        self._current_continuous_action = Action(
            pitch=0, roll=0, yaw=np.pi / 4, v_z=-0.1
        )
