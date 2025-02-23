#! /usr/bin/python3
"""
Definition and creation of simulation environment
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import gym  # type: ignore
import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from gym.envs.registration import register  # type: ignore
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion

from dql_multirotor_landing.mdp import ContinuousObservation, SimulationMdp, TrainingMdp
from dql_multirotor_landing.msg import Action, Observation
from dql_multirotor_landing.utils import get_publisher  # type: ignore


class AbstractLandingEnv(gym.Env, ABC):
    """Abstract class fo he environment."""

    _observation: Observation = Observation()
    """Current observation coming from the the `manager_node`"""

    def __init__(
        self,
        initial_curriculum_step: int = 0,
        *,
        t_max: int = 20,
        f_ag: float = 22.92,
        p_max: float = 4.5,
        z_init: float = 2.0,
    ):
        # Setup publishers
        self._action_publisher = get_publisher(
            "action_to_interface", Action, queue_size=0
        )
        """Publishes current action to perform to `manager_node`"""
        self._reset_publisher = get_publisher("reset_simulation", Bool, queue_size=0)
        """Publishes the end of the current episode to `manager_node`"""
        # Setup subscribers
        self._observation_subscriber = rospy.Subscriber(
            "/hummingbird/observations",
            Observation,
            self.read_training_continuous_observations,
        )
        """Subscribe to the current observation coming from the `manager_node`"""

        # Set up services
        rospy.wait_for_service("/gazebo/reset_world")
        self._reset_world_gazebo_service = rospy.ServiceProxy(
            "/gazebo/reset_world", Empty
        )
        """Service for resetting the world in gazebo"""
        rospy.wait_for_service("/gazebo/set_model_state")
        self._set_model_state_service = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )
        rospy.wait_for_service("/gazebo/pause_physics")
        self._pause_sim = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        """Service for pausing the simulation in gazebo"""

        rospy.wait_for_service("/gazebo/unpause_physics")
        self._unpause_sim = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        """Service for unpausing the simulation in gazebo"""

        rospy.wait_for_service("/gazebo/get_model_state")
        self._model_coordinates = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )
        """Service for getting model informations from gazebo"""

        self._t_max = t_max
        self._f_ag = f_ag
        self._z_init = z_init
        self._p_max = p_max
        self._flyzone_x = (-p_max, p_max)
        """Section 3.1.2: Maximum distance from the platform in the x direction"""
        self._flyzone_y = (-p_max, p_max)
        """Section 3.1.2: Maximum distance from the platform in the y direction"""
        self._flyzone_z = (0.0, p_max)
        """Section 3.1.2: Maximum distance from the platform in the y direction"""
        # Other variables needed during execution
        self._working_curriculum_step = initial_curriculum_step
        """Curriculum step in which we're working, it's zero indexed"""

    def close(self):
        self._model_coordinates.close()
        self._unpause_sim.close()
        self._pause_sim.close()
        self._set_model_state_service.close()
        self._reset_world_gazebo_service.close()
        self._observation_subscriber.unregister()
        self._action_publisher.unregister()
        self._reset_publisher.unregister()

    def read_training_continuous_observations(self, observation: Observation):
        self._observation = observation

    def get_robot_rotation(self, drone) -> Tuple[float, float, float]:
        roll, pitch, yaw = euler_from_quaternion(
            quaternion=[
                drone.pose.orientation.x,
                drone.pose.orientation.y,
                drone.pose.orientation.z,
                drone.pose.orientation.w,
            ]
        )
        return roll, pitch, yaw

    def get_robot_absolute_altitude(self, drone):
        abs_v_z = drone.pose.position.z
        return abs_v_z

    @abstractmethod
    def reset(
        self,
    ) -> Union[
        Tuple[int, int, int, int, int],
        Tuple[Tuple[int, int, int, int, int], Tuple[int, int, int, int, int]],
    ]: ...

    @abstractmethod
    def step(
        self, action: Union[int, Tuple[int, int]]
    ) -> Union[
        Tuple[Tuple[int, int, int, int, int], float, bool, Dict[str, Any]],
        Tuple[
            Tuple[int, int, int, int, int],
            Tuple[int, int, int, int, int],
            bool,
            Dict[str, Any],
        ],
    ]: ...


class TrainingLandingEnv(AbstractLandingEnv):
    def __init__(
        self,
        initial_curriculum_step: int = 0,
        *,
        t_max: int = 20,
        f_ag: float = 22.92,
        p_max: float = 4.5,
        z_init: float = 2.0,
    ):
        super().__init__(
            t_max=t_max,
            initial_curriculum_step=initial_curriculum_step,
            f_ag=f_ag,
            p_max=p_max,
            z_init=z_init,
        )
        self._mdp = TrainingMdp(
            initial_curriculum_step,
            self._f_ag,
            self._t_max,
            self._p_max,
        )
        """Underlying Markov decision process"""

    def reset(self) -> Tuple[int, int, int, int, int]:
        # Reset the undelying MDP state
        self._mdp.reset()

        # Pause simulation
        self._pause_sim()
        moving_platform = self._model_coordinates("moving_platform", "world")
        # Initialize drone state
        initial_drone = ModelState()
        initial_drone.model_name = "hummingbird"
        # Section 4.3 Initializazion
        # "we use the following normal distribution to determine the UAV’s
        # initial position within the fly zone during
        # the first curriculum step"
        if self._working_curriculum_step == 0:
            # This value doens't seem to be discussed directly in the paper.
            # However they do say (still in Section 4.3):
            # "UAV is initialized close to the center of the
            # flyzone and thus in proximity to the moving
            # platform more frequently."
            # Leading to thing that this choice of mu is coherent
            mu = 0
            sigma = self._p_max / 3
            x_init = np.random.normal(mu, sigma)
        else:
            # Not much it is said, but successive initial states are choosen uniformly
            # within he fly zone
            x_init = np.random.uniform(self._flyzone_x[0], self._flyzone_x[1])

        # Clip to stay within fly zone
        initial_drone.pose.position.x = np.clip(
            x_init + moving_platform.pose.position.x,
            moving_platform.pose.position.x + self._flyzone_x[0],
            moving_platform.pose.position.x + self._flyzone_x[1],
        )
        initial_drone.pose.position.y = 0.0
        initial_drone.pose.position.z = self._z_init
        # Section 3.12:
        # Each landing trial will begin with the UAV being in hover state,
        # leading to the following initial conditions for rotational movement
        initial_drone.twist.linear.x = 0
        initial_drone.twist.linear.y = 0
        initial_drone.twist.linear.z = 0
        initial_drone.twist.angular.x = 0
        initial_drone.twist.angular.y = 0
        initial_drone.twist.angular.z = 0
        initial_drone.pose.orientation.x = 0
        initial_drone.pose.orientation.y = 0
        initial_drone.pose.orientation.z = 0
        initial_drone.pose.orientation.w = 1.0

        self._set_model_state_service(initial_drone)

        self._reset_publisher.publish(Bool(True))
        # Let simulation update values
        self._unpause_sim()
        rospy.sleep(1 / self._f_ag)
        self._pause_sim()
        # Save temporarily the current continuous state to avoid sinchronization issues
        # due to callbacks
        drone = self._model_coordinates("hummingbird", "world")
        roll, pitch, _ = euler_from_quaternion(
            quaternion=[
                drone.pose.orientation.x,
                drone.pose.orientation.y,
                drone.pose.orientation.z,
                drone.pose.orientation.w,
            ]
        )
        abs_v_z = drone.pose.position.z
        observation_continuous = ContinuousObservation(
            self._observation, pitch, roll, abs_v_z
        )
        observation_x = self._mdp.discrete_state(
            observation_continuous,
        )
        return observation_x  # type: ignore

    def step(self, action_x: int, action_y: int = 2):  # type: ignore
        """Function performs one timestep of the training."""

        # Update the setpoints based on the current action and publish them to the ROS network
        continuous_action = self._mdp.continuous_action(action_x, action_y)
        self._action_publisher.publish(continuous_action)

        # Let the simulation run for one RL timestep and allow to recieve obsevation
        self._unpause_sim()
        rospy.sleep(1 / self._f_ag)
        self._pause_sim()

        # Map the current observation to a discrete state value
        drone = self._model_coordinates("hummingbird", "world")
        roll, pitch, _ = euler_from_quaternion(
            quaternion=[
                drone.pose.orientation.x,
                drone.pose.orientation.y,
                drone.pose.orientation.z,
                drone.pose.orientation.w,
            ]
        )
        abs_v_z = drone.pose.position.z
        observation_continuous = ContinuousObservation(
            self._observation, pitch, roll, abs_v_z
        )

        discrete_observation = self._mdp.discrete_state(observation_continuous)

        info = self._mdp.check()
        reward = self._mdp.reward()
        info["Current reward"] = reward
        return (
            discrete_observation,
            reward,
            "Termination condition" in info.keys(),
            info,
        )


class SimulationLandingEnv(AbstractLandingEnv):
    def __init__(
        self,
        initial_curriculum_step: int = 4,
        *,
        t_max: int = 20,
        f_ag: float = 22.92,
        p_max: float = 4.5,
        z_init: float = 4,
    ):
        super().__init__(
            t_max=t_max,
            initial_curriculum_step=initial_curriculum_step,
            f_ag=f_ag,
            p_max=p_max,
            z_init=z_init,
        )
        self._mdp = SimulationMdp(
            initial_curriculum_step,
            self._f_ag,
            self._t_max,
        )

    def reset(
        self,
    ) -> Tuple[Tuple[int, int, int, int, int], Tuple[int, int, int, int, int]]:
        # Reset the setpoints for the low-level controllers of the copter

        self._mdp.reset()

        # Pause simulation
        # self._reset_world_gazebo_service()
        self._pause_sim()
        moving_platform = self._model_coordinates("moving_platform", "world")
        # Initialize drone state
        initial_drone = ModelState()
        initial_drone.model_name = "hummingbird"
        # Section 4.3 Initializazion
        # "we use the following normal distribution to determine the UAV’s
        # initial position within the fly zone during
        # the first curriculum step"

        x_init = np.random.uniform(self._flyzone_x[0], self._flyzone_x[1])
        y_init = np.random.uniform(self._flyzone_y[0], self._flyzone_y[1])

        # Clip to stay within fly zone
        initial_drone.pose.position.x = np.clip(
            moving_platform.pose.position.x - x_init,
            self._flyzone_x[0],
            self._flyzone_x[1],
        )
        initial_drone.pose.position.y = 0 * np.clip(
            moving_platform.pose.position.y - y_init,
            self._flyzone_y[0],
            self._flyzone_y[1],
        )
        initial_drone.pose.position.z = self._z_init
        # Section 3.12:
        # Each landing trial will begin with the UAV being in hover state,
        # leading to the following initial conditions for rotational movement
        initial_drone.twist.linear.x = 0
        initial_drone.twist.linear.y = 0
        initial_drone.twist.linear.z = 0
        initial_drone.twist.angular.x = 0
        initial_drone.twist.angular.y = 0
        initial_drone.twist.angular.z = 0
        initial_drone.pose.orientation.x = 0
        initial_drone.pose.orientation.y = 0
        initial_drone.pose.orientation.z = 0
        initial_drone.pose.orientation.w = 1.0

        self._set_model_state_service(initial_drone)

        # Reset episode counters and logging variables
        self.step_number_in_episode = 0
        self.episode_reward = 0
        self.done_numeric = 0

        self._reset_publisher.publish(Bool(True))
        # Let simulation update values
        self._unpause_sim()
        rospy.sleep(1 / self._f_ag)
        self._pause_sim()
        # Save temporarily the current continuous state to avoid sinchronization issues
        # due to callbacks
        drone = self._model_coordinates("hummingbird", "world")
        roll, pitch, _ = euler_from_quaternion(
            quaternion=[
                drone.pose.orientation.x,
                drone.pose.orientation.y,
                drone.pose.orientation.z,
                drone.pose.orientation.w,
            ]
        )
        abs_v_z = drone.pose.position.z

        observation_continuous = ContinuousObservation(
            self._observation, pitch, roll, abs_v_z
        )
        observation_x, observation_y = self._mdp.discrete_state(
            observation_continuous,
        )
        return (observation_x, observation_y)

    def step(self, action_x: int, action_y: int):  # type: ignore
        """Function performs one timestep of the training."""

        # Update the setpoints based on the current action and publish them to the ROS network

        continuous_action = self._mdp.continuous_action(action_x, action_y)

        self._action_publisher.publish(continuous_action)

        # Let the simulation run for one RL timestep and allow to recieve obsevation
        self._unpause_sim()
        rospy.sleep(1 / self._f_ag)
        self._pause_sim()

        # Map the current observation to a discrete state value
        drone = self._model_coordinates("hummingbird", "world")
        roll, pitch, _ = euler_from_quaternion(
            quaternion=[
                drone.pose.orientation.x,
                drone.pose.orientation.y,
                drone.pose.orientation.z,
                drone.pose.orientation.w,
            ]
        )
        abs_v_z = drone.pose.position.z
        observation_continuous = ContinuousObservation(
            self._observation, pitch, roll, abs_v_z
        )

        discrete_observation_x, discrete_observation_y = self._mdp.discrete_state(
            observation_continuous
        )

        info = self._mdp.check()
        return (
            discrete_observation_x,
            discrete_observation_y,
            "Termination condition" in info.keys(),
            info,
        )


# Register the training environment in gym as an available one
register(
    id="Landing-Training-v0",
    entry_point="dql_multirotor_landing.landing_simulation_env:TrainingLandingEnv",
)
register(
    id="Landing-Simulation-v0",
    entry_point="dql_multirotor_landing.landing_simulation_env:SimulationLandingEnv",
)
