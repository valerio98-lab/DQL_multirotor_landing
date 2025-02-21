#! /usr/bin/python3
"""
Script contains the definition of the class defining the training environment and some of its interfaces to other ros nodes.
Furthermore, it registers the landing scenario as an environment in gym.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import gym  # type: ignore
import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from gym.envs.registration import register  # type: ignore
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion

from dql_multirotor_landing.mdp import AbstractMdp, ContinuousObservation, TrainingMdp
from dql_multirotor_landing.msg import Action, Observation
from dql_multirotor_landing.utils import get_publisher  # type: ignore


class AbstractLandingEnv(gym.Env, ABC):
    observation: Observation = Observation()
    cumulative_reward: float = 0.0
    number_of_steps: int = 0
    mdp: AbstractMdp

    def __init__(
        self,
        f_ag: float = 22.92,
        t_max: int = 40,
        flyzone_x: Tuple[float, float] = (-4.5, 4.5),
        flyzone_y: Tuple[float, float] = (-4.5, 4.5),
        flyzone_z: Tuple[float, float] = (0, 9),
        z_init: float = 4.0,
        *,
        initial_curriculum_step: int = 0,
    ):
        # region Ros Boilerplate
        rospy.init_node("landing_simulation_gym_node")
        # Setup publishers
        self.action_to_interface_publisher = get_publisher(
            "training_action_interface/action_to_interface", Action, queue_size=0
        )
        self.reset_simulation_publisher = get_publisher(
            "training/reset_simulation", Bool, queue_size=0
        )
        # Setup subscribers
        self.observation_continuous_subscriber = rospy.Subscriber(
            "/hummingbird/training_observation_interface/observations",
            Observation,
            self.read_training_continuous_observations,
        )

        # Set up services
        rospy.wait_for_service("/gazebo/reset_world")
        self.reset_world_gazebo_service = rospy.ServiceProxy(
            "/gazebo/reset_world", Empty
        )
        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_model_state_service = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )
        rospy.wait_for_service("/gazebo/pause_physics")
        self.pause_sim = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.unpause_sim = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        rospy.wait_for_service("/gazebo/get_model_state")
        self.model_coordinates = rospy.ServiceProxy(
            "/gazebo/get_model_state", GetModelState
        )
        # endregion
        # Other variables needed during execution
        self.current_curriculum_step = initial_curriculum_step
        self.f_ag = f_ag
        self.t_max = t_max
        self.flyzone_x = flyzone_x
        self.flyzone_y = flyzone_y
        self.flyzone_z = flyzone_z
        self.z_init = z_init

    def read_training_continuous_observations(self, observation: Observation):
        self.observation = observation

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
    def set_curriculum_step(self, curriculum_step: int): ...

    @abstractmethod
    def reset(
        self,
    ) -> Union[
        Tuple[int, int, int, int, int],
        Union[Tuple[int, int, int, int, int], Tuple[int, int, int, int, int]],
    ]: ...

    @abstractmethod
    def step(
        self, action: Union[int, Tuple[int, int]]
    ) -> Union[
        Tuple[int, int, int, int, int],
        Union[Tuple[int, int, int, int, int], Tuple[int, int, int, int, int]],
    ]: ...


class TrainingLandingEnv(AbstractLandingEnv):
    def __init__(
        self,
        f_ag: float = 22.92,
        t_max: int = 40,
        flyzone_x: Tuple[float, float] = (-4.5, 4.5),
        flyzone_y: Tuple[float, float] = (-4.5, 4.5),
        flyzone_z: Tuple[float, float] = (0, 9),
        z_init: float = 4,
        *,
        initial_curriculum_step: int = 0,
    ):
        super().__init__(
            f_ag,
            t_max,
            flyzone_x,
            flyzone_y,
            flyzone_z,
            z_init,
            initial_curriculum_step=initial_curriculum_step,
        )
        self.mdp = TrainingMdp(
            initial_curriculum_step,
            self.f_ag,
            self.t_max,
            self.flyzone_x,
            self.flyzone_y,
            self.flyzone_z,
        )

    def set_curriculum_step(self, curriculum_step: int):
        self.current_curriculum_step = curriculum_step
        # TODO: Maybe implement set_curriculum_step
        self.mdp = TrainingMdp(
            curriculum_step,
            self.f_ag,
            self.t_max,
            self.flyzone_x,
            self.flyzone_y,
            self.flyzone_z,
        )

    def reset(self) -> Tuple[int, int, int, int, int]:
        # Reset the setpoints for the low-level controllers of the copter
        self.mdp.reset()

        # Pause simulation
        self.pause_sim()
        moving_platform = self.model_coordinates("moving_platform", "world")
        # Initialize drone state
        initial_drone = ModelState()
        initial_drone.model_name = "hummingbird"
        # Section 4.3 Initializazion
        # "we use the following normal distribution to determine the UAV’s
        # initial position within the fly zone during
        # the first curriculum step"
        if self.current_curriculum_step == 0:
            # This value doens't seem to be discussed directly in the paper.
            # However hey do say (still in Section 4.3):
            # "UAV is initialized close to the center of the
            # flyzone and thus in proximity to the moving
            # platform more frequently."
            # Leading to thing that this choice of mu is coherent
            mu = 0
            sigma = self.mdp.p_max / 3
            x_init = np.random.normal(mu, sigma)
        else:
            x_init = np.random.uniform(self.flyzone_x[0], self.flyzone_x[1])

        # Clip to stay within fly zone
        initial_drone.pose.position.x = np.clip(
            x_init + moving_platform.pose.position.x,
            moving_platform.pose.position.x + self.flyzone_x[0],
            moving_platform.pose.position.x + self.flyzone_x[1],
        )
        initial_drone.pose.position.y = 0.0
        initial_drone.pose.position.z = self.z_init
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

        self.set_model_state_service(initial_drone)

        # Reset episode counters and logging variables
        self.step_number_in_episode = 0
        self.episode_reward = 0
        self.done_numeric = 0

        self.reset_simulation_publisher.publish(Bool(True))
        # Let simulation update values
        self.unpause_sim()
        rospy.sleep(1 / self.f_ag)
        self.pause_sim()
        # Save temporarily the current continuous state to avoid sinchronization issues
        # due to callbacks
        drone = self.model_coordinates("hummingbird", "world")
        roll, pitch, _ = euler_from_quaternion(
            quaternion=[
                drone.pose.orientation.x,
                drone.pose.orientation.y,
                drone.pose.orientation.z,
                drone.pose.orientation.w,
            ]
        )
        abs_v_z = drone.pose.position.z
        observation = self.observation
        observation_continuous = ContinuousObservation(
            observation, pitch, roll, abs_v_z
        )
        observation_x = self.mdp.discrete_state(
            observation_continuous,
        )
        return observation_x  # type: ignore

    def step(self, action_x: int, action_y: int = 2):  # type: ignore
        """Function performs one timestep of the training."""

        # Update the setpoints based on the current action and publish them to the ROS network

        continuous_action = self.mdp.continuous_action(action_x, action_y)

        self.action_to_interface_publisher.publish(continuous_action)

        # Let the simulation run for one RL timestep and allow to recieve obsevation
        self.unpause_sim()
        rospy.sleep(1 / self.f_ag)
        self.pause_sim()

        # Map the current observation to a discrete state value
        drone = self.model_coordinates("hummingbird", "world")
        roll, pitch, _ = euler_from_quaternion(
            quaternion=[
                drone.pose.orientation.x,
                drone.pose.orientation.y,
                drone.pose.orientation.z,
                drone.pose.orientation.w,
            ]
        )
        abs_v_z = drone.pose.position.z
        observation = self.observation
        observation_continuous = ContinuousObservation(
            observation, pitch, roll, abs_v_z
        )

        discrete_observation = self.mdp.discrete_state(observation_continuous)

        info = self.mdp.check()
        reward = self.mdp.reward()
        info["Curent reward"] = reward
        return (
            discrete_observation,
            reward,
            "Termination condition" in info.keys(),
            info,
        )


# Register the training environment in gym as an available one
register(
    id="landing_simulation-v0",
    entry_point="dql_multirotor_landing.landing_simulation_env:TrainingLandingEnv",
)
