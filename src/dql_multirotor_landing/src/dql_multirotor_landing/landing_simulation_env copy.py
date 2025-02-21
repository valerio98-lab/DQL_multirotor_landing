#! /usr/bin/python3
"""
Script contains the definition of the class defining the training environment and some of its interfaces to other ros nodes.
Furthermore, it registers the landing scenario as an environment in gym.
"""

from typing import List, Literal, Optional, Tuple

import gym  # type: ignore
import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from gym.envs.registration import register  # type: ignore
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion

from dql_multirotor_landing.mdp import ContinuousObservation, Mdp
from dql_multirotor_landing.msg import Action, Observation
from dql_multirotor_landing.utils import get_publisher  # type: ignore


class LandingSimulationEnv(gym.Env):
    # The paper doesn't directly specify this values,
    # or at the very least they give contrasting values.
    # They reference some possible values of in Section 5.4.3
    # but they refer to the cascaded PI controller.
    # Table 2 specify that the flyzone must be 9x9, since this seemed
    # like the most correct value, as it was backed up by Table 7,
    # so we decided to opt for this.
    flyzone_x: Tuple[float, float] = (-4.5, 4.5)
    """Fly zone in the x direction"""

    flyzone_y: Tuple[float, float] = (-4.5, 4.5)
    """Fly zone in the y direction"""

    flyzone_z: Tuple[float, float] = (0, 9)
    """Fly zone in the z direction"""

    # Table 2
    f_ag: float = 22.92
    """Agent frequency rensponse time"""

    # Section 3.5 ""
    delta_t: float = 1 / f_ag
    """Agent response time"""

    # Section 4.2:
    # "For all trainings we used and initial altitude
    # of the UAV of z_init= 4m"
    z_init: float = 4.0
    """Initial altitude for the UAV"""

    done_descriptions = [
        "\x1b[1;32mSUCCESS\x1b[0m: Touched platform",
        "\x1b[1;32mSUCCESS\x1b[0m: Goal state reached",
        "\x1b[1;31mFAILURE\x1b[0m: Maximum episode duration",
        "\x1b[1;31mFAILURE\x1b[0m: Reached minimum altitude",
        "\x1b[1;31mFAILURE\x1b[0m: Drone moved too far from platform in x direction",
        "\x1b[1;31mFAILURE\x1b[0m: Drone moved too far from platform in y direction",
        "\x1b[1;31mFAILURE\x1b[0m: Drone moved too far from platform in z direction",
    ]
    t_max: int = 40

    def __init__(
        self,
        initial_curriculum_step: int = 0,
        *,
        two_dimensional: bool = False,
    ):
        # Validate inputs
        rospy.init_node("landing_simulation_gym_node")

        self.drone_name = "hummingbird"
        # Setup publishers
        self.action_to_interface_publisher = get_publisher(
            "training_action_interface/action_to_interface", Action, queue_size=0
        )
        self.reset_simulation_publisher = get_publisher(
            "training/reset_simulation", Bool, queue_size=0
        )
        # Setup subscribers
        self.observation_continuous_subscriber = rospy.Subscriber(
            f"/{self.drone_name}/training_observation_interface/observations",
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

        self.two_dimensional = two_dimensional
        self.mdp_x = Mdp(initial_curriculum_step, self.f_ag, self.t_max, direction="x")
        if self.two_dimensional:
            self.mdp_y = Mdp(
                initial_curriculum_step, self.f_ag, self.t_max, direction="y"
            )

        # Messages for ros comunication
        self.continuous_observation = Observation()

        # Other variables needed during execution
        self.current_curriculum_step = initial_curriculum_step
        self.cumulative_reward = 0
        self.number_of_steps = 0

    def publish_action_to_interface(
        self, continuous_action_x: Action, continuous_action_y: Optional[Action]
    ):
        """Function publishes the action values that are currently set to the ROS network."""

        action = continuous_action_x
        if continuous_action_y is not None:
            action.roll = continuous_action_y.roll

        self.action_to_interface_publisher.publish(action)

    def read_training_continuous_observations(self, msg: Observation):
        """Functions reads the continouos observations of the environment whenever the corresponding subsriber to the corresponding ROS topic is triggered."""
        self.observation_continuous = msg

    def set_curriculum_step(self, curriculum_step: int):
        self.current_curriculum_step = 0
        self.mdp_x = Mdp(
            curriculum_step,
            self.f_ag,
            self.t_max,
            direction="x",
        )
        if self.two_dimensional:
            self.mdp_y = Mdp(
                curriculum_step,
                self.f_ag,
                self.t_max,
                direction="y",
            )

    def reset(self):
        """Function resets the training environment and updates logging data"""
        self.cumulative_reward = 0
        self.number_of_steps = 0
        # Reset the setpoints for the low-level controllers of the copter
        self.mdp_x.reset()
        if self.two_dimensional:
            self.mdp_y.reset()

        # Pause simulation and reset Gazebo
        self.pause_sim()
        # self.reset_world_gazebo_service()
        moving_platform = self.model_coordinates("moving_platform", "world")
        # Initialize drone state
        init_drone = ModelState()
        init_drone.model_name = self.drone_name
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
            sigma = self.mdp_x.p_max / 3
            x_init = np.random.normal(mu, sigma)
            # At the first curiculum step we can safely skip y
            y_init = 0
        else:
            x_init = np.random.uniform(self.flyzone_x[0], self.flyzone_x[1])
            y_init = 0.0
            if self.two_dimensional:
                y_init = np.random.uniform(self.flyzone_y[0], self.flyzone_y[1])

        # Clip to stay within fly zone
        init_drone.pose.position.x = np.clip(
            x_init + moving_platform.pose.position.x,
            moving_platform.pose.position.x + self.flyzone_x[0],
            moving_platform.pose.position.x + self.flyzone_x[1],
        )
        init_drone.pose.position.y = np.clip(
            y_init + moving_platform.pose.position.y,
            moving_platform.pose.position.y + self.flyzone_y[0],
            moving_platform.pose.position.y + self.flyzone_y[1],
        )
        init_drone.pose.position.z = self.z_init
        # Section 3.12:
        # Each landing trial will begin with the UAV being in hover state,
        # leading to the following initial conditions for rotational movement
        init_drone.twist.linear.x = 0
        init_drone.twist.linear.y = 0
        init_drone.twist.linear.z = 0
        init_drone.twist.angular.x = 0
        init_drone.twist.angular.y = 0
        init_drone.twist.angular.z = 0
        init_drone.pose.orientation.x = 0
        init_drone.pose.orientation.y = 0
        init_drone.pose.orientation.z = 0
        init_drone.pose.orientation.w = 1.0

        self.set_model_state_service(init_drone)

        # Reset episode counters and logging variables
        self.step_number_in_episode = 0
        self.episode_reward = 0
        self.done_numeric = 0

        self.reset_simulation_publisher.publish(Bool(True))
        # Let simulation update values
        self.unpause_sim()
        rospy.sleep(self.delta_t)
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
        observation = self.observation_continuous
        observation_continuous = ContinuousObservation(
            observation, pitch, roll, abs_v_z
        )
        observation_x = self.mdp_x.discrete_state(
            observation_continuous,
        )
        observation_y = None
        if self.two_dimensional:
            observation_y = self.mdp_y.discrete_state(observation_continuous)
        return observation_x, observation_y

    def step(self, action_x: int, action_y: int = 0):  # type: ignore
        """Function performs one timestep of the training."""

        # Update the setpoints based on the current action and publish them to the ROS network
        continuous_action_x = self.mdp_x.continuous_action(
            action_x,
        )
        continuous_action_y = None

        if self.two_dimensional:
            continuous_action_y = self.mdp_y.continuous_action(
                action_y,
            )

        # self.publish_action_to_interface(continuous_action_x, continuous_action_y)
        action = continuous_action_x
        if continuous_action_y:
            action.roll = continuous_action_y.roll

        self.action_to_interface_publisher.publish(action)
        self.mdp_x.action_values = action
        self.mdp_y.action_values = action

        # Let the simulation run for one RL timestep and allow to recieve obsevation
        self.unpause_sim()
        rospy.sleep(self.delta_t)
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
        observation = self.observation_continuous
        observation_continuous = ContinuousObservation(
            observation, pitch, roll, abs_v_z
        )
        observation_x = self.mdp_x.discrete_state(observation_continuous)
        observation_y = None
        if self.two_dimensional:
            observation_y = self.mdp_y.discrete_state(observation_continuous)

        done_x = self.mdp_x.check()

        info = {}
        reward = 0
        if not self.two_dimensional:
            reward = self.mdp_x.reward()
            self.mdp_x.step_count += 1
            self.cumulative_reward += reward
            if done_x is not None:
                info["termination_condition"] = done_x
                info["termination_description"] = self.done_descriptions[done_x]
                info["cumulative_reward"] = self.cumulative_reward
                info["mean_rewad"] = self.cumulative_reward / self.mdp_x.step_count
                info["duration"] = self.mdp_x.step_count

        done_bool: bool = done_x is not None

        if self.two_dimensional and done_x is not None:
            done_y = self.mdp_y.check()
            info["termination_description_x"] = self.done_descriptions[done_x]

            print(done_y)
            # Sanity check
            done_bool |= (done_y is not None) and (done_y == done_x)
            # If we stay in limits exiting is a false positive
            if done_x == 1:
                done_bool = False

        return observation_x, observation_y, reward, done_bool, info


# Register the training environment in gym as an available one
register(
    id="landing_simulation-v0",
    entry_point="dql_multirotor_landing.landing_simulation_env:LandingSimulationEnv",
)
