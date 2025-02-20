#! /usr/bin/python3
"""
Script contains the definition of the class defining the training environment and some of its interfaces to other ros nodes.
Furthermore, it registers the landing scenario as an environment in gym.
"""

from typing import List, Literal, Optional

import gym
import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from gym.envs.registration import register
from std_msgs.msg import Bool
from std_srvs.srv import Empty

from dql_multirotor_landing.mdp import Mdp
from dql_multirotor_landing.msg import Action, Observation

# from dql_multirotor_landing.srv import ResetRandomSeed
from dql_multirotor_landing.utils import get_publisher

# Register the training environment in gym as an available one
reg = register(
    id="landing_simulation-v0",
    entry_point="dql_multirotor_landing.landing_simulation_env:LandingSimulationEnv",
)


class LandingSimulationEnv(gym.Env):
    # The paper doesn't directly specify this values,
    # or at the very least they give contrasting values.
    # They reference some possible values of in Section 5.4.3
    # but they refer to the cascaded PI controller.
    # Table 2 specify that the flyzone must be 9x9, since this seemed
    # like the most correct value, as it was backed up by Table 7,
    # so we decided to opt for this.
    flyzone_x: np.ndarray = np.array([-4.5, 4.5])
    """Fly zone in the x direction"""

    flyzone_y: np.ndarray = np.array([-4.5, 4.5])
    """Fly zone in the y direction"""

    flyzone_z: np.ndarray = np.array([0, 9])
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

    minimum_altitude: float = 0.2
    """This is the height of the platform"""

    done_descriptions = [
        "SUCCESS: Touched platform",
        "SUCCESS: Goal state reached",
        "FAILURE: Maximum episode duration",
        "FAILURE: Reached minimum altitude",
        "FAILURE: Drone moved too far from platform in x direction",
        "FAILURE: Drone moved too far from platform in y direction",
        "FAILURE: Drone moved too far from platform in z direction",
    ]
    t_max: int = 20

    def __init__(
        self,
        directions: List[Literal["x", "y"]] = ["x"],
        curriculum_step: int = 0,
        *,
        initial_seed: Optional[int] = 42,
    ):
        # Validate inputs
        if directions != ["x"] and directions != ["x", "y"]:
            # It isn't specified directly but the gist of all can be found in Section 3.3.1.
            # They claim that:
            #
            #   - The use of a specific controller structure allows us to
            #       introduce a 1D motion in the x direction
            #
            #   - After the training we can reuse the same values
            #       for the y direction allowing us full 2D moion capabilities
            raise ValueError(
                'Direction must be either ["x"](training situation) or ["x","y"](test situation)'
            )
        self.initial_seed = initial_seed

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

        np.random.seed(self.initial_seed)

        self.directions = directions
        self.mdp_x = Mdp(curriculum_step, self.f_ag, self.t_max)
        if len(directions) == 2:
            self.mdp_y = Mdp(curriculum_step, self.f_ag, self.t_max)

        # Messages for ros comunication
        self.observation_continuous = Observation()
        self.observation_continuous_actions = Action()

        # Other variables needed during execution
        self.current_curriculum_step = curriculum_step
        self.cumulative_reward = 0

    def publish_action_to_interface(
        self, continuous_action_x: Action, continuous_action_y: Optional[Action]
    ):
        """Function publishes the action values that are currently set to the ROS network."""
        action = continuous_action_x
        if continuous_action_y:
            action.roll = continuous_action_y.roll
        self.action_to_interface_publisher.publish(action)

    def read_training_continuous_observations(self, msg: Observation):
        """Functions reads the continouos observations of the environment whenever the corresponding subsriber to the corresponding ROS topic is triggered."""
        self.observation_continuous = msg

    def set_curriculum_step(self, curriculum_step: int):
        self.current_curriculum_step = 0
        self.mdp_x = Mdp(curriculum_step, self.f_ag, self.t_max)
        if len(self.directions) == 2:
            self.mdp_y = Mdp(curriculum_step, self.f_ag, self.t_max)

    def reset(self):
        """Function resets the training environment and updates logging data"""

        # Reset the setpoints for the low-level controllers of the copter
        self.mdp_x.reset()
        if len(self.directions) == 2:
            self.mdp_y.reset()

        # Pause simulation and reset Gazebo
        self.pause_sim()
        self.reset_world_gazebo_service()

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
            y_init = np.random.uniform(self.flyzone_y[0], self.flyzone_y[1])

        # Clip to stay within fly zone
        init_drone.pose.position.x = np.clip(
            x_init, self.flyzone_x[0], self.flyzone_x[1]
        )
        init_drone.pose.position.y = np.clip(
            y_init, self.flyzone_y[0], self.flyzone_y[1]
        )
        object_coordinates_moving_platform = self.model_coordinates(
            "moving_platform", "world"
        )
        object_coordinates_drone_after_reset = self.model_coordinates(
            "hummingbird", "world"
        )
        print(
            f"{object_coordinates_moving_platform=},{object_coordinates_drone_after_reset=}"
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

        self.set_model_state_service(init_drone)

        # Reset episode counters and logging variables
        self.step_number_in_episode = 0
        self.episode_reward = 0
        self.done_numeric = 0

        self.reset_simulation_publisher.publish(Bool(True))
        # Let simulation update values
        self.unpause_sim()
        rospy.sleep(0.3)
        self.pause_sim()
        # Save temporarily the current continuous state to avoid sinchronization issues
        # due to callbacks
        observation_continuous = self.observation_continuous
        observation_x = self.mdp_x.discrete_state(observation_continuous)
        observation_y = None
        if len(self.directions) == 2:
            observation_y = self.mdp_y.discrete_state(observation_continuous)
        return observation_x, observation_y

    def step(self, action_x: int, action_y: int = 0):
        """Function performs one timestep of the training."""

        # Update the setpoints based on the current action and publish them to the ROS network
        continuous_action_x = self.mdp_x.continuous_action(
            action_x,
        )
        continuous_action_y = None

        if len(self.directions) == 2:
            continuous_action_y = self.mdp_y.continuous_action(
                action_y,
            )

        self.publish_action_to_interface(continuous_action_x, continuous_action_y)

        # Let the simulation run for one RL timestep and allow to recieve obsevation
        self.unpause_sim()
        rospy.sleep(self.delta_t)
        self.pause_sim()

        # Map the current observation to a discrete state value
        observation_continuous = self.observation_continuous
        observation_x = self.mdp_x.discrete_state(observation_continuous)
        observation_y = None
        if len(self.directions) == 2:
            observation_y = self.mdp_y.discrete_state(observation_continuous)

        done_x = self.mdp_x.check()

        info = {}
        if len(self.directions) == 1:
            reward = self.mdp_x.reward()
            info = {"reward": reward}
            self.mdp_x.step_count += 1
            self.cumulative_reward += reward
            if done_x is not None:
                info["termination_condition"] = done_x
                info["termination_description"] = self.done_descriptions[done_x]
                info["cumulative_reward"] = self.cumulative_reward
                info["mean_rewad"] = self.cumulative_reward / self.mdp_x.step_count
                info["duration"] = self.mdp_x.step_count

        done_bool: bool = done_x is not None
        if len(self.directions) == 2 and not done_bool:
            done_y = self.mdp_y.check()
            # Sanity check
            done_bool |= (done_y is None) and (done_y == done_x)

        return observation_x, observation_y, reward, done_bool, info
