#! /usr/bin/python3
"""
Script contains the definition of the class defining the training environment and some of its interfaces to other ros nodes.
Furthermore, it registers the landing scenario as an environment in gym.
"""

import gym
import numpy as np
import rospy
from gazebo_msgs.msg import ContactsState, ModelState
from gazebo_msgs.srv import GetModelState, SetModelState
from geometry_msgs.msg import Vector3
from gym.envs.registration import register
from std_msgs.msg import Bool, Float64, Float64MultiArray
from std_srvs.srv import Empty

from training_q_learning.mdp import Mdp
from training_q_learning.msg import (
    Action,  # type:ignore
    LandingSimulationObjectState,  # type:ignore
)
from training_q_learning.msg import (
    ObservationRelativeState as ObservationRelativeStateMsg,  # type:ignore
)
from training_q_learning.parameters import Parameters
from training_q_learning.srv import ResetRandomSeed  # type:ignore
from training_q_learning.utils import get_publisher

# Register the training environment in gym as an available one
reg = register(
    id="landing_simulation-v0",
    entry_point="training_q_learning.landing_simulation_env:LandingSimulationEnv",
)


class LandingSimulationEnv(gym.Env):
    f_ag: float = 22.92
    running_step_time: float = 1 / f_ag

    initial_altitude: float = 4
    limits: dict = {
        "position": [1.0, 0.64, 0.4096, 0.262144, 0.16777216],
        "velocity": [1.0, 0.8, 0.64, 0.512, 0.4096],
        "acceleration": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    max_x: np.ndarray = np.array([0, 4.5])
    min_x: np.ndarray = np.array([-4.5, 0])
    max_y: np.ndarray = np.array([0, 0])
    min_y: np.ndarray = np.array([0, 0])
    max_z: np.ndarray = np.array([0, 0])
    min_z: np.ndarray = np.array([0, 0])
    # TODO: MAKE AMERICA GREAT AGAIN
    minimum_altitude: float = 0.3

    def __init__(
        self,
        initial_seed: int = 43,
        starting_curriculum_step: int = 0,
        max_num_timesteps_episode: int = 100,
    ):
        """Class for setting up a gym based training environment that can be used in combination with Gazebo to train an agent to land on a moving platform."""
        # Get parameters
        rospy.init_node("landing_simulation_gym_node")
        self.initial_seed = initial_seed
        self.parameters = Parameters()
        # TODO: Add starting_curriculum_step
        self.mdp = Mdp()

        self.drone_name: str = "hummingbird"
        self.topic_prefix = "/" + self.drone_name + "/"
        # Set up publishers
        self.action_to_interface_publisher = get_publisher(
            "training_action_interface/action_to_interface", Action, queue_size=0
        )
        self.reset_simulation_publisher = get_publisher(
            "training/reset_simulation", Bool, queue_size=0
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
        rospy.wait_for_service("/moving_platform/reset_random_seed")
        self.mp_reset_randon_seed = rospy.ServiceProxy(
            "/moving_platform/reset_random_seed", ResetRandomSeed
        )

        # Set the random number seed used for initializing the drone
        print("Set seed for initial values to ", self.initial_seed)
        np.random.seed(self.initial_seed)
        print(
            "Send signal to reset random number generator to moving platform trajectory generator "
        )
        self.mp_reset_randon_seed(str(self.initial_seed))

        # region: LandingSimulationObject
        # SUbscribers da rivedere

        # TODO: Metodo Getter
        self.observation_continous_subscriber = rospy.Subscriber(
            self.topic_prefix + "training_observation_interface/observations",
            ObservationRelativeStateMsg,
            self.read_training_continous_observations,
        )
        # TODO: Metodo Getter
        self.observation_drone_state_subscriber = rospy.Subscriber(
            self.topic_prefix + "landing_simulation/world_frame/drone/state",
            LandingSimulationObjectState,
            self.read_drone_state,
        )
        # TODO: Da aggiungere al nodo centrale
        self.mp_contact_subscriber = rospy.Subscriber(
            "/moving_platform/contact", ContactsState, self.read_contact_state
        )

        self.reset_observation = False
        self.observation_continous = ObservationRelativeStateMsg()
        self.observation_continous_actions = Action()
        self.observation_drone_state = LandingSimulationObjectState()
        self.current_cur_step_counter = 0

        # Episode completion variables
        self.done = 0
        self.touchdown_on_platform = False
        self.step_number_in_episode = 0

        # Other variables required for execution
        self.cum_reward = 0

        self.done_numeric = 0
        self.max_number_of_steps_in_episode = max_num_timesteps_episode
        self.test_mode_activated = False
        self.mp_contact_occured = True
        # endregion

        # Other script variables
        self.test_mode_activated = False
        self.log_dir_path = None
        self.date_prefix = None

        print("Done setting up training environment...")
        return

    def set_curriculum_step(self, curriculum_step: int):
        new_limits = {
            state: limit[:curriculum_step] for state, limit in self.limits.items()
        }
        self.mdp = Mdp(new_limits)

    def reset(self):
        """Function resets the training environment and updates logging data"""

        # Reset the setpoints for the low-level controllers of the copter

        self.mdp.reset()

        self.publish_action_to_interface()

        # Extract terminal condition that led to reset
        self.done_numeric = self.done_numeric
        self.touchdown_on_platform = self.touchdown_on_platform

        self.pause_sim()

        # Reset gazebo
        self.reset_world_gazebo_service()

        # Begin computation of new init state of the drone
        init_drone = ModelState()
        init_drone.model_name = self.drone_name

        # Determine the init value according to a probability distribution
        if len(self.parameters.uav_parameters.action_max_values) == 1:
            if self.parameters.simulation_parameters.init_distribution == "uniform":
                x_vec = [
                    np.random.uniform(low=self.min_x[0], high=self.min_x[1], size=None),
                    np.random.uniform(low=self.max_x[0], high=self.max_x[1], size=None),
                ]
                y_vec = [
                    np.random.uniform(low=self.min_y[0], high=self.min_y[1], size=None),
                    np.random.uniform(low=self.max_y[0], high=self.max_y[1], size=None),
                ]
                x_init = np.random.choice(x_vec, 1)
                y_init = np.random.choice(y_vec, 1)

            elif self.parameters.simulation_parameters.init_distribution == "normal":
                # Get parameters specifying the normal distribution
                init_mu_x = self.parameters.simulation_parameters.init_mu_x
                init_sigma_x = self.parameters.simulation_parameters.init_sigma_x
                init_mu_y = self.parameters.simulation_parameters.init_mu_y
                init_sigma_y = self.parameters.simulation_parameters.init_sigma_y

                # Draw init value from normal distribution
                x_init = np.random.normal(init_mu_x, init_sigma_x)
                y_init = np.random.normal(init_mu_y, init_sigma_y)
        else:
            print("Only 1D case is implemented. Aborting...")
            exit()

        # Compute the init position within the specified fly zone ('absolute') or relative to the moving platform ('relative'). Clip to make sure the drone is not initialized outside the flyzone

        init_drone.pose.position.x = np.clip(
            x_init,
            -self.parameters.simulation_parameters.max_abs_p_x,
            self.parameters.simulation_parameters.max_abs_p_x,
        )
        init_drone.pose.position.y = np.clip(
            y_init,
            -self.parameters.simulation_parameters.max_abs_p_y,
            self.parameters.simulation_parameters.max_abs_p_y,
        )

        # Define the new init state (init velocity is 0)
        init_drone.pose.position.z = self.initial_altitude
        init_drone.twist.linear.x = 0
        init_drone.twist.linear.y = 0
        init_drone.twist.linear.z = 0
        init_drone.twist.angular.x = 0
        init_drone.twist.angular.y = 0
        init_drone.twist.angular.z = 0
        self.set_model_state_service(init_drone)

        # Let simulation run for a short time to collect the new values after the initialization
        self.unpause_sim()
        rospy.sleep(0.1)
        object_coordinates_moving_platform_after_reset = self.model_coordinates(
            "moving_platform", "world"
        )
        object_coordinates_drone_after_reset = self.model_coordinates(
            self.drone_name, "world"
        )
        self.pause_sim()

        # observation = self.convert_observation_msg(observation_msg)
        # Normalize and clip the observations
        # Map the current observation to a discrete state value
        self.reset_observation = True
        self.reset_observation = False
        observation = self.mdp.discrete_state(self.observation_continous)

        # Execute reward function once to update the time dependent components in the reward function, when a training and not a test is running
        if self.test_mode_activated:
            _ = self.mdp.reward((self.observation_continous))

        # Update the parameters required to run the simulation

        self.step_number_in_episode = 0
        self.step_number_in_episode = 0
        self.episode_reward = 0
        self.cum_reward = 0
        self.done_numeric = 0
        self.reset_happened = True

        # Send reset signal to ROS network -> Manda a Valerio
        self.send_reset_simulation_signal(True)  # type: ignore

        return observation

    def send_reset_simulation_signal(self, status: Bool):
        """Function sends out a boolean value indicating that a reset has been comnpleted. This can be used in other nodes that need reset, such as the action to training interface node."""
        msg_reset = Bool()
        msg_reset.data = True
        # TODO: Comunicazione a Valerio
        self.reset_simulation_publisher.publish(msg_reset)
        self.mp_contact_occured = False
        return

    # TODO: Chiamata a Valerio
    def publish_action_to_interface(self):
        """Function publishes the action values that are currently set to the ROS network."""
        msg_action = Action()
        for msg_string in self.mdp.action_values.keys():
            setattr(msg_action, msg_string, self.mdp.action_values[msg_string])
        self.action_to_interface_publisher.publish(msg_action)
        return

    def step(self, action):
        """Function performs one timestep of the training."""
        # Reset values if previous step was the reset step
        if self.reset_happened:
            self.touchdown_on_platform = False
            self.touchdown_on_platform = False
            self.reset_happened = False

        # Update the setpoints based on the current action and publish them to the ROS network
        self.mdp.update_action_values(action)
        self.publish_action_to_interface()

        # Let the simulation run for one RL timestep and measure the wall clock time that has passed during that timestep
        self.unpause_sim()
        rospy.sleep(self.running_step_time)
        self.pause_sim()

        # Map the current observation to a discrete state value

        observation = self.mdp.discrete_state(self.observation_continous)

        # Update the number of episodes
        self.step_number_in_episode += 1
        self.step_number_in_episode = self.step_number_in_episode

        # Check if terminal condition is reached
        (done, reward) = self.process_termination_and_reward(self.observation_continous)

        info = {"culo", 1}
        self.reward = reward
        self.episode_reward += reward

        # Publish the reward
        msg_reward = Float64()
        msg_reward.data = reward

        return observation, reward, done, info

    # TODO: Valerio deve darmi un setter adeguato :(
    def read_drone_state(self, msg: LandingSimulationObjectState):
        """Function reads the current state of the drone whenever triggered by the corresponding subscriber to the corresponding ROS topic."""
        self.observation_drone_state = msg
        return

    def read_training_continous_observations(self, msg: ObservationRelativeStateMsg):
        """Functions reads the continouos observations of the environment whenever the corresponding subsriber to the corresponding ROS topic is triggered."""
        self.observation_continous = msg
        return

    # TODO: Valeio deve darmi un setter adeguato
    def read_contact_state(self, msg: ContactsState):
        """Function checks if the contact sensor on top of the moving platform sends values. If yes, a flag is set to true."""
        if msg.states:
            self.mp_contact_occured = True
        return

    def process_termination_and_reward(self, msg_rel_state):
        """Determines whether the episode should terminate and assigns the corresponding reward."""

        # Track goal state
        goal_state_reached = False
        if self.mdp.current_discrete_state[0] == self.mdp.previous_discrete_state[0]:
            self.current_cur_step_counter += 1
            success_duration = int(
                self.parameters.rl_parameters.cur_step_success_duration
                / self.parameters.rl_parameters.running_step_time
            )
            if (
                self.current_cur_step_counter > success_duration
                and self.mdp.current_discrete_state[0]
                == len(self.mdp.limits["position"]) - 1
                and self.mdp.current_discrete_state[1] == 1
                and self.mdp.current_discrete_state[2] == 1
            ):
                goal_state_reached = True
        else:
            self.current_cur_step_counter = 0

        # List of termination conditions
        done_conditions = {
            1: self.step_number_in_episode >= self.max_number_of_steps_in_episode,
            3: abs(self.observation_drone_state.pose.pose.position.x)
            >= self.parameters.simulation_parameters.max_abs_p_x,
            4: abs(self.observation_drone_state.pose.pose.position.y)
            >= self.parameters.simulation_parameters.max_abs_p_y,
            7: self.observation_continous.rel_p_z > -self.minimum_altitude
            and self.parameters.simulation_parameters.done_criteria["minimum_altitude"],
            8: goal_state_reached
            and self.parameters.simulation_parameters.done_criteria["success"],
            9: self.mp_contact_occured
            and self.parameters.simulation_parameters.done_criteria[
                "touchdown_contact"
            ],
        }

        # Find first termination reason
        done_numeric = next(
            (key for key, condition in done_conditions.items() if condition), 0
        )

        # Reset counter on termination
        if done_numeric:
            self.current_cur_step_counter = 0

        # Reward & messages mapping
        reward_mapping = {
            1: (
                self.parameters.simulation_parameters.w_fail,
                "Max. number of steps reached.",
            ),
            3: (
                self.parameters.simulation_parameters.w_fail,
                "Longitudinal distance too big.",
            ),
            4: (
                self.parameters.simulation_parameters.w_fail,
                "Lateral distance too big.",
            ),
            7: (0, "Minimum altitude reached."),
            8: (
                self.parameters.simulation_parameters.w_suc,
                "SUCCESS: GOAL STATE REACHED",
            ),
            9: (0, "Touchdown on platform."),
        }

        # Default reward if episode continues
        if done_numeric == 0:
            reward = 0 if self.test_mode_activated else self.mdp.reward(msg_rel_state)
            return False, reward

        # Print message & compute reward
        reward_weight, message = reward_mapping.get(
            done_numeric, (0, "Unknown termination reason")
        )
        print(f"END OF EPISODE: {message}")

        reward = reward_weight * self.mdp.max_possible_reward_for_one_timestep
        done = self.parameters.simulation_parameters.done_criteria.get(
            {
                1: "max_num_timesteps",
                3: "max_lon_distance",
                4: "max_lat_distance",
                7: "minimum_altitude",
                8: "success",
                9: "touchdown_contact",
            }[done_numeric],
            False,
        )

        return done, reward
