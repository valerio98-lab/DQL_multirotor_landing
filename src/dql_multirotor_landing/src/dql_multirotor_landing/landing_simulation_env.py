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
from gym.envs.registration import register
from std_msgs.msg import Bool, Float64
from std_srvs.srv import Empty

from dql_multirotor_landing.mdp import Mdp
from dql_multirotor_landing.msg import (
    Action,  # type:ignore
    LandingSimulationObjectState,  # type:ignore
)
from dql_multirotor_landing.msg import (
    ObservationRelativeState as ObservationRelativeStateMsg,  # type:ignore
)
from dql_multirotor_landing.parameters import Parameters
from dql_multirotor_landing.srv import ResetRandomSeed  # type:ignore
from dql_multirotor_landing.utils import get_publisher

# Register the training environment in gym as an available one
reg = register(
    id="landing_simulation-v0",
    entry_point="dql_multirotor_landing.landing_simulation_env:LandingSimulationEnv",
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
        np.random.seed(self.initial_seed)
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

        self.observation_continous = ObservationRelativeStateMsg()
        self.observation_continous_actions = Action()
        self.observation_drone_state = LandingSimulationObjectState()
        self.current_cur_step_counter = 0

        self.step_number_in_episode = 0
        self.done_numeric = 0
        self.max_number_of_steps_in_episode = max_num_timesteps_episode
        self.test_mode_activated = False
        self.mp_contact_occured = True

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

        # Pause simulation and reset Gazebo
        self.pause_sim()
        self.reset_world_gazebo_service()

        # Initialize drone state
        init_drone = ModelState()
        init_drone.model_name = self.drone_name

        # Determine the initial position based on the probability distribution
        if len(self.parameters.uav_parameters.action_max_values) == 1:
            if self.parameters.simulation_parameters.init_distribution == "uniform":
                x_init = np.random.choice(
                    [
                        np.random.uniform(self.min_x[0], self.min_x[1]),
                        np.random.uniform(self.max_x[0], self.max_x[1]),
                    ],
                    1,
                )

                y_init = np.random.choice(
                    [
                        np.random.uniform(self.min_y[0], self.min_y[1]),
                        np.random.uniform(self.max_y[0], self.max_y[1]),
                    ],
                    1,
                )

            elif self.parameters.simulation_parameters.init_distribution == "normal":
                x_init = np.random.normal(
                    self.parameters.simulation_parameters.init_mu_x,
                    self.parameters.simulation_parameters.init_sigma_x,
                )
                y_init = np.random.normal(
                    self.parameters.simulation_parameters.init_mu_y,
                    self.parameters.simulation_parameters.init_sigma_y,
                )
        else:
            exit()

        # Clip to stay within fly zone
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

        # Set initial altitude and velocity
        init_drone.pose.position.z = self.initial_altitude
        init_drone.twist.linear.x = init_drone.twist.linear.y = (
            init_drone.twist.linear.z
        ) = 0
        init_drone.twist.angular.x = init_drone.twist.angular.y = (
            init_drone.twist.angular.z
        ) = 0

        self.set_model_state_service(init_drone)

        # Let simulation run briefly to update values
        self.unpause_sim()
        rospy.sleep(0.1)
        self.pause_sim()

        # Get observations
        observation = self.mdp.discrete_state(self.observation_continous)

        # Update reward function components if in training mode
        if self.test_mode_activated:
            _ = self.mdp.reward()

        # Reset episode counters and logging variables
        self.step_number_in_episode = 0
        self.episode_reward = 0
        self.done_numeric = 0
        self.reset_happened = True

        self.reset_simulation_publisher.publish(Bool(True))
        return observation

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
            reward = 0 if self.test_mode_activated else self.mdp.reward()
            return False, reward

        reward_weight, message = reward_mapping.get(
            done_numeric, (0, "Unknown termination reason")
        )

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
