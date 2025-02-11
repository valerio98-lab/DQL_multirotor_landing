"""Definition of the `gym` environment"""
import gym
import rospy

NODE_NAME = "landing_simulation"
# TODO: Check where in the launch file this is set, it should be in either `training_q_learning/launch/launch_training.sh` 
# or `training_q_learning/launch/launch_environment_in_virtual_screens.sh` or both
DRONE_NAME = "hummingbird"

def fun1():
    print("ciao")
class LandingSimulation(gym.Env):
    """The environment publishes messages to the following topics """
    def __init__(self):
        super().__init__()
        self.drone_name = DRONE_NAME
        # self.






# These line defines the ROS topics, used for communication within the simulation.
# Each topic is in the form of `(topic_name,message_type)`
action_topic = ...
reset_topic= ...
init_topic = ...
reqad_topic= ...
step_execution_frequency_topic= ...
timestep_error_topic = ...