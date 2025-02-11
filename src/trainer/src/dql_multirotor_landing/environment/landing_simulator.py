"""Definition of the `gym` environment"""
import gym
import rospy
from gazebo_msgs.srv import GetModelState, SetModelState
from std_msgs.msg import Bool, Float64
from std_srvs.srv import Empty

NODE_NAME = "landing_simulation"
# TODO: Check where in the launch file this is set, it should be in either `training_q_learning/launch/launch_training.sh` 
# or `training_q_learning/launch/launch_environment_in_virtual_screens.sh` or both
DRONE_NAME = "hummingbird"
class LandingSimulation(gym.Env):
    """The environment publishes messages to the following topics """
    def __init__(self):
        super().__init__()
        self.drone_name = DRONE_NAME

        # TODO: Qui dovrebbero andare i publisher possiamo pero 
        # incorpoarli qui evitando i troppi laye di indirezione che utilizzano
        ...

        # Set up services, kinda of like function call ove the newtwork.
        # Before being able to use them we need to make sure that they are online.

        # Resets the simulation world.
        rospy.wait_for_service('/gazebo/reset_world')
        self.reset_world_gazebo_service = rospy.ServiceProxy('/gazebo/reset_world',Empty)

        # Services for manually setting/getting the position and velocity of an object in Gazebo
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
        rospy.wait_for_service('/gazebo/get_model_state')
        self.model_coordinates = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)

        # Services for stopping/unstopping physics simulation. The world stops moving, but sensors and processes continue running.
        rospy.wait_for_service('/gazebo/pause_physics')
        self.pause_sim = rospy.ServiceProxy('/gazebo/pause_physics',Empty)
        rospy.wait_for_service('/gazebo/unpause_physics')
        self.unpause_sim = rospy.ServiceProxy('/gazebo/unpause_physics',Empty)
        

        # rospy.wait_for_service('/moving_platform/reset_random_seed')
        # self.mp_reset_randon_seed = rospy.ServiceProxy('/moving_platform/reset_random_seed',ResetRandomSeed)

        # TODO: Setting up the parameters




gym.register(
    "Gazebo-Landing-Simulator-v0"
    , entry_point= "dql_multi_rotor_landing.environment.landing_simulator:LandingSimulation"
)