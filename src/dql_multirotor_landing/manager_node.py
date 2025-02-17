#!/usr/bin/env python3
import rospy
import tf2_ros
from std_msgs.msg import Float64MultiArray, Bool, Header
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped, Quaternion, Vector3
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import Odometry
from dql_multirotor_landing.msg import LandingSimulationObjectState, ObservationRelativeState, Action
from dql_multirotor_landing.srv import ResetRandomSeed, ResetRandomSeedResponse
from dql_multirotor_landing.parameters import Parameters
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose
from dql_multirotor_landing.moving_platform import MovingPlatform
from dql_multirotor_landing.rotors_interface_cpp import RotorsInterface_CPP # type: ignore
from dql_multirotor_landing.filters import KalmanFilter3D
from dql_multirotor_landing.observation_utils import ObservationUtils, ObservationRelativeStateData
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np


class State():
    def __init__(self):
        self.pose = PoseStamped()
        self.pose.header.frame_id = 'world'
        self.twist = TwistStamped()
        self.twist.header.frame_id = 'world'
        self.twist.twist.linear = Vector3Stamped()
        self.twist.twist.angular = Vector3Stamped()
        self.linear_acceleration = Vector3Stamped()
        self.linear_acceleration.header.frame_id = 'world'


class ManagerNode():
    """
    ROS node responsible for managing the drone and moving platform state, handling sensor data, 
    and publishing necessary transformations and observations for reinforcement learning. 

    This node listens to:
    - Gazebo model states for tracking drone and platform position.
    - Odometry data to compute stability axes.
    - Training commands for reset and action execution.

    It also publishes:
    - Drone and platform states in both world and target frames.
    - Relative position and velocity of the drone with respect to the platform.
    - Observations and action commands for reinforcement learning.
    """

    def __init__(self):
        rospy.init_node('central_logic_node')
        self.parameters = Parameters()
        ns = rospy.get_namespace()
        self.node_name = 'central_logic_node'
        self.drone_name = rospy.get_param(ns + self.node_name + '/drone_name', 'hummingbird')
        self.publish_rate = float(rospy.get_param(ns + self.node_name + '/publish_rate_hz', '100'))
        self.noise_pos_sd = float(rospy.get_param(ns + self.node_name + '/noise_pos_sd', '0.25'))
        self.noise_vel_sd = float(rospy.get_param(ns + self.node_name + '/noise_vel_sd', '0.1'))
        
        self.gazebo_frame = 'world'
        self.target_frame = self.drone_name + '/stability_axes'
        
        self.kalman_filter = KalmanFilter3D(
            process_variance=1e-4, 
            measurement_variance=self.noise_vel_sd
        )

        self.utils = ObservationUtils(target_frame=self.target_frame, world_frame=self.gazebo_frame, noise_pos_sd=self.noise_pos_sd, noise_vel_sd=self.noise_vel_sd, filter=self.kalman_filter)


        self.drone_wf = State()
        self.mp_wf = State()
        self.drone_tf = State()
        self.mp_state_target = State()
        self.observation_data = ObservationRelativeStateData()

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.br = tf2_ros.TransformBroadcaster()

        self._init_publisher()
        self._init_subscriber()

        self.reset_service = rospy.Service('/moving_platform/reset_random_seed', ResetRandomSeed, self._reset_random_seed)



        self.moving_platform = MovingPlatform()
        self.action_interface = RotorsInterface_CPP()


    def _init_publisher(self):
        self.gazebo_pose_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=3) ##Incorpora i wait for service
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
            self.set_gazebo_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        except rospy.ROSException:
            rospy.logerr("Service /gazebo/set_model_state not available")
            rospy.signal_shutdown("Service /gazebo/set_model_state not available")

        self.relative_vel_pub = rospy.Publisher('landing_simulation/relative_moving_platform_drone/state/twist', TwistStamped, queue_size=0)
        self.relative_pos_pub = rospy.Publisher('landing_simulation/relative_moving_platform_drone/state/pose', PoseStamped, queue_size=0)
        self.relative_rpy_pub = rospy.Publisher('landing_simulation/relative_moving_platform_drone/debug_target_frame/roll_pitch_yaw', Float64MultiArray, queue_size=0)
        
        ##devo pubblicare training/reset_simulation e ti fornisco metodo setter a josè


        ############################DA ELIMINARE##############################
        self.drone_state_world_pub = rospy.Publisher('landing_simulation/world_frame/drone/state', LandingSimulationObjectState, queue_size=0) #fatto metodo per josè
        self.observation_pub = rospy.Publisher('training_observation_interface/observations', ObservationRelativeState, queue_size=0) #fatto metodo per josè
        ############################DA ELIMINARE##############################
        

    def _init_subscriber(self):
        reset_topic = "/" + self.drone_name + "/training/reset_simulation"
        self.reset_sub = rospy.Subscriber(reset_topic, Bool, self._read_reset)
        self.odom_sub = rospy.Subscriber('odometry_sensor1/odometry', Odometry, self._odometry_callback)
        self.environment_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self._environment_callback)
        self.reset_sub = rospy.Subscriber('training/reset_simulation', Bool, self._reset_callback) ##forse si passa internamente a interface
        self.action_sub = rospy.Subscriber('training_action_interface/action_to_interface', Action, self._action_callback) #Io lo dovrei pubblicare ed esporre setter a josè
        self.contact = ... ##Sottoscrittore per il contatto con piattaforma + metodo get contact


    def _read_reset(self, msg):
        if msg.data:
            self.t = np.random.uniform(0, self.parameters.rl_parameters.max_num_timesteps_episode * self.parameters.rl_parameters.running_step_time)


    def get_observation(self, drone_tf, mp_tf):
        #Update the moving platform state
        pose, u, v = self.moving_platform.update()
        self._publish_trajectory(pose, u, v)
        #Compute and publish the relative position and velocity between drone and platform
        rel_pos, rel_vel = self.utils.get_relative_state(drone_tf, mp_tf)
        self.relative_pos_pub.publish(rel_pos)
        self.relative_vel_pub.publish(rel_vel)

        #Compute and publish the relative roll, pitch, and yaw angles
        rpy_angles = np.degrees(euler_from_quaternion([
        rel_pos.pose.orientation.x,
        rel_pos.pose.orientation.y,
        rel_pos.pose.orientation.z,
        rel_pos.pose.orientation.w
        ]))
        rpy_msg = Float64MultiArray(data=rpy_angles)
        self.relative_rpy_pub.publish(rpy_msg)

        #Compute and publish observation data for reinforcement learning
        obs_msg = self.utils.get_observation_from_env(rel_pos, rel_vel, self.observation_data)
        self.observation_pub.publish(obs_msg)

        #return obs_msg

    def get_drone_state_wf(self):
        self.drone_state_world_pub.publish(self._compute_landing_state_msg(self.drone_wf))


    def _publish_trajectory(self, pose, u, v):

        gazebo_msg = ModelState()
        gazebo_msg.model_name = 'moving_platform'
        gazebo_msg.reference_frame = 'ground_plane'
        gazebo_msg.pose = pose
        gazebo_msg.twist.linear.x = u
        gazebo_msg.twist.linear.y = v
        self.gazebo_pose_pub.publish(gazebo_msg)
    
    

    def _odometry_callback(self, odom_msg):
        """
        Callback for processing odometry data. 

        The odometry provides the full orientation (roll, pitch, yaw), but for the stability frame transformation, 
        only yaw is extracted. This is because:
        - By construction the stability frame is designed to remain always parallel to the ground, ignoring roll and pitch.
        - Yaw represents the drone's heading in the horizontal plane, which is relevant for navigation and control.

        The extracted yaw is then used to publish the stability frame relative to the world.

        :param odom_msg: ROS Odometry message containing the drone's current pose.
        """
        yaw = self._extract_yaw(odom_msg.pose.pose.orientation)

        pos = (
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z
        )

        self._broadcast_stability_tf(odom_msg.header.frame_id, yaw, pos)


    def _environment_callback(self, msg):
        """
        Callback for Gazebo model states.
        Extracts the position and velocity of the drone and moving platform.

        :param msg: Gazebo ModelStates message.
        """
        try:
            drone_index = msg.name.index(self.drone_name)
            mp_index = msg.name.index("moving_platform")
        except ValueError:
            rospy.logwarn("Unable to find drone or moving_platform in the model states")
            return

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.gazebo_frame

        self.drone_wf.pose = PoseStamped(header=header, pose=msg.pose[drone_index])
        self.mp_wf.pose = PoseStamped(header=header, pose=msg.pose[mp_index])

        self.drone_wf.twist.twist.linear.vector = msg.twist[drone_index].linear
        self.drone_wf.twist.twist.angular.vector = msg.twist[drone_index].angular
        self.mp_wf.twist.twist.linear.vector = msg.twist[mp_index].linear
        self.mp_wf.twist.twist.angular.vector = msg.twist[mp_index].angular


    def _reset_callback(self, msg):
        """
        Callback for reset command. If the reset flag is received, 
        the simulation reset request is stored in observation data.

        :param msg: Bool message indicating whether to reset the simulation.
        """
        if msg.data:
            self.observation_data.request_simulation_reset = True


    def _action_callback(self, msg):
        """
        Callback for receiving control actions.

        :param msg: Action message containing roll, pitch, yaw, and vertical velocity commands.
        """
        self.observation_data.action_setpoints = msg


    def _compute_landing_state_msg(self, state):
        """
        Converts a given state into a LandingSimulationObjectState message.

        :param state: The state object to be converted.
        :return: LandingSimulationObjectState message with the current timestamp.
        """
        msg = LandingSimulationObjectState()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = state.pose.header.frame_id
        msg.pose = state.pose
        msg.twist = state.twist 
        msg.linear_acceleration = state.linear_acceleration 
        return msg
    

    def _broadcast_stability_tf(self, source_frame, yaw, pos):
        """
        Publishes a stability reference frame transformation in the TF tree.

        :param source_frame: The parent frame for the transformation.
        :param yaw: Rotation around the Z-axis (yaw angle in radians).
        :param pos: (x, y, z) position tuple.
        """
        current_time = rospy.Time.now()

        transform_msg = tf2_ros.TransformStamped()
        transform_msg.header = Header(stamp=current_time, frame_id=source_frame)
        transform_msg.child_frame_id = f"{self.drone_name}/stability_axes"

        x, y, z = pos
        transform_msg.transform.translation = Vector3(x, y, z)

        q = quaternion_from_euler(0.0, 0.0, yaw)
        transform_msg.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        self.br.sendTransform(transform_msg)


    
    def _extract_yaw(self, orientation):
        """
        Extracts the yaw (rotation around Z-axis) from a quaternion.

        :param orientation: Quaternion representing the object's orientation.
        :return: Yaw angle in radians.
        """
        # Estrae solo il terzo elemento (yaw) dalla conversione
        return euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])[2]


    def run(self):
        """
        Main loop that continuously updates and publishes the state of the drone and moving platform.
        """
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            
            #Transform drone and moving platform states into the target frame
            check_trans, drone_tf, mp_tf = self.utils.transform_world_to_target_frame(
                drone_wf=self.drone_wf, 
                mp_wf=self.mp_wf, 
                drone_tf=self.drone_tf, 
                mp_tf=self.mp_state_target,
                buffer=self.tfBuffer
            )
            if not check_trans:
                rate.sleep()
                continue
            
            self.get_observation(drone_tf, mp_tf)
            self.get_drone_state_wf()

            #Apply control commands to the drone
            self.action_interface.update()
            rate.sleep()

    def _reset_random_seed(self, req):
        """
        Function handles the service request to reset the seed for the random number generator.

        :param req: ResetRandomSeed request message containing the new seed value.
        :return: ResetRandomSeed response message.
        """
        seed = None if req.seed == 'None' else int(req.seed)
        rospy.loginfo("Set seed for random initial values to %s", seed)
        np.random.seed(seed)
        return ResetRandomSeedResponse()

if __name__ == '__main__':
    node = ManagerNode()
    node.run()

