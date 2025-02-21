#!/usr/bin/env python3
from dataclasses import dataclass

import numpy as np
import rospy
import tf2_ros

# from dql_multirotor_landing.moving_platform import MovingPlatformNode
from gazebo_msgs.msg import ContactsState, ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3, Vector3Stamped
from mav_msgs.msg import RollPitchYawrateThrust
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64, Float64MultiArray, Header
from tf.transformations import euler_from_quaternion

# from dql_multirotor_landing.rotors_interface_cpp import RotorsInterface_CPP # type: ignore
from dql_multirotor_landing.filters import KalmanFilter3D
from dql_multirotor_landing.moving_platform import MovingPlatform
from dql_multirotor_landing.msg import Action, Observation
from dql_multirotor_landing.observation_utils import ObservationData, ObservationUtils
from dql_multirotor_landing.srv import ResetRandomSeed, ResetRandomSeedResponse


@dataclass
class PID_Setpoints:
    pitch: float
    roll: float
    v_z: float
    yaw: float


@dataclass
class thrust_cmd:
    vz_effort: float
    vz_state: float
    yaw_effort: float
    yaw_state: float


DEBUG = True


class State:
    def __init__(self):
        self.pose = PoseStamped()
        self.pose.header.frame_id = "world"
        self.twist = TwistStamped()
        self.twist.header.frame_id = "world"
        self.twist.twist.linear = Vector3Stamped()  # type: ignore
        self.twist.twist.angular = Vector3Stamped()  # type: ignore
        self.linear_acceleration = Vector3Stamped()
        self.linear_acceleration.header.frame_id = "world"


class ManagerNode:
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

    Additionally, this class manages a set of functions interfacing with C++ nodes
    responsible for PID control and thrust management.
    """

    def __init__(self):
        rospy.init_node("central_logic_node")
        ns = rospy.get_namespace()
        self.node_name = "central_logic_node"
        self.drone_name: str = rospy.get_param(
            ns + self.node_name + "/drone_name", "hummingbird"
        )  # type: ignore
        self.publish_rate: float = float(
            rospy.get_param(ns + self.node_name + "/publish_rate_hz", "100")  # type: ignore
        )

        self.noise_pos_sd: float = float(
            rospy.get_param(ns + self.node_name + "/noise_pos_sd", "0.25")  # type: ignore
        )
        self.noise_vel_sd: float = float(
            rospy.get_param(ns + self.node_name + "/noise_vel_sd", "0.1")  # type: ignore
        )
        self.t_max: int = int(
            rospy.get_param(ns + self.node_name + "/t_max", "20")  # type: ignore
        )

        self.gazebo_frame = "world"
        self.target_frame = self.drone_name + "/stability_axes"

        self.kalman_filter = KalmanFilter3D(
            process_variance=1e-4, measurement_variance=self.noise_vel_sd
        )

        self.utils = ObservationUtils(
            drone_name=self.drone_name,
            target_frame=self.target_frame,
            world_frame=self.gazebo_frame,
            noise_pos_sd=self.noise_pos_sd,
            noise_vel_sd=self.noise_vel_sd,
            filter=self.kalman_filter,  # type: ignore
        )

        self.drone_wf = State()
        self.mp_wf = State()
        self.drone_tf = State()
        self.mp_tf = State()
        self.observation_data = ObservationData()

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.br = tf2_ros.TransformBroadcaster()

        self._init_publisher()
        self._init_subscriber()

        ##Set up service
        try:
            rospy.wait_for_service("/gazebo/set_model_state", timeout=10)
            self.set_gazebo_state = rospy.ServiceProxy(
                "/gazebo/set_model_state", SetModelState
            )
        except rospy.ROSException:
            rospy.logerr("Service /gazebo/set_model_state not available")
            rospy.signal_shutdown("Service /gazebo/set_model_state not available")

        self.reset_service = rospy.Service(
            "/moving_platform/reset_random_seed",
            ResetRandomSeed,
            self._reset_random_seed,
        )
        ##

        ## Set up module for moving platform and pid controllers
        self.moving_platform = MovingPlatform()
        self.pid_setpoints = PID_Setpoints(0, 0, 0, 0)
        self.effort = thrust_cmd(0, 0, 0, 0)
        self.mp_contact_occured = False

    def _init_publisher(self):
        self.gazebo_pose_pub = rospy.Publisher(
            "/gazebo/set_model_state", ModelState, queue_size=3
        )

        self.relative_vel_pub = rospy.Publisher(
            "landing_simulation/relative_moving_platform_drone/state/twist",
            TwistStamped,
            queue_size=0,
        )
        self.relative_pos_pub = rospy.Publisher(
            "landing_simulation/relative_moving_platform_drone/state/pose",
            PoseStamped,
            queue_size=0,
        )
        self.relative_rpy_pub = rospy.Publisher(
            "landing_simulation/relative_moving_platform_drone/debug_target_frame/roll_pitch_yaw",
            Float64MultiArray,
            queue_size=0,
        )
        self.observation_pub = rospy.Publisher(
            "training_observation_interface/observations",
            Observation,
            queue_size=0,
        )

        self._pub_vz_setpoint = rospy.Publisher(
            "training_action_interface/setpoint/v_z", Float64, queue_size=3
        )
        self._pub_vz_state = rospy.Publisher(
            "training_action_interface/state/v_z", Float64, queue_size=3
        )
        self._pub_yaw_setpoint = rospy.Publisher(
            "training_action_interface/setpoint/yaw", Float64, queue_size=3
        )
        self._pub_yaw_state = rospy.Publisher(
            "training_action_interface/state/yaw", Float64, queue_size=3
        )
        self._pub_rpy_thrust = rospy.Publisher(
            "command/roll_pitch_yawrate_thrust", RollPitchYawrateThrust, queue_size=3
        )

    def _init_subscriber(self):
        rospy.Subscriber("odometry_sensor1/odometry", Odometry, self._odometry_callback)
        rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self._environment_callback
        )
        rospy.Subscriber("training/reset_simulation", Bool, self._reset_callback)
        rospy.Subscriber(
            "training_action_interface/action_to_interface",
            Action,
            self._action_callback,
        )

        rospy.Subscriber(
            "training_action_interface/control_effort/v_z",
            Float64,
            self._vz_effort_callback,
        )
        rospy.Subscriber(
            "training_action_interface/control_effort/yaw",
            Float64,
            self._yaw_effort_callback,
        )
        rospy.Subscriber(
            "landing_simulation/relative_moving_platform_drone/state/pose",
            PoseStamped,
            self._pose_callback,
        )
        rospy.Subscriber(
            "landing_simulation/relative_moving_platform_drone/state/twist",
            TwistStamped,
            self._twist_callback,
        )
        rospy.Subscriber(
            "/moving_platform/contact", ContactsState, self._read_contact_state_callback
        )

    def publish_obs(self, drone_tf, mp_tf):
        # Update the moving platform state
        pose, u, v = self.moving_platform.update()
        self._publish_trajectory(pose, u, v)

        # Compute and publish the relative position and velocity between drone and platform
        rel_pos, rel_vel = self.utils.get_relative_state(drone_tf, mp_tf)
        self.relative_pos_pub.publish(rel_pos)
        self.relative_vel_pub.publish(rel_vel)

        # Compute and publish the relative roll, pitch, and yaw angles
        rpy_angles = np.degrees(
            euler_from_quaternion(
                [
                    rel_pos.pose.orientation.x,
                    rel_pos.pose.orientation.y,
                    rel_pos.pose.orientation.z,
                    rel_pos.pose.orientation.w,
                ]
            )
        )
        rpy_msg = Float64MultiArray(data=rpy_angles)
        self.relative_rpy_pub.publish(rpy_msg)

        # Compute and publish observation data
        obs_msg = self.utils.get_observation(rel_pos, rel_vel, self.mp_contact_occured)
        self.observation_pub.publish(obs_msg)

        # return obs_msg

    def _publish_trajectory(self, pose, u, v):
        gazebo_msg = ModelState()
        gazebo_msg.model_name = "moving_platform"
        gazebo_msg.reference_frame = "ground_plane"
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
            odom_msg.pose.pose.position.z,
        )

        transform_msg = self.utils.broadcast_stability_tf(
            odom_msg.header.frame_id, yaw, pos
        )
        self.br.sendTransform(transform_msg)

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

        self.drone_wf.twist.twist.linear.vector = msg.twist[drone_index].linear  # type: ignore
        self.drone_wf.twist.twist.angular.vector = msg.twist[drone_index].angular  # type: ignore
        self.mp_wf.twist.twist.linear.vector = msg.twist[mp_index].linear  # type: ignore
        self.mp_wf.twist.twist.angular.vector = msg.twist[mp_index].angular  # type: ignore

    def _vz_effort_callback(self, msg: Float64):
        self.effort.vz_effort = msg.data

    def _yaw_effort_callback(self, msg: Float64):
        self.effort.yaw_effort = msg.data

    def _pose_callback(self, msg: PoseStamped):
        """
        Callback for pose messages.

        Converts the quaternion orientation to Euler angles to extract yaw,
        updates the internal yaw state, and publishes the new yaw state for the c++ controllers.

        Args:
            msg (PoseStamped): Contains the UAV's pose data (position and orientation).
        """

        orient = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])
        self.effort.yaw_state = yaw
        self._pub_yaw_state.publish(Float64(data=yaw))

    def _twist_callback(self, msg: TwistStamped):
        self.effort.vz_state = -msg.twist.linear.z
        self._pub_vz_state.publish(Float64(data=self.effort.vz_state))

    def _reset_callback(self, msg):
        """
        Callback for reset command. If the reset flag is received,
        the simulation reset request is stored in observation data.

        :param msg: Bool message indicating whether to reset the simulation.
        """
        if msg.data:
            self.t = np.random.uniform(
                0,
                self.t_max,
            )
            self.moving_platform.reset_time(self.t)
            self.observation_data.request_simulation_reset = True

            rospy.loginfo("Reset initiated")
            self.pid_setpoints = PID_Setpoints(0, 0, 0, 0)
            rospy.loginfo("New action values: %s", self.pid_setpoints)

    def _action_callback(self, msg: Action):
        """
        Callback for receiving control actions from the training interface.

        :param msg: Action message containing roll, pitch, yaw, and vertical velocity commands.
        """
        for setpoint in self.pid_setpoints.__annotations__:
            setattr(self.pid_setpoints, setpoint, getattr(msg, setpoint))
        self._publish_setpoints()

    def _read_contact_state_callback(self, msg: ContactsState):
        """Function checks if the contact sensor on top of the moving platform sends values. If yes, a flag is set to true."""
        if msg.states:
            self.mp_contact_occured = True

    def _publish_setpoints(self):
        self._pub_vz_setpoint.publish(Float64(data=self.pid_setpoints.v_z))
        self._pub_yaw_setpoint.publish(Float64(data=self.pid_setpoints.yaw))

    def _extract_yaw(self, orientation):
        """
        Extracts the yaw (rotation around Z-axis) from a quaternion.

        :param orientation: Quaternion representing the object's orientation.
        :return: Yaw angle in radians.
        """
        return euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )[2]

    def _publish_rpy_thrust(self):
        """
        Publishes the composite command message for roll, pitch, yaw rate, and thrust.

        Constructs a RollPitchYawrateThrust message using the current action setpoints and control efforts,
        and publishes it to the designated topic.
        """
        cmd = RollPitchYawrateThrust()
        cmd.roll = self.pid_setpoints.roll
        cmd.pitch = self.pid_setpoints.pitch
        cmd.yaw_rate = self.effort.yaw_effort
        thrust_vector = Vector3()
        thrust_vector.z = self.effort.vz_effort
        cmd.thrust = thrust_vector
        self._pub_rpy_thrust.publish(cmd)

    def run(self):
        """
        Main loop that continuously updates and publishes the state of the drone and moving platform.
        """
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            # Transform drone and moving platform states into the target frame
            check_trans, drone_tf, mp_tf = self.utils.transform_world_to_target_frame(
                drone_wf=self.drone_wf,
                mp_wf=self.mp_wf,
                drone_tf=self.drone_tf,
                mp_tf=self.mp_tf,
                buffer=self.tfBuffer,
            )
            if not check_trans:
                rate.sleep()
                continue

            self.publish_obs(drone_tf, mp_tf)

            # Apply control commands to the drone publishing towards c++ controllers
            self._publish_rpy_thrust()
            rate.sleep()

    def _reset_random_seed(self, req):
        """
        Function handles the service request to reset the seed for the random number generator.

        :param req: ResetRandomSeed request message containing the new seed value.
        :return: ResetRandomSeed response message.
        """
        seed = None if req.seed == "None" else int(req.seed)
        rospy.loginfo("Set seed for random initial values to %s", seed)
        np.random.seed(seed)
        return ResetRandomSeedResponse()


if __name__ == "__main__":
    node = ManagerNode()
    node.run()
