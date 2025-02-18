#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from training_q_learning.msg import Action
from mav_msgs.msg import RollPitchYawrateThrust
from training_q_learning.parameters import Parameters
from copy import deepcopy
from tf.transformations import euler_from_quaternion


class RotorsInterface_CPP:
    """
    RotorsInterface_CPP acts as a communication layer between our Python code and standard C++ nodes in the ROS system.
    
    This class subscribes to various sensor and command topics, processes incoming messages, and publishes
    control setpoints and composite commands. It bridges the gap between our Python implementation and
    C++ nodes by handling the necessary message transformations and topic communications.
    """
    
    def __init__(self):
        self._params = Parameters()
        self._default_actions = deepcopy(self._params.uav_parameters.initial_action_values)
        self._actions = deepcopy(self._default_actions)
        self._vz_effort = 0.0
        self._vz_state = 0.0
        self._yaw_effort = 0.0
        self._yaw_state = 0.0
        self._init_publishers()
        self._init_subscribers()



    def _init_publishers(self):
        self._pub_vz_setpoint = rospy.Publisher(
            'training_action_interface/setpoint/v_z', Float64, queue_size=3
        )
        self._pub_vz_state = rospy.Publisher(
            'training_action_interface/state/v_z', Float64, queue_size=3
        )
        self._pub_yaw_setpoint = rospy.Publisher(
            'training_action_interface/setpoint/yaw', Float64, queue_size=3
        )
        self._pub_yaw_state = rospy.Publisher(
            'training_action_interface/state/yaw', Float64, queue_size=3
        )
        self._pub_rpy_thrust = rospy.Publisher(
            'command/roll_pitch_yawrate_thrust', RollPitchYawrateThrust, queue_size=3
        )

    def _init_subscribers(self):
        rospy.Subscriber('training_action_interface/action_to_interface', Action, self._handle_action)
        rospy.Subscriber('training/reset_simulation', Bool, self._handle_reset)
        rospy.Subscriber('training_action_interface/control_effort/v_z', Float64, self._handle_vz_effort)
        rospy.Subscriber('training_action_interface/control_effort/yaw', Float64, self._handle_yaw_effort)
        rospy.Subscriber('landing_simulation/relative_moving_platform_drone/state/pose', PoseStamped, self._handle_pose)
        rospy.Subscriber('landing_simulation/relative_moving_platform_drone/state/twist', TwistStamped, self._handle_twist)

    def _handle_vz_effort(self, msg: Float64):
        self._vz_effort = msg.data

    def _handle_yaw_effort(self, msg: Float64):
        self._yaw_effort = msg.data

    def _handle_pose(self, msg: PoseStamped):
        """
        Callback for pose messages.
        
        Converts the quaternion orientation to Euler angles to extract yaw,
        updates the internal yaw state, and publishes the new yaw state.
        
        Args:
            msg (PoseStamped): Contains the UAV's pose data (position and orientation).
        """

        orient = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([orient.x, orient.y, orient.z, orient.w])
        self._yaw_state = yaw
        self._pub_yaw_state.publish(Float64(data=yaw))

    def _handle_twist(self, msg: TwistStamped):
        self._vz_state = -msg.twist.linear.z
        self._pub_vz_state.publish(Float64(data=self._vz_state))

    def _handle_action(self, msg: Action):
        for key in self._actions.keys():
            self._actions[key] = getattr(msg, key)
        self._publish_setpoints()

    def _handle_reset(self, msg: Bool):
        if msg.data:
            rospy.loginfo("Reset initiated")
            self._actions = deepcopy(self._default_actions)
            rospy.loginfo("New action values: %s", self._actions)

    def _publish_setpoints(self):
        self._pub_vz_setpoint.publish(Float64(data=self._actions["v_z"]))
        self._pub_yaw_setpoint.publish(Float64(data=self._actions["yaw"]))

    def _publish_rpy_thrust(self):
        """
        Publishes the composite command message for roll, pitch, yaw rate, and thrust.
        
        Constructs a RollPitchYawrateThrust message using the current action setpoints and control efforts,
        and publishes it to the designated topic.
        """
        cmd = RollPitchYawrateThrust()
        cmd.roll = self._actions["roll"]
        cmd.pitch = self._actions["pitch"]
        cmd.yaw_rate = self._yaw_effort
        thrust_vector = Vector3()
        thrust_vector.z = self._vz_effort
        cmd.thrust = thrust_vector
        self._pub_rpy_thrust.publish(cmd)

    def update(self):
        """
        Updates the UAV control interface by publishing the current roll, pitch, yaw rate, and thrust commands.
        This method should be called periodically to ensure the UAV receives the latest control commands.
        """
        self._publish_rpy_thrust()
