#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
from std_msgs.msg import Float64, Float64MultiArray, Bool, Header
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped, Quaternion, Vector3
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import ModelStates
from dql_multirotor_landing.msg import LandingSimulationObjectState, ObservationRelativeState, Action
from dql_multirotor_landing.parameters import Parameters
# from training_q_learning.moving_platform import MovingPlatformNode
from dql_multirotor_landing.moving_platform import MovingPlatformNode
from training_q_learning.utils import KalmanFilter3D, Utils
from tf.transformations import euler_from_quaternion, quaternion_multiply, quaternion_inverse
import numpy as np
from copy import deepcopy


class PoseTwistAccelerationState():
    def __init__(self):
        self.pose = PoseStamped()
        self.pose.header.frame_id = 'world'
        self.twist = TwistStamped()
        self.twist.header.frame_id = 'world'
        self.twist.twist.linear = Vector3Stamped()
        self.twist.twist.angular = Vector3Stamped()
        self.linear_acceleration = Vector3Stamped()
        self.linear_acceleration.header.frame_id = 'world'


class ObservationRelativeStateData():
    def __init__(self):
        self.relative_position = PoseStamped()
        self.relative_velocity = TwistStamped()
        self.relative_acceleration = Imu()
        self.action_setpoints = Action()
        self.request_simulation_reset = False
    def check_frames(self):
        frames = [self.relative_position.header.frame_id,
                  self.relative_velocity.header.frame_id,
                  self.relative_acceleration.header.frame_id]
        return all(f == frames[0] for f in frames)



class ManagerLayer():
    def __init__(self):
        rospy.init_node('central_logic_node')
        self.parameters = Parameters()
        ns = rospy.get_namespace()
        self.node_name = 'central_logic_node'
        self.drone_name = rospy.get_param(ns + self.node_name + '/drone_name', 'hummingbird')
        self.publish_rate = float(rospy.get_param(ns + self.node_name + '/publish_rate', '100'))
        self.std_rel_p_x = float(rospy.get_param(ns + self.node_name + '/std_rel_p_x', '0.25'))
        self.std_rel_p_y = float(rospy.get_param(ns + self.node_name + '/std_rel_p_y', '0.25'))
        self.std_rel_p_z = float(rospy.get_param(ns + self.node_name + '/std_rel_p_z', '0.1'))
        self.std_rel_v_x = float(rospy.get_param(ns + self.node_name + '/std_rel_v_x', '0.1'))
        self.std_rel_v_y = float(rospy.get_param(ns + self.node_name + '/std_rel_v_y', '0.1'))
        self.std_rel_v_z = float(rospy.get_param(ns + self.node_name + '/std_rel_v_z', '0.05'))
        
        self.kalman_filter = KalmanFilter3D(
            process_variance=1e-4, 
            measurement_variance=[self.std_rel_v_x, self.std_rel_v_y, self.std_rel_v_z]
        )
        self.utils = Utils()

        self.gazebo_frame = 'world'
        self.target_frame = self.drone_name + '/stability_axes'

        self.drone_state_original = PoseTwistAccelerationState()
        self.mp_state_original = PoseTwistAccelerationState()
        self.drone_state_target = PoseTwistAccelerationState()
        self.mp_state_target = PoseTwistAccelerationState()
        self.observation_data = ObservationRelativeStateData()

        self._init_publisher()
        self._init_subscriber()

        # TF
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)

        self.prev_rel_vel = None
        self.prev_time = None   

        self.moving_platform = MovingPlatformNode()



    def _init_publisher(self):
        self.relative_vel_pub = rospy.Publisher('landing_simulation/relative_moving_platform_drone/state/twist', TwistStamped, queue_size=0)
        self.relative_pos_pub = rospy.Publisher('landing_simulation/relative_moving_platform_drone/state/pose', PoseStamped, queue_size=0)
        self.relative_rpy_pub = rospy.Publisher('landing_simulation/relative_moving_platform_drone/debug_target_frame/roll_pitch_yaw', Float64MultiArray, queue_size=0)
        self.drone_state_pub = rospy.Publisher('landing_simulation/drone/state', LandingSimulationObjectState, queue_size=0)
        self.mp_state_pub = rospy.Publisher('landing_simulation/moving_platform/state', LandingSimulationObjectState, queue_size=0)
        self.drone_state_world_pub = rospy.Publisher('landing_simulation/world_frame/drone/state', LandingSimulationObjectState, queue_size=0)
        self.mp_state_world_pub = rospy.Publisher('landing_simulation/world_frame/moving_platform/state', LandingSimulationObjectState, queue_size=0)
        self.observation_pub = rospy.Publisher('training_observation_interface/observations', ObservationRelativeState, queue_size=0)
        self.observation_action_pub = rospy.Publisher('training_observation_interface/observations_actions', Action, queue_size=0)

    def _init_subscriber(self):
        self.model_states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_cb)
        self.reset_sub = rospy.Subscriber('training/reset_simulation', Bool, self.reset_cb)
        self.action_sub = rospy.Subscriber('training_action_interface/action_to_interface', Action, self.action_cb)

    def model_states_cb(self, msg):
        try:
            drone_idx = msg.name.index(self.drone_name)
            mp_idx = msg.name.index("moving_platform")
        except ValueError:
            rospy.logwarn("Drone o moving_platform non trovati in model_states.")
            return
        
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.gazebo_frame

        stamped_pose_drone = PoseStamped()
        stamped_pose_drone.header = header 
        stamped_pose_drone.pose = msg.pose[drone_idx]
        self.drone_state_original.pose = stamped_pose_drone

        stamped_pose_mp = PoseStamped()
        stamped_pose_mp.header = header
        stamped_pose_mp.pose = msg.pose[mp_idx]
        self.mp_state_original.pose = stamped_pose_mp

        self.drone_state_original.twist.twist.linear.vector = msg.twist[drone_idx].linear
        self.drone_state_original.twist.twist.angular.vector = msg.twist[drone_idx].angular
        self.mp_state_original.twist.twist.linear.vector = msg.twist[mp_idx].linear
        self.mp_state_original.twist.twist.angular.vector = msg.twist[mp_idx].angular


    def reset_cb(self, msg):
        if msg.data:
            self.observation_data.request_simulation_reset = True

    def action_cb(self, msg):
        self.observation_data.action_setpoints = msg

    def transform_states(self):
        try:
            trans = self.tfBuffer.lookup_transform(self.target_frame, self.gazebo_frame, rospy.Time(0), rospy.Duration(1.0))
        except Exception as e:
            rospy.logwarn("Trasformazione non trovata: " + str(e))
            return False
        self.drone_state_target.pose = tf2_geometry_msgs.do_transform_pose(self.drone_state_original.pose, trans)
        self.mp_state_target.pose = tf2_geometry_msgs.do_transform_pose(self.mp_state_original.pose, trans)
        self.drone_state_target.twist.twist.linear = tf2_geometry_msgs.do_transform_vector3(self.drone_state_original.twist.twist.linear, trans)
        self.drone_state_target.twist.twist.angular = tf2_geometry_msgs.do_transform_vector3(self.drone_state_original.twist.twist.angular, trans)
        self.mp_state_target.twist.twist.linear = tf2_geometry_msgs.do_transform_vector3(self.mp_state_original.twist.twist.linear, trans)
        self.mp_state_target.twist.twist.angular = tf2_geometry_msgs.do_transform_vector3(self.mp_state_original.twist.twist.angular, trans)

        self.drone_state_target.pose.header.frame_id = self.target_frame
        self.mp_state_target.pose.header.frame_id = self.target_frame
        self.drone_state_target.twist.header.frame_id = self.target_frame
        self.mp_state_target.twist.header.frame_id = self.target_frame
        return True


    def compute_observation(self, rel_pos, rel_vel):
        obs = ObservationRelativeState()
        obs.header.stamp = rospy.Time.now()
        obs.rel_p_x = rel_pos.pose.position.x + np.random.normal(0, self.std_rel_p_x)
        obs.rel_p_y = rel_pos.pose.position.y + np.random.normal(0, self.std_rel_p_y)
        obs.rel_p_z = rel_pos.pose.position.z + np.random.normal(0, self.std_rel_p_z)
        obs.rel_v_x = rel_vel.twist.linear.x + np.random.normal(0, self.std_rel_v_x)
        obs.rel_v_y = rel_vel.twist.linear.y + np.random.normal(0, self.std_rel_v_y)
        obs.rel_v_z = rel_vel.twist.linear.z + np.random.normal(0, self.std_rel_v_z)
        _, _, yaw = euler_from_quaternion([rel_pos.pose.orientation.x, rel_pos.pose.orientation.y,
                                                      rel_pos.pose.orientation.z, rel_pos.pose.orientation.w])
        obs.rel_yaw = yaw

        timestep = rospy.Time.now().to_sec()
        current_rel_v = rel_vel.twist.linear

        if self.prev_rel_vel is None or self.prev_time is None:
            self.prev_rel_vel = current_rel_v
            self.prev_time = timestep
            accel = Vector3Stamped()
            accel.vector.x = 0
            accel.vector.y = 0
            accel.vector.z = 0
        else:
            accel = self.kalman_filter.filter(
                current_rel_v=current_rel_v,
                timestep=timestep,
                last_vel=self.prev_rel_vel,
                last_timestep=self.prev_time
            )
        
        obs.rel_a_x = accel.vector.x
        obs.rel_a_y = accel.vector.y
        obs.rel_a_z = accel.vector.z
        obs.roll = self.observation_data.action_setpoints.roll
        obs.pitch = self.observation_data.action_setpoints.pitch
        obs.yaw = self.observation_data.action_setpoints.yaw
        obs.v_z = self.observation_data.action_setpoints.v_z
        obs.roll_rate = float("nan")
        obs.pitch_rate = float("nan")
        obs.yaw_rate = float("nan")
        if not self.observation_data.check_frames():
            obs.header.frame_id = 'FAILED'
        else:
            obs.header.frame_id = rel_pos.header.frame_id
        return obs

    def compute_landing_state_msg(self, state):
        msg = LandingSimulationObjectState()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = state.pose.header.frame_id
        msg.pose = deepcopy(state.pose)
        msg.twist = deepcopy(state.twist)
        msg.linear_acceleration = deepcopy(state.linear_acceleration)
        return msg

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():

            if not self.transform_states():
                rate.sleep()
                continue
            self.moving_platform.update()
            rel_pos, rel_vel = self.utils.compute_relative_state(self.drone_state_target, self.mp_state_target, self.target_frame)
            self.relative_pos_pub.publish(rel_pos)
            self.relative_vel_pub.publish(rel_vel)
            rpy = Float64MultiArray()
            rpy.data = (180/np.pi) * np.array(euler_from_quaternion([rel_pos.pose.orientation.x,
                                                                      rel_pos.pose.orientation.y,
                                                                      rel_pos.pose.orientation.z,
                                                                      rel_pos.pose.orientation.w]))
            self.relative_rpy_pub.publish(rpy)
            self.drone_state_pub.publish(self.compute_landing_state_msg(self.drone_state_target))
            self.mp_state_pub.publish(self.compute_landing_state_msg(self.mp_state_target))
            self.drone_state_world_pub.publish(self.compute_landing_state_msg(self.drone_state_original))
            self.mp_state_world_pub.publish(self.compute_landing_state_msg(self.mp_state_original))
            obs_msg = self.compute_observation(rel_pos, rel_vel)
            self.observation_pub.publish(obs_msg)
            act = Action()
            act.roll = self.observation_data.action_setpoints.roll
            act.pitch = self.observation_data.action_setpoints.pitch
            act.yaw = self.observation_data.action_setpoints.yaw
            act.v_z = self.observation_data.action_setpoints.v_z
            act.header.stamp = rospy.Time.now()
            if not self.observation_data.check_frames():
                act.header.frame_id = 'FAILED'
            else:
                act.header.frame_id = rel_pos.header.frame_id
            self.observation_action_pub.publish(act)

            self.action_interface.update()
            rate.sleep()

if __name__ == '__main__':
    node = ManagerLayer()
    node.run()



