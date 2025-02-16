#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
from std_msgs.msg import Float64MultiArray, Bool, Header
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import ModelStates
from dql_multirotor_landing.msg import LandingSimulationObjectState, ObservationRelativeState, Action
from dql_multirotor_landing.parameters import Parameters
# from dql_multirotor_landing.moving_platform import MovingPlatformNode
from dql_multirotor_landing.moving_platform import MovingPlatformNode
from dql_multirotor_landing.publish_stability import StabilityFramePublisher
from dql_multirotor_landing.interface import ActionInterface
from dql_multirotor_landing.filters import KalmanFilter3D
from dql_multirotor_landing.observation_layer import ObservationLayer, ObservationRelativeStateData
from tf.transformations import euler_from_quaternion
import numpy as np
from copy import deepcopy



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


class ManagerLayer():
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

        self.utils = ObservationLayer(target_frame=self.target_frame, world_frame=self.gazebo_frame, noise_pos_sd=self.noise_pos_sd, noise_vel_sd=self.noise_vel_sd, filter=self.kalman_filter)


        self.drone_state_original = State()
        self.mp_state_original = State()
        self.drone_state_target = State()
        self.mp_state_target = State()
        self.observation_data = ObservationRelativeStateData()

        self._init_publisher()
        self._init_subscriber()

        # TF
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)

        self.prev_rel_vel = None
        self.prev_time = None   

        self.stability_frame_pub = StabilityFramePublisher(self.drone_name)
        self.moving_platform = MovingPlatformNode()
        self.action_interface = ActionInterface()



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
            
            check_trans, drone_tf, mp_tf = self.utils.transform_world_to_target_frame(
                                         drone_wf=self.drone_state_original, 
                                         mp_wf=self.mp_state_original, 
                                         drone_tf=self.drone_state_target, 
                                         mp_tf=self.mp_state_target,
                                         buffer=self.tfBuffer)
            
            if not check_trans:
                rate.sleep()
                continue

            self.moving_platform.update()
            rel_pos, rel_vel = self.utils.get_relative_state(drone_tf, mp_tf, self.target_frame)
            self.relative_pos_pub.publish(rel_pos)
            self.relative_vel_pub.publish(rel_vel)
            rpy = Float64MultiArray()
            rpy.data = (180/np.pi) * np.array(euler_from_quaternion([rel_pos.pose.orientation.x,
                                                                      rel_pos.pose.orientation.y,
                                                                      rel_pos.pose.orientation.z,
                                                                      rel_pos.pose.orientation.w]))
            self.relative_rpy_pub.publish(rpy)
            # Pubblica gli stati per la simulazione (in frame target e world)
            self.drone_state_pub.publish(self.compute_landing_state_msg(drone_tf))
            self.mp_state_pub.publish(self.compute_landing_state_msg(mp_tf))
            self.drone_state_world_pub.publish(self.compute_landing_state_msg(self.drone_state_original))
            self.mp_state_world_pub.publish(self.compute_landing_state_msg(self.mp_state_original))
            # Calcola e pubblica il messaggio di osservazione
            obs_msg = self.utils.get_observation(rel_pos, rel_vel, self.observation_data)
            self.observation_pub.publish(obs_msg)
            # Pubblica anche il messaggio di azione (i setpoint attuali)
            act = Action()
            act.roll = self.observation_data.action_setpoints.roll
            act.pitch = self.observation_data.action_setpoints.pitch
            act.yaw = self.observation_data.action_setpoints.yaw
            act.v_z = self.observation_data.action_setpoints.v_z
            act.header.stamp = rospy.Time.now()
            if not self.observation_data.check_frames():
                act.header.frame_id = 'FAILED TO COMPUTE ACTION'
            else:
                act.header.frame_id = rel_pos.header.frame_id
            self.observation_action_pub.publish(act)

            self.action_interface.update()
            rate.sleep()

if __name__ == '__main__':
    node = ManagerLayer()
    node.run()
