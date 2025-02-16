import rospy
import numpy as np
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped, Quaternion
from tf.transformations import quaternion_multiply, quaternion_inverse, euler_from_quaternion
from sensor_msgs.msg import Imu
from tf2_geometry_msgs import do_transform_pose, do_transform_vector3
from training_q_learning.msg import ObservationRelativeState, Action
from typing import Union
from training_q_learning.filters import KalmanFilter3D, LowPassFilter3D


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

class ObservationLayer():
    
    def __init__(self, target_frame, world_frame, noise_pos_sd=0, noise_vel_sd=0, filter=Union[KalmanFilter3D, LowPassFilter3D, None]):
        self.noise_pos_sd = noise_pos_sd
        self.noise_vel_sd = noise_vel_sd
        self.last_velocity = None
        self.last_timestep = None
        self.filter = filter
        self.target_frame = target_frame
        self.world_frame = world_frame  
        
    
    def get_relative_state(self, drone_state_target, mp_state_target, target_frame):
        # Calcola velocità e posizione relative (in frame target)
        rel_vel = TwistStamped()
        rel_vel.header.stamp = rospy.Time.now()
        rel_vel.header.frame_id = target_frame
        rel_vel.twist.linear.x = mp_state_target.twist.twist.linear.vector.x - drone_state_target.twist.twist.linear.vector.x
        rel_vel.twist.linear.y = mp_state_target.twist.twist.linear.vector.y - drone_state_target.twist.twist.linear.vector.y
        rel_vel.twist.linear.z = mp_state_target.twist.twist.linear.vector.z - drone_state_target.twist.twist.linear.vector.z
        rel_vel.twist.angular.x = mp_state_target.twist.twist.angular.vector.x - drone_state_target.twist.twist.angular.vector.x
        rel_vel.twist.angular.y = mp_state_target.twist.twist.angular.vector.y - drone_state_target.twist.twist.angular.vector.y
        rel_vel.twist.angular.z = mp_state_target.twist.twist.angular.vector.z - drone_state_target.twist.twist.angular.vector.z

        rel_pos = PoseStamped()
        rel_pos.header.stamp = rospy.Time.now()
        rel_pos.header.frame_id = target_frame
        rel_pos.pose.position.x = mp_state_target.pose.pose.position.x - drone_state_target.pose.pose.position.x
        rel_pos.pose.position.y = mp_state_target.pose.pose.position.y - drone_state_target.pose.pose.position.y
        rel_pos.pose.position.z = mp_state_target.pose.pose.position.z - drone_state_target.pose.pose.position.z

        q_mp = [mp_state_target.pose.pose.orientation.x, mp_state_target.pose.pose.orientation.y,
                mp_state_target.pose.pose.orientation.z, mp_state_target.pose.pose.orientation.w]
        q_drone = [drone_state_target.pose.pose.orientation.x, drone_state_target.pose.pose.orientation.y,
                   drone_state_target.pose.pose.orientation.z, drone_state_target.pose.pose.orientation.w]
        q_rel = quaternion_multiply(q_drone, quaternion_inverse(q_mp))
        rel_pos.pose.orientation = Quaternion(*q_rel)
        return rel_pos, rel_vel
    

    def get_observation(self, rel_pos, rel_vel, obs_data):
        obs = ObservationRelativeState()
        obs.header.stamp = rospy.Time.now()
        noise_pos = np.random.normal(0, self.noise_pos_sd, 3)
        noise_vel = np.random.normal(0, self.noise_vel_sd, 3)
        obs.rel_p_x = rel_pos.pose.position.x + noise_pos[0]
        obs.rel_p_y = rel_pos.pose.position.y + noise_pos[1]
        obs.rel_p_z = rel_pos.pose.position.z + noise_pos[2]
        obs.rel_v_x = rel_vel.twist.linear.x + noise_vel[0]
        obs.rel_v_y = rel_vel.twist.linear.y + noise_vel[1]
        obs.rel_v_z = rel_vel.twist.linear.z + noise_vel[2]
        _, _, yaw = euler_from_quaternion([rel_pos.pose.orientation.x, rel_pos.pose.orientation.y,
                                                      rel_pos.pose.orientation.z, rel_pos.pose.orientation.w])
        obs.rel_yaw = yaw

        timestep = rospy.Time.now().to_sec()
        current_rel_v = rel_vel.twist.linear

        if self.last_velocity is None or self.last_timestep is None:
            self.last_velocity = current_rel_v
            self.last_timestep = timestep
            accel = Vector3Stamped()
            accel.vector.x = 0
            accel.vector.y = 0
            accel.vector.z = 0
        else:
            accel = self.filter.filter(
                current_rel_v=current_rel_v,
                timestep=timestep,
                last_vel=self.last_velocity,
                last_timestep=self.last_timestep
            )
        
        obs.rel_a_x = accel.vector.x
        obs.rel_a_y = accel.vector.y
        obs.rel_a_z = accel.vector.z
        obs.roll = obs_data.action_setpoints.roll
        obs.pitch = obs_data.action_setpoints.pitch
        obs.yaw = obs_data.action_setpoints.yaw
        obs.v_z = obs_data.action_setpoints.v_z
        obs.roll_rate = float("nan")
        obs.pitch_rate = float("nan")
        obs.yaw_rate = float("nan")

        if not obs_data.check_frames():
            obs.header.frame_id = 'FAILED TO COMPUTE OBSERVATION'
        else:
            obs.header.frame_id = rel_pos.header.frame_id
        return obs


    def transform_world_to_target_frame(self, drone_wf, mp_wf, drone_tf, mp_tf, buffer):

        try:
            trans = buffer.lookup_transform(self.target_frame, self.world_frame, rospy.Time(0), rospy.Duration(1.0))
        except Exception as e:
            rospy.logwarn("Unable to lookup transform: " + str(e))
            return False, drone_tf, mp_tf
        
        drone_tf.pose = do_transform_pose(drone_wf.pose, trans)
        drone_tf.twist.twist.linear = do_transform_vector3(drone_wf.twist.twist.linear, trans)
        drone_tf.twist.twist.angular = do_transform_vector3(drone_wf.twist.twist.angular, trans)
        drone_tf.pose.header.frame_id = self.target_frame
        drone_tf.twist.header.frame_id = self.target_frame

        mp_tf.pose = do_transform_pose(mp_wf.pose, trans)
        mp_tf.twist.twist.linear = do_transform_vector3(mp_wf.twist.twist.linear, trans)
        mp_tf.twist.twist.angular = do_transform_vector3(mp_wf.twist.twist.angular, trans)
        mp_tf.pose.header.frame_id = self.target_frame
        mp_tf.twist.header.frame_id = self.target_frame

        return True, drone_tf, mp_tf