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

class ObservationUtils():
    """
    Utility class for computing relative states, observation, and managing transformations.
    """

    def __init__(self, target_frame, world_frame, noise_pos_sd=0, noise_vel_sd=0, filter=Union[KalmanFilter3D, LowPassFilter3D, None]):
        self.noise_pos_sd = noise_pos_sd
        self.noise_vel_sd = noise_vel_sd
        self.last_velocity = None
        self.last_timestep = None
        self.filter = filter
        self.target_frame = target_frame
        self.world_frame = world_frame
    
        
    
    def get_relative_state(self, drone_tf, mp_tf):
        """
        Computes the relative velocity, position, and orientation between the drone and the moving platform.

        Args:
            drone_tf (PoseStamped): The drone's state in the target frame.
            mp_tf (PoseStamped): The moving platform's state in the target frame.

        Returns:
            tuple(PoseStamped, TwistStamped): The relative position and velocity.
        """

        current_time = rospy.Time.now()

        rel_vel = self._get_relative_velocity(drone_tf, mp_tf, current_time)

        rel_pos = self._get_relative_position(drone_tf, mp_tf, current_time)

        rel_pos.pose.orientation = self._get_relative_orientation(drone_tf, mp_tf)

        return rel_pos, rel_vel
    

    def get_observation_from_env(self, rel_pos, rel_vel, obs_data):
        """
        Constructs an observation message including position, velocity, yaw, and acceleration.

        Args:
            rel_pos (PoseStamped): Relative position between drone and platform.
            rel_vel (TwistStamped): Relative velocity between drone and platform.
            obs_data (ObservationRelativeStateData): Additional action setpoints and frame validation.

        Returns:
            ObservationRelativeState: The generated observation message.
        """

        obs = ObservationRelativeState()
        current_time = rospy.Time.now() 
        obs.header.stamp = current_time

        pos_fields = ["rel_p_x", "rel_p_y", "rel_p_z"]
        vel_fields = ["rel_v_x", "rel_v_y", "rel_v_z"]

        pos_array = np.array([
            rel_pos.pose.position.x,
            rel_pos.pose.position.y,
            rel_pos.pose.position.z
        ])
        
        vel_array = np.array([
            rel_vel.twist.linear.x,
            rel_vel.twist.linear.y,
            rel_vel.twist.linear.z
        ])

        noisy_pos = pos_array + np.random.normal(0, self.noise_pos_sd, 3)
        noisy_vel = vel_array + np.random.normal(0, self.noise_vel_sd, 3)

        self.numpy_to_msg(obs, pos_fields, noisy_pos)
        self.numpy_to_msg(obs, vel_fields, noisy_vel)

        _, _, yaw = euler_from_quaternion([
            rel_pos.pose.orientation.x,
            rel_pos.pose.orientation.y,
            rel_pos.pose.orientation.z,
            rel_pos.pose.orientation.w
        ])
        obs.rel_yaw = yaw

        timestep = current_time.to_sec()
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
    


    def msg_to_numpy_vector(self, msg, attributes):
        """
        Converts a ROS message field into a NumPy array.

        Args:
            msg (ROS message): The input ROS message object.
            attributes (list[str]): The list of attributes to extract.

        Returns:
            np.array: A NumPy array of extracted values.
        """

        return np.array([getattr(msg, attr) for attr in attributes])


    def numpy_to_msg(self, msg, fields, values):
        """
        Updates a ROS message with values from a NumPy array.

        Args:
            msg (ROS message): The message to update.
            fields (list[str]): The list of message fields to update.
            values (np.array): The corresponding values.
        """

        for field, value in zip(fields, values):
            setattr(msg, field, value)


    def _get_relative_velocity(self, drone_state, mp_state, timestamp):
        """Computes relative velocity between the drone and the moving platform."""
        rel_vel = TwistStamped()
        rel_vel.header.stamp = timestamp
        rel_vel.header.frame_id = self.target_frame

        mp_lin_vel = self.msg_to_numpy_vector(mp_state.twist.twist.linear.vector, ["x", "y", "z"])
        drone_lin_vel = self.msg_to_numpy_vector(drone_state.twist.twist.linear.vector, ["x", "y", "z"])
        
        mp_ang_vel = self.msg_to_numpy_vector(mp_state.twist.twist.angular.vector, ["x", "y", "z"])
        drone_ang_vel = self.msg_to_numpy_vector(drone_state.twist.twist.angular.vector, ["x", "y", "z"])

        rel_lin_vel = mp_lin_vel - drone_lin_vel
        rel_ang_vel = mp_ang_vel - drone_ang_vel

        rel_vel.twist.linear.x, rel_vel.twist.linear.y, rel_vel.twist.linear.z = rel_lin_vel
        rel_vel.twist.angular.x, rel_vel.twist.angular.y, rel_vel.twist.angular.z = rel_ang_vel

        return rel_vel


    def _get_relative_position(self, drone_state, mp_state, timestamp):
        """Computes relative position between the drone and the moving platform."""
        rel_pos = PoseStamped()
        rel_pos.header.stamp = timestamp
        rel_pos.header.frame_id = self.target_frame

        mp_pos = self.msg_to_numpy_vector(mp_state.pose.pose.position, ["x", "y", "z"])
        drone_pos = self.msg_to_numpy_vector(drone_state.pose.pose.position, ["x", "y", "z"])

        rel_pos_vals = mp_pos - drone_pos

        rel_pos.pose.position.x, rel_pos.pose.position.y, rel_pos.pose.position.z = rel_pos_vals

        return rel_pos

    def _get_relative_orientation(self, drone_state, mp_state):
        """Computes relative orientation using quaternions."""
        q_mp = self.msg_to_numpy_vector(mp_state.pose.pose.orientation, ["x", "y", "z", "w"])
        q_drone = self.msg_to_numpy_vector(drone_state.pose.pose.orientation, ["x", "y", "z", "w"])

        q_rel = quaternion_multiply(q_drone, quaternion_inverse(q_mp))
        
        return Quaternion(*q_rel) 