import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import (
    PoseStamped,
    Quaternion,
    TwistStamped,
    Vector3,
    Vector3Stamped,
)
from std_msgs.msg import Header
from tf.transformations import (
    quaternion_from_euler,
    quaternion_inverse,
    quaternion_multiply,
)
from tf2_geometry_msgs import do_transform_pose, do_transform_vector3

from dql_multirotor_landing.filters import KalmanFilter3D
from dql_multirotor_landing.msg import Observation

class ObservationData:
    def __init__(self):
        self.relative_position = PoseStamped()
        self.relative_velocity = TwistStamped()
        self.request_simulation_reset = False


DEBUG = False


class ObservationUtils:
    """
    Utility class for computing relative states, observation, and managing transformations.
    """

    def __init__(
        self,
        drone_name,
        target_frame,
        world_frame,
        noise_pos_sd=0.0,
        noise_vel_sd=0.0,
        filter=KalmanFilter3D(),
    ):
        self.noise_pos_sd = noise_pos_sd
        self.noise_vel_sd = noise_vel_sd
        self.last_velocity = None
        self.last_timestep = None
        self.filter = filter
        self.target_frame = target_frame
        self.world_frame = world_frame
        self.drone_name = drone_name

    def broadcast_stability_tf(self, source_frame, yaw, pos):
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

        return transform_msg

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

    def get_observation(self, rel_pos, rel_vel, contact):
        """
        Constructs an observation message including position, velocity, yaw, and acceleration.

        Args:
            rel_pos (PoseStamped): Relative position between drone and platform.
            rel_vel (TwistStamped): Relative velocity between drone and platform.

        Returns:
            Observation: The generated observation message.
        """

        obs = Observation()
        current_time = rospy.Time.now()
        obs.header.stamp = current_time
        obs.header.frame_id = rel_pos.header.frame_id

        fields = ["rel_p_x", "rel_p_y", "rel_p_z", "rel_v_x", "rel_v_y", "rel_v_z"]

        pos_array = np.array(
            [rel_pos.pose.position.x, rel_pos.pose.position.y, rel_pos.pose.position.z]
        )

        vel_array = np.array(
            [rel_vel.twist.linear.x, rel_vel.twist.linear.y, rel_vel.twist.linear.z]
        )

        # Aggiunta del rumore
        noisy_pos = pos_array + np.random.normal(0, self.noise_pos_sd, 3)
        noisy_vel = vel_array + np.random.normal(0, self.noise_vel_sd, 3)
        noisy = np.concatenate((noisy_pos, noisy_vel))

        for field, value in zip(fields, noisy):
            setattr(obs, field, value)

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
                last_timestep=self.last_timestep,
            )  # type: ignore

        obs.rel_a_x = accel.vector.x
        obs.rel_a_y = accel.vector.y
        obs.rel_a_z = accel.vector.z

        obs.contact = contact

        return obs

    def transform_world_to_target_frame(self, drone_wf, mp_wf, drone_tf, mp_tf, buffer):
        try:
            trans = buffer.lookup_transform(
                self.target_frame, self.world_frame, rospy.Time(0), rospy.Duration(1)
            )
        except Exception as e:
            if DEBUG:
                rospy.logwarn("Unable to lookup transform: " + str(e))
            return False, drone_tf, mp_tf

        drone_tf.pose = do_transform_pose(drone_wf.pose, trans)
        drone_tf.twist.twist.linear = do_transform_vector3(
            drone_wf.twist.twist.linear, trans
        )
        drone_tf.twist.twist.angular = do_transform_vector3(
            drone_wf.twist.twist.angular, trans
        )
        drone_tf.pose.header.frame_id = self.target_frame
        drone_tf.twist.header.frame_id = self.target_frame

        mp_tf.pose = do_transform_pose(mp_wf.pose, trans)
        mp_tf.twist.twist.linear = do_transform_vector3(mp_wf.twist.twist.linear, trans)
        mp_tf.twist.twist.angular = do_transform_vector3(
            mp_wf.twist.twist.angular, trans
        )
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


    def _get_relative_velocity(self, drone_state, mp_state, timestamp):
        """Computes relative velocity between the drone and the moving platform."""
        rel_vel = TwistStamped()
        rel_vel.header.stamp = timestamp
        rel_vel.header.frame_id = self.target_frame

        mp_lin_vel = self.msg_to_numpy_vector(
            mp_state.twist.twist.linear.vector, ["x", "y", "z"]
        )
        drone_lin_vel = self.msg_to_numpy_vector(
            drone_state.twist.twist.linear.vector, ["x", "y", "z"]
        )

        mp_ang_vel = self.msg_to_numpy_vector(
            mp_state.twist.twist.angular.vector, ["x", "y", "z"]
        )
        drone_ang_vel = self.msg_to_numpy_vector(
            drone_state.twist.twist.angular.vector, ["x", "y", "z"]
        )

        rel_lin_vel = mp_lin_vel - drone_lin_vel
        rel_ang_vel = mp_ang_vel - drone_ang_vel

        rel_vel.twist.linear.x, rel_vel.twist.linear.y, rel_vel.twist.linear.z = (
            rel_lin_vel  # type: ignore
        )
        rel_vel.twist.angular.x, rel_vel.twist.angular.y, rel_vel.twist.angular.z = (
            rel_ang_vel  # type: ignore
        )

        return rel_vel

    def _get_relative_position(self, drone_state, mp_state, timestamp):
        """Computes relative position between the drone and the moving platform."""
        rel_pos = PoseStamped()
        rel_pos.header.stamp = timestamp
        rel_pos.header.frame_id = self.target_frame

        mp_pos = self.msg_to_numpy_vector(mp_state.pose.pose.position, ["x", "y", "z"])
        
        drone_pos = self.msg_to_numpy_vector(
            drone_state.pose.pose.position, ["x", "y", "z"]
        )

        rel_pos_vals = mp_pos - drone_pos

        rel_pos.pose.position.x, rel_pos.pose.position.y, rel_pos.pose.position.z = (
            rel_pos_vals  # type: ignore
        )

        return rel_pos

    def _get_relative_orientation(self, drone_state, mp_state):
        """Computes relative orientation using quaternions."""
        q_mp = self.msg_to_numpy_vector(
            mp_state.pose.pose.orientation, ["x", "y", "z", "w"]
        )
        q_drone = self.msg_to_numpy_vector(
            drone_state.pose.pose.orientation, ["x", "y", "z", "w"]
        )

        q_rel = quaternion_multiply(q_drone, quaternion_inverse(q_mp))

        return Quaternion(*q_rel)
    

