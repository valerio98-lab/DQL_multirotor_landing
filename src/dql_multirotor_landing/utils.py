from geometry_msgs.msg import Vector3Stamped
import rospy
from tf2_geometry_msgs import do_transform_pose, do_transform_vector3
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped, Quaternion, Vector3
from tf.transformations import euler_from_quaternion, quaternion_multiply, quaternion_inverse


class KalmanFilter1D:
    def __init__(self, process_variance, measurement_variance):
        self.x = 0.0 
        self.P = 1.0 
        self.Q = process_variance 
        self.R = measurement_variance  

    def update(self, measurement):
        self.P += self.Q 

        K = self.P / (self.P + self.R)  
        self.x += K * (measurement - self.x) 
        self.P *= (1 - K) 

        return self.x 


class KalmanFilter3D:
    def __init__(self, process_variance=1e-4, measurement_variance=(0.1, 0.1, 0.1)):
        self.kf_x = KalmanFilter1D(process_variance, measurement_variance[0]**2)
        self.kf_y = KalmanFilter1D(process_variance, measurement_variance[1]**2)
        self.kf_z = KalmanFilter1D(process_variance, measurement_variance[2]**2)

    def filter(self, current_rel_v, timestep, last_vel, last_timestep):
        dt = timestep - last_timestep
        if dt <= 0:
            dt = 0.01  

        raw_accel_x = (current_rel_v.x - last_vel.x) / dt
        raw_accel_y = (current_rel_v.y - last_vel.y) / dt
        raw_accel_z = (current_rel_v.z - last_vel.z) / dt

        accel = Vector3Stamped()
        accel.vector.x = self.kf_x.update(raw_accel_x)
        accel.vector.y = self.kf_y.update(raw_accel_y)
        accel.vector.z = self.kf_z.update(raw_accel_z)

        return accel
    

class Utils():
    def compute_relative_state(self, drone_state_target, mp_state_target, target_frame):
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
