import numpy as np
import math
import tf.transformations as tft

class Rotor:
    """
    Represents a single rotor in the quadcopter configuration.
    Each rotor has:
    - angle: Position of the rotor in the body frame.
    - arm_length: Distance from the center of mass to the rotor.
    - rotor_force_constant: Coefficient mapping squared rotor speed to thrust.
    - rotor_moment_constant: Coefficient mapping squared rotor speed to torque.
    - direction: Rotation direction (+1 for CCW, -1 for CW).
    """
    def __init__(self, angle, arm_length, rotor_force_constant, rotor_moment_constant, direction):
        self.angle = angle
        self.arm_length = arm_length
        self.rotor_force_constant = rotor_force_constant
        self.rotor_moment_constant = rotor_moment_constant
        self.direction = direction  

class RotorConfiguration:
    """
    Defines the quadrotor configuration with four rotors positioned at 90-degree intervals.
    """
    def __init__(self):
        self.rotors = [
            Rotor(angle=0.0,
                  arm_length=0.17,
                  rotor_force_constant=8.54858e-06,
                  rotor_moment_constant=0.016,
                  direction=-1),
            Rotor(angle=math.pi/2,
                  arm_length=0.17,
                  rotor_force_constant=8.54858e-06,
                  rotor_moment_constant=0.016,
                  direction=1),
            Rotor(angle=math.pi,
                  arm_length=0.17,
                  rotor_force_constant=8.54858e-06,
                  rotor_moment_constant=0.016,
                  direction=-1),
            Rotor(angle=-math.pi/2,
                  arm_length=0.17,
                  rotor_force_constant=8.54858e-06,
                  rotor_moment_constant=0.016,
                  direction=1)
        ]

class Drone:
    """
    Defines the physical properties of the drone.
    - mass: Total mass of the drone.
    - gravity: Acceleration due to gravity.
    - inertia: Moment of inertia tensor (assumed diagonal for simplicity).
    """
    def __init__(self):
        self.mass = 0.68 
        self.gravity = 9.81   
        self.inertia = np.diag([0.007, 0.007, 0.012])
        self.rotor_configuration = RotorConfiguration()


class StateMsg:
    """
    Holds the state message that represents the desired drone orientation and thrust.
    - roll, pitch: Desired roll and pitch angles.
    - yaw_rate: Desired yaw rate.
    - thrust: Desired thrust vector.
    """
    def __init__(self, roll=0.0, pitch=0.0, yaw_rate=0.0, thrust=None):
        self.roll = roll
        self.pitch = pitch
        self.yaw_rate = yaw_rate
        if thrust is None:
            self.thrust = np.zeros(3)
        else:
            self.thrust = thrust

class AttitudeController:
    """
    Implements the attitude control logic based on Lee et al.'s geometric control on SO(3).
    Controls the orientation of the drone by computing the required moments.
    """
    def __init__(self):
        self.attitude_gain = np.array([0.7, 0.7, 0.035])
        self.angular_rate_gain = np.array([0.1, 0.1, 0.025])
        self.odometry = None
        self.drone = Drone()
        self.allocation_matrix = self.compute_allocation_matrix()

        self.state = StateMsg()

    def compute_allocation_matrix(self):
        """
        Computes the allocation matrix that maps rotor thrusts to body torques and total thrust.
        """
        A = np.zeros((4, 4))
        for i, rotor in enumerate(self.drone.rotor_configuration.rotors):
            A[0, i] = math.sin(rotor.angle) * rotor.arm_length * rotor.rotor_force_constant
            A[1, i] = -math.cos(rotor.angle) * rotor.arm_length * rotor.rotor_force_constant
            A[2, i] = -rotor.direction * rotor.rotor_force_constant * rotor.rotor_moment_constant
            A[3, i] = rotor.rotor_force_constant
        return A


    def compute_rotor_velocities(self):
        """
        Computes the rotor speeds from moment and thrust.
        """
        angular_acceleration = self._compute_desired_ang_acc()

        # Costruisci il vettore di comando a 4 elementi: [angular_acceleration; thrust_z]
        moment_thrust = np.zeros(4)
        moment_thrust[:3] = angular_acceleration
        moment_thrust[3] = self.state.thrust[2]

        inv_term = np.linalg.inv(self.allocation_matrix)
        rotor_velocities = inv_term @ moment_thrust
        rotor_velocities = np.maximum(rotor_velocities, 0)
        rotor_velocities = np.sqrt(rotor_velocities)
        return rotor_velocities


    def _compute_desired_ang_acc(self):
        """
        Computes the desired control moments using the geometric control approach.
        """
        if self.odometry is None:
            raise Exception("L'odometria non è stata impostata.")
        
        # Obtain the current rotation matrix from the quaternion
        R_full = tft.quaternion_matrix(self.odometry.orientation)
        R = R_full[:3, :3]

        # Costruisci la matrice di rotazione desiderata: R_des = R_yaw * R_roll * R_pitch
        yaw = math.atan2(R[1, 0], R[0, 0])
        R_yaw = tft.rotation_matrix(yaw, (0, 0, 1))[:3, :3]
        R_roll = tft.rotation_matrix(self.state.roll, (1, 0, 0))[:3, :3]
        R_pitch = tft.rotation_matrix(self.state.pitch, (0, 1, 0))[:3, :3]
        R_des = R_yaw @ R_roll @ R_pitch

        # Compute attitude error e_R = 0.5 * (R_des^T * R - R^T * R_des)
        angle_error_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
        angle_error = self._vec_from_skew_matrix(angle_error_matrix)

        # Compute angular velocity error e_w = w - R^T * R_des * w_des
        angular_rate_des = np.zeros(3)
        angular_rate_des[2] = self.state.yaw_rate
        angular_rate_error = self.odometry.angular_velocity - (R_des.T @ (R @ angular_rate_des))

        # Compute control moment M = -k_R * e_R - k_w * e_w + w x Jw
        
        moment = - np.multiply(angle_error, self.attitude_gain) \
                                - np.multiply(angular_rate_error, self.angular_rate_gain) \
                                + np.cross(self.odometry.angular_velocity, self.odometry.angular_velocity)
        return moment


    def _vec_from_skew_matrix(self, skew_mat):
        return np.array([skew_mat[2, 1], skew_mat[0, 2], skew_mat[1, 0]])
    