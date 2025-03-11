import numpy as np
import math
import tf.transformations as tft

class Rotor:
    def __init__(self, angle, arm_length, rotor_force_constant, rotor_moment_constant, direction):
        self.angle = angle
        self.arm_length = arm_length
        self.rotor_force_constant = rotor_force_constant
        self.rotor_moment_constant = rotor_moment_constant
        self.direction = direction  

class RotorConfiguration:
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
    def __init__(self):
        self.mass = 0.68 
        self.gravity = 9.81   
        self.inertia = np.diag([0.007, 0.007, 0.012])
        self.rotor_configuration = RotorConfiguration()


class StateMsg:
    def __init__(self, roll=0.0, pitch=0.0, yaw_rate=0.0, thrust=None):
        self.roll = roll
        self.pitch = pitch
        self.yaw_rate = yaw_rate
        if thrust is None:
            self.thrust = np.zeros(3)
        else:
            self.thrust = thrust

class AttitudeController:
    def __init__(self):
        self.attitude_gain = np.array([0.7, 0.7, 0.035])
        self.angular_rate_gain = np.array([0.1, 0.1, 0.025])
        self.odometry = None
        self.drone = Drone()
        self.allocation_matrix = self.calculate_allocation_matrix()

        self.state = StateMsg()

    def calculate_allocation_matrix(self):
        A = np.zeros((4, 4))
        for i, rotor in enumerate(self.drone.rotor_configuration.rotors):
            A[0, i] = math.sin(rotor.angle) * rotor.arm_length * rotor.rotor_force_constant
            A[1, i] = -math.cos(rotor.angle) * rotor.arm_length * rotor.rotor_force_constant
            A[2, i] = -rotor.direction * rotor.rotor_force_constant * rotor.rotor_moment_constant
            A[3, i] = rotor.rotor_force_constant
        return A


    def compute_rotor_velocities(self):

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
        Calcola l'accelerazione angolare desiderata secondo la logica di Lee et al.
        Restituisce un vettore NumPy di dimensione 3.
        """
        if self.odometry is None:
            raise Exception("L'odometria non è stata impostata.")
        
        # Ottieni la matrice di rotazione dall'orientamento attuale 
        R_full = tft.quaternion_matrix(self.odometry.orientation)
        R = R_full[:3, :3]

        # Calcola l'angolo di yaw corrente: yaw = atan2(R[1,0], R[0,0]) necessario per la matrice di rotazione desiderata
        yaw = math.atan2(R[1, 0], R[0, 0])

        # Costruisci la matrice di rotazione desiderata: R_des = R_yaw * R_roll * R_pitch
        R_yaw = tft.rotation_matrix(yaw, (0, 0, 1))[:3, :3]
        R_roll = tft.rotation_matrix(self.state.roll, (1, 0, 0))[:3, :3]
        R_pitch = tft.rotation_matrix(self.state.pitch, (0, 1, 0))[:3, :3]
        R_des = R_yaw @ R_roll @ R_pitch

        # Calcola l'errore angolare come: 0.5 * (R_des^T*R - R^T*R_des)
        angle_error_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
        angle_error = self._vec_from_skew_matrix(angle_error_matrix)

        # Imposta la velocità angolare desiderata: normalmente solo la componente z (yaw rate) è non nulla
        angular_rate_des = np.zeros(3)
        angular_rate_des[2] = self.state.yaw_rate

        # Calcola l'errore di velocità angolare: odometry.angular_velocity - R_des^T * R * angular_rate_des
        angular_rate_error = self.odometry.angular_velocity - (R_des.T @ (R @ angular_rate_des))

        # Legge la legge di controllo:
        # - gain * errore angolare - gain * errore di velocità angolare + termine non lineare (cross product)

        moment = - np.multiply(angle_error, self.attitude_gain) \
                                - np.multiply(angular_rate_error, self.angular_rate_gain) \
                                + np.cross(self.odometry.angular_velocity, self.odometry.angular_velocity)
        return moment


    def _vec_from_skew_matrix(self, skew_mat):
        return np.array([skew_mat[2, 1], skew_mat[0, 2], skew_mat[1, 0]])
    