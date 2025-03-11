# roll_pitch_yawrate_thrust_controller.py
import numpy as np
import math
import tf.transformations as tft
from dql_multirotor_landing.observation_utils import ObservationUtils

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


# Definiamo una semplice classe per il comando roll_pitch_yawrate_thrust
class EigenRollPitchYawrateThrust:
    def __init__(self, roll=0.0, pitch=0.0, yaw_rate=0.0, thrust=None):
        self.roll = roll
        self.pitch = pitch
        self.yaw_rate = yaw_rate
        if thrust is None:
            self.thrust = np.zeros(3)
        else:
            self.thrust = thrust


class RollPitchYawrateThrustController:
    def __init__(self):
        self.attitude_gain = np.array([0.7, 0.7, 0.035])
        self.angular_rate_gain = np.array([0.1, 0.1, 0.025])
        self.drone = Drone()
        self.allocation_matrix = self.calculate_allocation_matrix()
        self.initialized_params = False
        self.controller_active = False

        self.normalized_attitude_gain = None
        self.normalized_angular_rate_gain = None
        self.angular_acc_to_rotor_velocities = None

        # Comando corrente (roll, pitch, yaw_rate, thrust)
        self.roll_pitch_yawrate_thrust = EigenRollPitchYawrateThrust()
        self.odometry = None

        self.InitializeParameters()

    def InitializeParameters(self):
        # Aggiorna la matrice di allocazione usando la configurazione dei rotori del veicolo
        self.allocation_matrix = self.calculate_allocation_matrix()

        inv_inertia = np.linalg.inv(self.drone.inertia)
        self.normalized_attitude_gain = inv_inertia @ self.attitude_gain
        self.normalized_angular_rate_gain = inv_inertia @ self.angular_rate_gain

        # Costruisci la matrice I (4x4): blocco superiore sinistro = inertia, I[3,3] = 1
        I = np.zeros((4, 4))
        I[:3, :3] = self.drone.inertia
        I[3, 3] = 1.0

        A = self.allocation_matrix  # dimensione (4, num_rotors)
        inv_term = np.linalg.inv(A @ A.T)
        self.angular_acc_to_rotor_velocities = A.T @ inv_term @ I

        self.initialized_params = True

    def calculate_allocation_matrix(self):
        num_rotors = len(self.drone.rotor_configuration.rotors)
        A = np.zeros((4, num_rotors))
        for i, rotor in enumerate(self.drone.rotor_configuration.rotors):
            A[0, i] = math.sin(rotor.angle) * rotor.arm_length * rotor.rotor_force_constant
            A[1, i] = -math.cos(rotor.angle) * rotor.arm_length * rotor.rotor_force_constant
            A[2, i] = -rotor.direction * rotor.rotor_force_constant * rotor.rotor_moment_constant
            A[3, i] = rotor.rotor_force_constant
        return A

    def CalculateRotorVelocities(self):
        """
        Calcola le velocità dei rotori.
        Restituisce un vettore NumPy di dimensione pari al numero di rotori.
        """
        if not self.initialized_params:
            raise Exception("I parametri del controller non sono stati inizializzati.")
        
        num_rotors = len(self.drone.rotor_configuration.rotors)
        if not self.controller_active:
            return np.zeros(num_rotors)

        angular_acceleration = self.ComputeDesiredAngularAcc()

        # Costruisci il vettore di comando a 4 elementi: [angular_acceleration; thrust_z]
        angular_acceleration_thrust = np.zeros(4)
        angular_acceleration_thrust[:3] = angular_acceleration
        angular_acceleration_thrust[3] = self.roll_pitch_yawrate_thrust.thrust[2]

        rotor_velocities = self.angular_acc_to_rotor_velocities @ angular_acceleration_thrust
        rotor_velocities = np.maximum(rotor_velocities, 0)
        rotor_velocities = np.sqrt(rotor_velocities)
        return rotor_velocities

    def SetOdometry(self, odometry):
        """
        Imposta lo stato odometrico.
        L'oggetto odometry dovrebbe avere attributi: position, orientation (quaternion [x,y,z,w]),
        velocity e angular_velocity (tutti vettori NumPy).
        """
        self.odometry = odometry

    def SetRollPitchYawrateThrust(self, roll_pitch_yawrate_thrust):
        """
        Imposta il comando di roll, pitch, yaw_rate e thrust.
        """
        self.roll_pitch_yawrate_thrust = roll_pitch_yawrate_thrust
        self.controller_active = True

    def ComputeDesiredAngularAcc(self):
        """
        Calcola l'accelerazione angolare desiderata secondo la logica di Lee et al.
        Restituisce un vettore NumPy di dimensione 3.
        """
        if self.odometry is None:
            raise Exception("L'odometria non è stata impostata.")
        
        # Ottieni la matrice di rotazione dall'orientamento (quaternion in formato [x,y,z,w])
        R_full = tft.quaternion_matrix(self.odometry.orientation)
        R = R_full[:3, :3]

        # Calcola l'angolo di yaw corrente: yaw = atan2(R[1,0], R[0,0])
        yaw = math.atan2(R[1, 0], R[0, 0])

        # Costruisci la matrice di rotazione desiderata: R_des = R_yaw * R_roll * R_pitch
        R_yaw = tft.rotation_matrix(yaw, (0, 0, 1))[:3, :3]
        R_roll = tft.rotation_matrix(self.roll_pitch_yawrate_thrust.roll, (1, 0, 0))[:3, :3]
        R_pitch = tft.rotation_matrix(self.roll_pitch_yawrate_thrust.pitch, (0, 1, 0))[:3, :3]
        R_des = R_yaw @ R_roll @ R_pitch

        # Calcola l'errore angolare come: 0.5 * (R_des^T*R - R^T*R_des)
        angle_error_matrix = 0.5 * (R_des.T @ R - R.T @ R_des)
        angle_error = ObservationUtils.vec_from_skew_matrix(angle_error_matrix)

        # Imposta la velocità angolare desiderata: normalmente solo la componente z (yaw rate) è non nulla
        angular_rate_des = np.zeros(3)
        angular_rate_des[2] = self.roll_pitch_yawrate_thrust.yaw_rate

        # Calcola l'errore di velocità angolare: odometry.angular_velocity - R_des^T * R * angular_rate_des
        angular_rate_error = self.odometry.angular_velocity - (R_des.T @ (R @ angular_rate_des))

        # Legge la legge di controllo:
        # - gain * errore angolare - gain * errore di velocità angolare + termine non lineare (cross product)
        angular_acceleration = - np.multiply(angle_error, self.normalized_attitude_gain) \
                                - np.multiply(angular_rate_error, self.normalized_angular_rate_gain) \
                                + np.cross(self.odometry.angular_velocity, self.odometry.angular_velocity)
        return angular_acceleration
