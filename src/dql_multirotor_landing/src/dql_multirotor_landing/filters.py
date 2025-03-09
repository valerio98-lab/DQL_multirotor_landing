from geometry_msgs.msg import Vector3Stamped
from collections import deque

class KalmanFilter1D:
    def __init__(self, process_variance, measurement_variance):
        """
        Implements a 1D Kalman Filter for state estimation.

        Parameters:
        - process_variance (float): Variance of the process noise (Q).
        - measurement_variance (float): Variance of the measurement noise (R).
        """

        self.x = 0.0 
        self.P = 1.0 
        self.Q = process_variance 
        self.R = measurement_variance  

    def update(self, measurement):
        """
        Performs a Kalman update step with a new measurement.

        Parameters:
        - measurement (float): The latest sensor measurement.

        Returns:
        - float: The updated state estimate.
        """

        self.P += self.Q 

        K = self.P / (self.P + self.R)  
        self.x += K * (measurement - self.x) 
        self.P *= (1 - K) 

        return self.x 


class KalmanFilter3D:
    def __init__(self, process_variance=1e-4, measurement_variance=0.1):
        """
        Implements a 3D Kalman Filter for acceleration estimation.

        Parameters:
        - process_variance (float): Variance of the process noise.
        - measurement_variance (float): Variance of the measurement noise.
        """

        self.kf_x = KalmanFilter1D(process_variance, measurement_variance**2)
        self.kf_y = KalmanFilter1D(process_variance, measurement_variance**2)
        self.kf_z = KalmanFilter1D(process_variance, measurement_variance**2)

    def filter(self, current_rel_v, timestep, last_vel, last_timestep):
        """
        Estimates filtered acceleration using a Kalman Filter.

        Parameters:
        - current_rel_v (Vector3): Current relative velocity.
        - timestep (float): Current timestamp.
        - last_vel (Vector3): Previous velocity.
        - last_timestep (float): Previous timestamp.

        Returns:
        - Vector3Stamped: Filtered acceleration estimate.
        """

        dt = timestep - last_timestep
        if dt <= 0:
            dt = 0.01  

        raw_acc_x = (current_rel_v.x - last_vel.x) / dt
        raw_acc_y = (current_rel_v.y - last_vel.y) / dt
        raw_acc_z = (current_rel_v.z - last_vel.z) / dt

        acc = Vector3Stamped()
        acc.vector.x = self.kf_x.update(raw_acc_x)
        acc.vector.y = self.kf_y.update(raw_acc_y)
        acc.vector.z = self.kf_z.update(raw_acc_z)

        return acc
    

class ButterworthFilter:
    """
    Second-order low-pass Butterworth filter using the bilinear transform.
    Maintains internal state (raw and filtered deques) and computes the filtered output.
    """
    def __init__(self):
        """
        :param c: Parameter related to cutoff frequency (c = tan(omega_c/2))
        :param init_value: Initial value for the filter queues
        """
        self.c = 1.0
        self.denom = 1 + self.c ** 2 + 1.414 * self.c
        self.raw_values = deque([0.0, 0.0, 0.0], maxlen=3)
        self.filtered_values = deque([0.0, 0.0, 0.0], maxlen=3)

    def update(self, new_value: float) -> float:
        """
        Update the filter with a new input value and return the new filtered value.
        """
        self.raw_values.appendleft(new_value)
        value = (1.0 / self.denom) * (
            self.raw_values[2] + 2 * self.raw_values[1] + self.raw_values[0]
            - (self.c ** 2 - 1.414 * self.c + 1) * self.filtered_values[2]
            - ((-2 * self.c ** 2 + 2) * self.filtered_values[1])
        )
        self.filtered_values.appendleft(value)
        return value