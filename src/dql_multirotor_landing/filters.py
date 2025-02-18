from geometry_msgs.msg import Vector3Stamped

class LowPassFilter1D:
    def __init__(self, alpha):
        """
        Implements a 1D low-pass filter to smooth noisy data.
        
        Parameters:
        - alpha (float): Smoothing factor (0 < alpha ≤ 1). Higher values give more weight to the new measurement.
        """

        self.alpha = alpha
        self.filtered_value = None 

    def update(self, measurement):
        """
        Applies the low-pass filter to a new measurement.

        Parameters:
        - measurement (float): The latest sensor measurement.

        Returns:
        - float: The filtered measurement.
        """

        if self.filtered_value is None:
            self.filtered_value = measurement  
        else:
            self.filtered_value = self.alpha * measurement + (1 - self.alpha) * self.filtered_value

        return self.filtered_value

class LowPassFilter3D:
    def __init__(self, alpha_x=0.5, alpha_y=0.5, alpha_z=0.5):
        """
        Implements a 3D low-pass filter for acceleration data.

        Parameters:
        - alpha_x, alpha_y, alpha_z (float): Smoothing factors for each axis.
        """

        self.lpf_x = LowPassFilter1D(alpha_x)
        self.lpf_y = LowPassFilter1D(alpha_y)
        self.lpf_z = LowPassFilter1D(alpha_z)

    def filter(self, current_rel_v, timestep, last_vel, last_timestep):
        """
        Estimates the filtered acceleration using a low-pass filter.

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

        raw_accel_x = (current_rel_v.x - last_vel.x) / dt
        raw_accel_y = (current_rel_v.y - last_vel.y) / dt
        raw_accel_z = (current_rel_v.z - last_vel.z) / dt

        accel = Vector3Stamped()
        accel.vector.x = self.lpf_x.update(raw_accel_x)
        accel.vector.y = self.lpf_y.update(raw_accel_y)
        accel.vector.z = self.lpf_z.update(raw_accel_z)

        return accel


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

        raw_accel_x = (current_rel_v.x - last_vel.x) / dt
        raw_accel_y = (current_rel_v.y - last_vel.y) / dt
        raw_accel_z = (current_rel_v.z - last_vel.z) / dt

        accel = Vector3Stamped()
        accel.vector.x = self.kf_x.update(raw_accel_x)
        accel.vector.y = self.kf_y.update(raw_accel_y)
        accel.vector.z = self.kf_z.update(raw_accel_z)

        return accel