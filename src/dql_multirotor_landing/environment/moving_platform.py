import torch


class MovingPlatform:
    """
    This class models a moving platform following a Rectilinear Periodic Motion (RPM).
    The motion is based on a sinusoidal function that dictates position, velocity,
    and acceleration along the x-axis.

    The motion equation used is:
        x_mp(t)  = r_mp * sin(ω_mp * t)   # Position
        v_mp(t)  = v_mp * cos(ω_mp * t)   # Velocity
        a_mp(t)  = - (v_mp^2 / r_mp) * sin(ω_mp * t)  # Acceleration

    Where:
        - r_mp  : Maximum displacement (amplitude) of the motion. (meters)
        - v_mp  : Maximum velocity of the platform (m/s)
        - ω_mp  : Angular frequency of oscillation (ω_mp = v_mp / r_mp).
        - t     : Time elapsed in the simulation.

    The motion oscillates back and forth between -r_mp and +r_mp, following a sinusoidal pattern.

    """

    def __init__(self, r_mp=2.0, v_mp=0.8):
        """
        r_wheel (float): Radius of the wheels (meters), used to compute angular velocity.
        dt (float): Simulation timestep (seconds).
        """

        self.r_mp = r_mp
        self.v_mp = v_mp
        self.omega_mp = v_mp / r_mp
        self.wheel_radius = 0.5
        self.t = torch.tensor(0.0, dtype=torch.float32)

    def compute_wheel_velocity(self, dt):
        """
        Computes the velocity of the platform and converts it into wheel angular velocity.

        Returns:
            torch.Tensor: Angular velocity of the wheels (rad/s).
        """
        self.t += torch.tensor(dt, dtype=torch.float32)

        x = self.r_mp * torch.sin(self.omega_mp * self.t)
        v = self.v_mp * torch.cos(self.omega_mp * self.t)
        a = -(self.v_mp**2 / self.r_mp) * torch.sin(self.omega_mp * self.t)
        w = v / self.wheel_radius

        return x, v, a, w


if __name__ == "__main__":
    platform = MovingPlatform()
    dt = 0.02
    for _ in range(50):
        rpm_values = platform.compute_rpm_input(dt)
        print(f"Posizione: {rpm_values[0]}, Velocità: {rpm_values[1]}, Accelerazione: {rpm_values[2]}")
