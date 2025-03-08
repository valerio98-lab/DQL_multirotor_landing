#!/usr/bin/env python3
from typing import Dict

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler


class MovingPlatform:
    """
    This class models a moving platform that can follow different trajectory modes.

    Supported trajectories:
      1. Mono-dimensional sinusoidal trajectory:
         - Motion is defined only along the x-axis (y remains constant).
         - Equations:
              x(t) = r_x * sin(ω * t) + x0
              y(t) = constant (y0)
              u(t) = r_x * ω * cos(ω * t)
              v(t) = 0

      2. Figure-eight trajectory (using a lemniscate of Gerono):
         - Motion follows a figure-eight pattern in the xy-plane.
         - Equations:
              x(t) = r_x * cos(ω * t) + x0
              y(t) = r_y * sin(ω * t) * cos(ω * t) + y0
              u(t) = -r_x * ω * sin(ω * t)
              v(t) = r_y * ω * (cos(ω * t)² - sin(ω * t)²)

    Parameters:
      - r_x, r_y: Amplitudes (or effective radii) along the x and y axes (meters)
      - t_x, t_y: Parameters used to compute the angular frequency (ω = t_x / r_x, etc.)
      - t: Time elapsed in the simulation

    The trajectory mode is determined by the ROS parameter "trajectory_type".
    """

    def __init__(self):
        """
        Initializes the MovingPlatformNode.

        Retrieves trajectory parameters from the ROS parameter server, initializes the platform's Pose,
        velocities, and time variables, and sets up a transform broadcaster.
        """
        rospy.loginfo("Initializing MovingPlatformNode")

        config_path = "central_logic_node"

        # Retrieve trajectory parameters from ROS parameters
        self.trajectory_type = rospy.get_param(
            f"{config_path}/moving_platform/trajectory_type", "rpm"
        )
        self.t_x = float(rospy.get_param(f"{config_path}/moving_platform/t_x", "1"))  # type: ignore
        self.frequency = float(
            rospy.get_param(f"{config_path}/moving_platform/frequency", "100")  # type:ignore
        )
        self.start_position: Dict[str, int] = rospy.get_param(
            f"{config_path}/moving_platform/start_position",
            {"x": 0, "y": 0, "z": 0},  # type: ignore
        )
        self.start_orientation = rospy.get_param(
            f"{config_path}/moving_platform/start_orientation",
            {"phi": 0, "theta": 0, "psi": 0},
        )
        self.r_x = float(rospy.get_param(f"{config_path}/moving_platform/r_x", "2"))  # type:ignore
        self.r_y = float(rospy.get_param(f"{config_path}/moving_platform/r_y", "2"))  # type:ignore
        self.t_y = float(rospy.get_param(f"{config_path}/moving_platform/t_y", "1"))  # type:ignore

        self.phi: int = self.start_orientation.get("phi", 0)  # type:ignore
        self.theta: int = self.start_orientation.get("theta", 0)  # type:ignore
        self.psi: int = self.start_orientation.get("psi", 0)  # type:ignore

        self.pose = self._initialize_pose()

        self.u = 0.0
        self.v = 0.0

        self.t = 0.0
        self.delta_t = 1.0 / self.frequency

        self.br = tf2_ros.TransformBroadcaster()

        rospy.loginfo("MovingPlatformNode initialized.")

    def compute_trajectory(self):
        """
        Computes the trajectory (x, y) of the moving platform and updates its Pose,
        based on the specified trajectory type. See above for trajectory and maths details.
        """
        if self.trajectory_type == "eight":
            self.r_x = 3
            self.r_y = 3
            self.t_x = 0.8
            self.t_y = 0.8
            omega = self.t_x / self.r_x
            self.pose.position.x = (
                self.r_x * np.cos(omega * self.t) + self.start_position["x"]
            )
            self.pose.position.y = (
                self.r_y * np.sin(omega * self.t) * np.cos(omega * self.t)
                + self.start_position["y"]
            )

            self.u = -self.r_x * omega * np.sin(omega * self.t)
            self.v = (
                self.r_y
                * omega
                * (np.cos(omega * self.t) ** 2 - np.sin(omega * self.t) ** 2)
            )
        else:
            ## position and velocity along y is constant to 0 for a mono-dimensional trajectory,
            # however we keep the implementation along y for a possible future extension

            omega = self.t_x / self.r_x
            omega_y = 0  ## constant to 0 for a mono-dimensional trajectory
            self.pose.position.x = (
                self.r_x * np.sin(omega * self.t) + self.start_position["x"]
            )
            self.pose.position.y = (
                self.r_y * np.sin(omega_y * self.t) + self.start_position["y"]
            )
            self.u = self.r_x * omega * np.cos(omega * self.t)
            self.v = self.r_y * omega_y * np.cos(omega_y * self.t)

        self.t += self.delta_t

    def _initialize_pose(self) -> Pose:
        """
        Initializes and returns the Pose representing the platform's starting state.

        Converts the Euler angles (phi, theta, psi) to a quaternion and sets the initial position.

        Returns:
            Pose: The initialized Pose of the moving platform.
        """
        quat = quaternion_from_euler(self.phi, self.theta, self.psi)
        pose = Pose()
        pose.position.x = self.start_position.get("x", 0)
        pose.position.y = self.start_position.get("y", 0)
        pose.position.z = self.start_position.get("z", 0)
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        return pose

    def reset_time(self, t: float = 0.0):
        """
        Resets the time variable to 0.
        """
        # self.t = t
        ...

    def update(self):
        """
        Updates the platform's state by computing the next point in the trajectory.

        Returns:
            tuple: Updated platform state in the format
                   (Pose, u, v)
        """
        self.compute_trajectory()
        return self.pose, self.u, self.v
