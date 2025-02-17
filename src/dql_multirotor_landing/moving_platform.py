#!/usr/bin/env python3
import numpy as np
import rospy
import tf2_ros
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, TransformStamped
from std_msgs.msg import Bool, Float64
from training_q_learning.parameters import Parameters
from tf.transformations import quaternion_from_euler
from training_q_learning.srv import ResetRandomSeed, ResetRandomSeedResponse


class MovingPlatform:
    """
    This class models a moving platform following a periodic sinusoidal trajectory along both the x-axis and y-axis.
    The motion is defined by sinusoidal functions that determine position and velocity based on the specified trajectory parameters.

    The motion equations used are:
        x_mp(t) = r_x * sin(ω_x * t)   # Position along x
        y_mp(t) = r_y * sin(ω_y * t)   # Position along y
        v_x(t)  = r_x * ω_x * cos(ω_x * t)   # Velocity along x
        v_y(t)  = r_y * ω_y * cos(ω_y * t)   # Velocity along y

    Where:
        - r_x  : Amplitude of motion along the x-axis (meters)
        - r_y  : Amplitude of motion along the y-axis (meters)
        - ω_x  : Angular frequency of motion along x (ω_x = t_x / r_x).
        - ω_y  : Angular frequency of motion along y (ω_y = t_y / r_y).
        - t    : Time elapsed in the simulation.
        
    The platform oscillates along the x and y axes independently, following a sinusoidal pattern determined by the specified speed and radius parameters.
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
        self.parameters = Parameters()
        self.trajectory_type = rospy.get_param(f"{config_path}/moving_platform/trajectory_type", "rectilinear_periodic_straight")
        self.t_x = float(rospy.get_param(f"{config_path}/moving_platform/t_x", "1"))
        self.frequency = float(rospy.get_param(f"{config_path}/moving_platform/frequency", "100"))
        self.start_position = rospy.get_param(
            f"{config_path}/moving_platform/start_position", {'x': 0, 'y': 0, 'z': 0}
        )
        self.start_orientation = rospy.get_param(
            f"{config_path}/moving_platform/start_orientation", {'phi': 0, 'theta': 0, 'psi': 0}
        )
        self.r_x = float(rospy.get_param(f"{config_path}/moving_platform/r_x", "2"))
        self.r_y = float(rospy.get_param(f"{config_path}/moving_platform/r_y", "2"))
        self.t_y = float(rospy.get_param(f"{config_path}/moving_platform/t_y", "1"))

        self.phi = self.start_orientation.get("phi", 0)
        self.theta = self.start_orientation.get("theta", 0)
        self.psi = self.start_orientation.get("psi", 0)


        self.pose = self._initialize_pose()

        self.u = 0.0
        self.v = 0.0
        self.w = 0.0

        self.t = 0.0
        self.delta_t = 1.0 / self.frequency

        self.br = tf2_ros.TransformBroadcaster()

        rospy.loginfo("MovingPlatformNode initialized.")

    def compute_trajectory(self):
        """
        Computes the trajectory (x, y) of the moving platform and updates its Pose.
        
        The method updates self.pose.position.x, self.pose.position.y, and the horizontal velocities (u, v)
        based on the specified horizontal trajectory type.
        """
        omega = self.t_x / self.r_x
        omega_y = self.t_y / self.r_y
        self.pose.position.x = self.r_x * np.sin(omega * self.t) + self.start_position["x"]
        self.pose.position.y = self.r_y * np.sin(omega_y * self.t) + self.start_position["y"]
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

    def update(self):
        """
        Updates the platform's state by computing the next point in the trajectory.
        
        Returns:
            tuple: Updated platform state in the format 
                   (Pose, u, v, w, phi, theta, psi)
        """
        self.compute_trajectory()
        return self.pose, self.u, self.v
