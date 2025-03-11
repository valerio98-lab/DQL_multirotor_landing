#!/usr/bin/env python
# roll_pitch_yawrate_thrust_controller_node.py

import rospy
import numpy as np

from dataclasses import dataclass
from nav_msgs.msg import Odometry
from mav_msgs.msg import RollPitchYawrateThrust, Actuators

# Importiamo i moduli che abbiamo definito
from dql_multirotor_landing.attitude_controller import AttitudeController, StateMsg

@dataclass
class OdometryStruct:
    position: np.ndarray
    orientation : np.ndarray
    velocity : np.ndarray
    angular_velocity : np.ndarray


class RollPitchYawrateThrustControllerNode:
    def __init__(self):
        rospy.init_node('attitude_node', anonymous=True)
        
        self.controller = AttitudeController()

        self.cmd_sub = rospy.Subscriber(rospy.get_param('~command_topic', 'command/roll_pitch_yawrate_thrust'),
                                        RollPitchYawrateThrust, self.state_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(rospy.get_param('~odometry_topic', 'odometry_sensor1/odometry'),
                                         Odometry, self.odometry_callback, queue_size=1)
        self.motor_pub = rospy.Publisher(rospy.get_param('~motor_topic', 'command/motor_speed'),
                                         Actuators, queue_size=1)

    def state_callback(self, msg):

        cmd = StateMsg(
            roll=msg.roll,
            pitch=msg.pitch,
            yaw_rate=msg.yaw_rate,
            thrust=np.array([msg.thrust.x, msg.thrust.y, msg.thrust.z])
        )
        self.controller.state = cmd

    def odometry_callback(self, msg):

        rospy.loginfo_once("RollPitchYawrateThrustController ha ricevuto il primo messaggio di odometria.")
        
        odom = self._odometry_from_msg(msg)
        self.controller.odometry = odom
        
        rotor_velocities = self.controller.compute_rotor_velocities()
        
        actuator_msg = Actuators()
        actuator_msg.angular_velocities = rotor_velocities.tolist()
        actuator_msg.header.stamp = msg.header.stamp
        
        self.motor_pub.publish(actuator_msg)


    def _odometry_from_msg(self, msg):

        position = np.array([msg.pose.pose.position.x,
                                msg.pose.pose.position.y,
                                msg.pose.pose.position.z])
        orientation = np.array([msg.pose.pose.orientation.x,
                                    msg.pose.pose.orientation.y,
                                    msg.pose.pose.orientation.z,
                                    msg.pose.pose.orientation.w])
        velocity = np.array([msg.twist.twist.linear.x,
                                msg.twist.twist.linear.y,
                                msg.twist.twist.linear.z])
        angular_velocity = np.array([msg.twist.twist.angular.x,
                                        msg.twist.twist.angular.y,
                                        msg.twist.twist.angular.z])
        
        odom = OdometryStruct(position, orientation, velocity, angular_velocity)
        return odom


    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = RollPitchYawrateThrustControllerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
