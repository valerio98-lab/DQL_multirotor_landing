#!/usr/bin/env python
# roll_pitch_yawrate_thrust_controller_node.py

import rospy
import numpy as np

from nav_msgs.msg import Odometry
from mav_msgs.msg import RollPitchYawrateThrust, Actuators

# Importiamo i moduli che abbiamo definito
from dql_multirotor_landing.attitude_controller import RollPitchYawrateThrustController, EigenRollPitchYawrateThrust
from dql_multirotor_landing.observation_utils import ObservationUtils  

class RollPitchYawrateThrustControllerNode:
    def __init__(self):
        rospy.init_node('attitude_node', anonymous=True)
        
        self.controller = RollPitchYawrateThrustController()
        self.controller.InitializeParameters()

        self.cmd_sub = rospy.Subscriber(rospy.get_param('~command_topic', 'command/roll_pitch_yawrate_thrust'),
                                        RollPitchYawrateThrust, self.roll_pitch_yawrate_thrust_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(rospy.get_param('~odometry_topic', 'odometry_sensor1/odometry'),
                                         Odometry, self.odometry_callback, queue_size=1)
        self.motor_pub = rospy.Publisher(rospy.get_param('~motor_topic', 'command/motor_speed'),
                                         Actuators, queue_size=1)

    def roll_pitch_yawrate_thrust_callback(self, msg):

        cmd = EigenRollPitchYawrateThrust(
            roll=msg.roll,
            pitch=msg.pitch,
            yaw_rate=msg.yaw_rate,
            thrust=np.array([msg.thrust.x, msg.thrust.y, msg.thrust.z])
        )
        self.controller.SetRollPitchYawrateThrust(cmd)

    def odometry_callback(self, msg):

        rospy.loginfo_once("RollPitchYawrateThrustController ha ricevuto il primo messaggio di odometria.")
        
        odom = ObservationUtils.eigen_odometry_from_msg(msg)
        self.controller.SetOdometry(odom)
        
        rotor_velocities = self.controller.CalculateRotorVelocities()
        
        actuator_msg = Actuators()
        actuator_msg.angular_velocities = rotor_velocities.tolist()
        actuator_msg.header.stamp = msg.header.stamp
        
        self.motor_pub.publish(actuator_msg)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = RollPitchYawrateThrustControllerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
