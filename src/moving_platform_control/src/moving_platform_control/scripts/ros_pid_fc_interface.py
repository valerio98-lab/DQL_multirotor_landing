#!/usr/bin/python

import rospy
from uav_msgs.msg import uav_pose
from librepilot.msg import LibrepilotActuators
from geometry_msgs.msg import PoseStamped
import numpy as np 
from datetime import datetime as dt
from std_msgs.msg import Float64



node_name = 'ros_pid_fc_interface_node'
moving_platform_name = rospy.get_param(rospy.get_namespace()+node_name+'/mp_name','moving_platform')  
publish_hz = float(rospy.get_param(rospy.get_namespace()+node_name+'/publish_hz','10'))
x_offset = float(rospy.get_param(rospy.get_namespace()+node_name+'/x_offset','0'))
y_offset = float(rospy.get_param(rospy.get_namespace()+node_name+'/y_offset','0'))
z_offset = float(rospy.get_param(rospy.get_namespace()+node_name+'/z_offset','0'))
max_amplitude = float(rospy.get_param(rospy.get_namespace()+node_name+'/max_amplitude','0'))
pwm_neutral = float(rospy.get_param(rospy.get_namespace()+node_name+'/pwm_neutral','1500'))
actuator_topic = ('actuatorcommand',LibrepilotActuators)
uav_pose_topic = ('command',uav_pose)
mp_pose_topic = ('/vicon/moving_platform/pose_enu_filtered',PoseStamped)
mp_control_effort_topic = ('/vicon/moving_platform/controller/control_effort',Float64)
mp_control_state_topic = ('/vicon/moving_platform/controller/state',Float64)

class RosPidFcInterface():
    def __init__(self):
        #Define subscribers
        self.mp_pose_subscriber = rospy.Subscriber(mp_pose_topic[0],mp_pose_topic[1],self.read_mp_pose_enu)
        self.mp_control_effort_subscriber = rospy.Subscriber(mp_control_effort_topic[0],mp_control_effort_topic[1],self.read_mp_control_effort)

        #Define publishers
        self.actuator_publisher = rospy.Publisher(actuator_topic[0],actuator_topic[1],queue_size = 1)
        self.uav_pose_publisher = rospy.Publisher(uav_pose_topic[0],uav_pose_topic[1],queue_size = 1)
        self.mp_linear_position_publisher = rospy.Publisher(mp_control_state_topic[0],mp_control_state_topic[1],queue_size = 1)

        #Define variables needed for execution
        self.pwm_values = np.array([1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500])
        self.mp_pose_corrected_enu = PoseStamped()
        return

    def read_mp_pose_enu(self,msg):
        self.mp_pose_corrected_enu = msg
        self.mp_pose_corrected_enu.pose.position.x = self.mp_pose_corrected_enu.pose.position.x - x_offset
        self.mp_pose_corrected_enu.pose.position.y = self.mp_pose_corrected_enu.pose.position.y - y_offset
        self.mp_pose_corrected_enu.pose.position.z = self.mp_pose_corrected_enu.pose.position.z - z_offset
        self.publish_mp_control_state()
        return

    def read_mp_control_effort(self,msg):
        self.pwm_values[0] = msg.data + pwm_neutral
        return

    def publish_mp_control_state(self):
        msg = Float64()
        msg.data = self.mp_pose_corrected_enu.pose.position.y
        self.mp_linear_position_publisher.publish(msg)
        return

    def apply_safety_features(self):
        #Make sure that pwm values are within valid range
        if np.any(self.pwm_values > 2000) or np.any(self.pwm_values < 1000):
            self.pwm_values = np.clip(self.pwm_values,1000,2000)
            print("At least one pwm value of pwm value list",self.pwm_values,"was out of bounds. Clipped all values to range [1000,2000].")
        
        #As soon as the platform position exceeds limits, send throttle neutral pwm command
        if np.abs(self.mp_pose_corrected_enu.pose.position.y) >= max_amplitude:
            self.pwm_values[0] = pwm_neutral
            print("moving platform outside bounds. Throttle neutral command sent...")
        return


    def send_pwm_msg(self):
        msg_actuatorcommand = LibrepilotActuators()
        self.apply_safety_features()
        msg_actuatorcommand.data.data = self.pwm_values
        self.actuator_publisher.publish(msg_actuatorcommand)
        msg_command = uav_pose()
        msg_command.flightmode = 3
        self.uav_pose_publisher.publish(msg_command)
        return


if __name__ == '__main__':
    rospy.init_node(node_name)
    ros_pid_fc_interface = RosPidFcInterface()
    rate = rospy.Rate(publish_hz)
    while not rospy.is_shutdown():
        ros_pid_fc_interface.publish_mp_control_state()
        ros_pid_fc_interface.send_pwm_msg()
        rate.sleep()



