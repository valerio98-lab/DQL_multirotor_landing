
import rospy
from std_msgs.msg import Float64
import math


#parameters
node_name = 'mp_setpoints_node'
mp_name = rospy.get_param(rospy.get_namespace()+node_name+'/mp_name')
vmax = float(rospy.get_param(rospy.get_namespace()+node_name+'/vmax'))
rmax = float(rospy.get_param(rospy.get_namespace()+node_name+'/rmax'))
f = float(rospy.get_param(rospy.get_namespace()+node_name+'/update_freq'))
x_offset = float(rospy.get_param(rospy.get_namespace()+node_name+'/x_offset'))


#Setpoint topic
mp_setpoint_topic = ("/vicon/moving_platform/controller/setpoint",Float64)

#Publishers
mp_setpoint_publisher = rospy.Publisher(mp_setpoint_topic[0],mp_setpoint_topic[1],queue_size = 0)


#Script variables
M_PI = 3.14159265359
omega = vmax/rmax
t = 0
dt = 1/f

#Functions
def compute_setpoint_msg():
    """ Compute setpoint for rectilinear periodic movement"""
    msg = Float64()
    msg.data = rmax*math.sin(omega*t) + x_offset
    return msg


if __name__ == '__main__':
    #Initialize node
    rospy.init_node(node_name)
    publish_rate = rospy.Rate(f)
    while not rospy.is_shutdown():
        msg = compute_setpoint_msg()
        mp_setpoint_publisher.publish(msg)
        t += dt
        publish_rate.sleep()

