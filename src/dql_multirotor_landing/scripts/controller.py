#!/usr/bin/env python
import rospy
from dql_multirotor_landing.pid import PidObject

if __name__ == '__main__':
    rospy.init_node("controller")
    pid = PidObject()
