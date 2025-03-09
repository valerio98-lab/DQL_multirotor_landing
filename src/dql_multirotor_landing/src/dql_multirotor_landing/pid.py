#!/usr/bin/env python
import rospy
import numpy as np
from collections import deque
from std_msgs.msg import Float64, Float64MultiArray
from dql_multirotor_landing.filters import ButterworthFilter

class PidObject:
    """
    PID Controller class that calculates control efforts based on the error between a setpoint
    and the current plant state. This controller uses Butterworth filters to smooth the error and
    its derivative before computing the PID terms.
    """
    def __init__(self, rate_hz=1000.0):
        self.rate_hz = rate_hz
        self.error = deque([0.0, 0.0], maxlen=3)
        self.error_deriv = deque([0.0, 0.0, 0.0], maxlen=3)
        self.filter_error = ButterworthFilter()
        self.filter_deriv = ButterworthFilter()
        self.error_integral = 0.0

        self.plant_state = 0.0
        self.control_effort = 0.0
        self.setpoint = 0.0
        self.new_state_or_setpt = False

        self.prev_time = None
        self.last_setpoint_msg_time = rospy.Time.now()

        self.proportional = 0.0
        self.integral = 0.0
        self.derivative = 0.0

        self.load_params()
        self.init_ros_comm()
        self.prev_time = rospy.Time.now()
        self.run()


    def load_params(self):
        ns = "~"
        self.Kp = rospy.get_param(ns + "Kp", 1.0)
        self.Ki = rospy.get_param(ns + "Ki", 0.0)
        self.Kd = rospy.get_param(ns + "Kd", 0.0)

        self.upper_limit = rospy.get_param(ns + "upper_limit", 1000.0)
        self.lower_limit = rospy.get_param(ns + "lower_limit", -1000.0)

        self.windup_limit = rospy.get_param(ns + "windup_limit", 1000.0)
        self.cutoff_frequency = rospy.get_param(ns + "cutoff_frequency", -1.0)
        self.topic_from_controller = rospy.get_param(ns + "topic_from_controller", "control_effort")
        self.topic_from_plant = rospy.get_param(ns + "topic_from_plant", "state")
        self.setpoint_topic = rospy.get_param(ns + "setpoint_topic", "setpoint")
        self.max_loop_frequency = rospy.get_param(ns + "max_loop_frequency", 1.0)
        self.min_loop_frequency = rospy.get_param(ns + "min_loop_frequency", 1000.0)

    def init_ros_comm(self):
        self.control_effort_pub = rospy.Publisher(self.topic_from_controller, Float64, queue_size=1)
        rospy.Subscriber(self.topic_from_plant, Float64, self.plant_state_callback)
        rospy.Subscriber(self.setpoint_topic, Float64, self.setpoint_callback)

    def setpoint_callback(self, msg):
        self.setpoint = msg.data
        self.last_setpoint_msg_time = rospy.Time.now()
        self.new_state_or_setpt = True

    def plant_state_callback(self, msg):
        self.plant_state = msg.data
        self.new_state_or_setpt = True

    def output(self):
        """
        Process a new state or setpoint update by:
          - Calculating error (setpoint - plant state)
          - Computing delta_t and updating error integral (with anti-windup)
          - Filtering error and its derivative
          - Computing PID control effort and publishing it via ROS
        """

        if not self.new_state_or_setpt:
            return

        current_error = self.setpoint - self.plant_state
        self.error.appendleft(current_error)

        current_time = rospy.Time.now()
        if self.prev_time is None:
            self.prev_time = current_time
            return

        delta_t = (current_time - self.prev_time).to_sec()
        if delta_t == 0:
            rospy.logerr("delta_t=0; jumping this loop. Current Time: {:.2f}".format(current_time.to_sec()))
            return
        self.prev_time = current_time

        self.error_integral += self.error[0] * delta_t
        self.error_integral = np.clip(self.error_integral, -self.windup_limit, self.windup_limit)

        filtered_error = self.filter_error.update(self.error[0])

        derivative_raw = (self.error[0] - self.error[1]) / delta_t

        filtered_deriv = self.filter_deriv.update(derivative_raw)

        self.proportional = self.Kp * filtered_error
        self.integral = self.Ki * self.error_integral
        self.derivative = self.Kd * filtered_deriv

        self.control_effort = self.proportional + self.integral + self.derivative
        self.control_effort = np.clip(self.control_effort, self.lower_limit, self.upper_limit)

        self.publish_control_effort()
        self.new_state_or_setpt = False

    def publish_control_effort(self):
        msg = Float64(data=self.control_effort)
        self.control_effort_pub.publish(msg)


    def run(self):
        """
        The main loop of the PID controller.
        Continuously calls the output() method at a fixed rate defined by self.rate_hz.
        """
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self.output()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("pid_controller")
    try:
        PidObject()
    except rospy.ROSInterruptException:
        pass
