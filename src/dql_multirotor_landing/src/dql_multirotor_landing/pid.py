#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64, Bool, Float64MultiArray
import math

class PidObject:
    def __init__(self):
        # Inizializzazione dei vettori per errori e derivati (3 elementi ciascuno)
        self.error = [0.0, 0.0, 0.0]
        self.filtered_error = [0.0, 0.0, 0.0]
        self.error_deriv = [0.0, 0.0, 0.0]
        self.filtered_error_deriv = [0.0, 0.0, 0.0]
        self.error_integral = 0.0

        # Variabili di stato
        self.plant_state = 0.0
        self.control_effort = 0.0
        self.setpoint = 0.0
        self.pid_enabled = True
        self.new_state_or_setpt = False

        # Timing
        self.prev_time = None
        self.last_setpoint_msg_time = rospy.Time.now()

        # Termini PID
        self.proportional = 0.0
        self.integral = 0.0
        self.derivative = 0.0

        # Guadagni PID
        self.Kp = 0.0
        self.Ki = 0.0
        self.Kd = 0.0

        # Parametri per errori con input discontinui (es. angolari)
        self.angle_error = False
        self.angle_wrap = 2.0 * math.pi

        # Filtro derivativo
        self.cutoff_frequency = -1.0
        self.c = 1.0
        self.tan_filt = 1.0

        # Limiti di saturazione e anti-windup
        self.upper_limit = 1000.0
        self.lower_limit = -1000.0
        self.windup_limit = 1000.0

        # Timeout per il setpoint (-1: pubblica indefinitamente, altrimenti tempo in secondi)
        self.setpoint_timeout = -1.0

        # Parametri diagnostici
        self.min_loop_frequency = 1.0
        self.max_loop_frequency = 1000.0

        # Nomi dei topic
        self.topic_from_controller = ""
        self.topic_from_plant = ""
        self.setpoint_topic = ""
        self.pid_enable_topic = ""
        self.pid_debug_pub_name = ""

        # Attendere che il tempo ROS sia inizializzato (non zero)
        while rospy.Time.now().to_sec() == 0 and not rospy.is_shutdown():
            rospy.loginfo("controller spinning, waiting for time to become non-zero")
            rospy.sleep(1)

        # Lettura dei parametri dal parameter server (nello spazio dei nomi privato "~")
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
        self.pid_enable_topic = rospy.get_param(ns + "pid_enable_topic", "pid_enable")
        self.max_loop_frequency = rospy.get_param(ns + "max_loop_frequency", 1.0)
        self.min_loop_frequency = rospy.get_param(ns + "min_loop_frequency", 1000.0)
        self.pid_debug_pub_name = rospy.get_param(ns + "pid_debug_topic", "pid_debug")
        self.setpoint_timeout = rospy.get_param(ns + "setpoint_timeout", -1.0)
        assert self.setpoint_timeout == -1 or self.setpoint_timeout > 0, "setpoint_timeout must be -1 or > 0"
        self.angle_error = rospy.get_param(ns + "angle_error", False)
        self.angle_wrap = rospy.get_param(ns + "angle_wrap", 2.0 * math.pi)

        # Stampa dei parametri
        self.printParameters()
        if not self.validateParameters():
            rospy.logerr("Error: invalid parameter")

        # Inizializzazione dei publisher e subscriber
        self.control_effort_pub = rospy.Publisher(self.topic_from_controller, Float64, queue_size=1)
        self.pid_debug_pub = rospy.Publisher(self.pid_debug_pub_name, Float64MultiArray, queue_size=1)
        rospy.Subscriber(self.topic_from_plant, Float64, self.plantStateCallback)
        rospy.Subscriber(self.setpoint_topic, Float64, self.setpointCallback)
        rospy.Subscriber(self.pid_enable_topic, Bool, self.pidEnableCallback)

        # Attendere il primo messaggio per il setpoint e per lo stato della pianta
        self.wait_for_first_message(self.setpoint_topic, Float64, "setpoint")
        self.wait_for_first_message(self.topic_from_plant, Float64, "state from plant")

        # Impostare il tempo precedente
        self.prev_time = rospy.Time.now()

        # Avvio del ciclo di calcolo (simile al loop in Pid.cpp)
        self.run()

    def wait_for_first_message(self, topic, msg_type, description):
        while not rospy.is_shutdown():
            try:
                rospy.wait_for_message(topic, msg_type, timeout=10.0)
                break
            except rospy.ROSException:
                rospy.logwarn("Waiting for first {} message on topic: {}".format(description, topic))

    def setpointCallback(self, msg):
        self.setpoint = msg.data
        self.last_setpoint_msg_time = rospy.Time.now()
        self.new_state_or_setpt = True

    def plantStateCallback(self, msg):
        self.plant_state = msg.data
        self.new_state_or_setpt = True

    def pidEnableCallback(self, msg):
        self.pid_enabled = msg.data

    def getParams(self, in_val):
        digits = 0
        value = in_val
        while (abs(value) > 1.0 or abs(value) < 0.1) and (digits < 2 and digits > -1):
            if abs(value) > 1.0:
                value /= 10.0
                digits += 1
            else:
                value *= 10.0
                digits -= 1
        if value > 1.0:
            value = 1.0
        if value < -1.0:
            value = -1.0
        scale = math.pow(10.0, digits)
        return value, scale

    def validateParameters(self):
        if self.lower_limit > self.upper_limit:
            rospy.logerr("The lower saturation limit cannot be greater than the upper saturation limit.")
            return False
        return True

    def printParameters(self):
        rospy.loginfo("\nPID PARAMETERS\n-----------------------------------------")
        rospy.loginfo("Kp: {}, Ki: {}, Kd: {}".format(self.Kp, self.Ki, self.Kd))
        if self.cutoff_frequency == -1:
            rospy.loginfo("LPF cutoff frequency: 1/4 of sampling rate")
        else:
            rospy.loginfo("LPF cutoff frequency: {}".format(self.cutoff_frequency))
        rospy.loginfo("pid node name: {}".format(rospy.get_name()))
        rospy.loginfo("Name of topic from controller: {}".format(self.topic_from_controller))
        rospy.loginfo("Name of topic from the plant: {}".format(self.topic_from_plant))
        rospy.loginfo("Name of setpoint topic: {}".format(self.setpoint_topic))
        rospy.loginfo("Integral-windup limit: {}".format(self.windup_limit))
        rospy.loginfo("Saturation limits: {}/{}".format(self.upper_limit, self.lower_limit))
        rospy.loginfo("-----------------------------------------")

    def doCalcs(self):
        if self.new_state_or_setpt:
            if not ((self.Kp <= 0 and self.Ki <= 0 and self.Kd <= 0) or
                    (self.Kp >= 0 and self.Ki >= 0 and self.Kd >= 0)):
                rospy.logwarn("All three gains (Kp, Ki, Kd) should have the same sign for stability.")

            # Aggiornamento degli errori: lo slot 0 contiene l'errore corrente
            self.error[2] = self.error[1]
            self.error[1] = self.error[0]
            self.error[0] = self.setpoint - self.plant_state

            # Gestione dell'errore angolare (se abilitato)
            if self.angle_error:
                while self.error[0] < -self.angle_wrap/2.0:
                    self.error[0] += self.angle_wrap
                    self.error_deriv = [0.0, 0.0, 0.0]
                    self.error_integral = 0.0
                while self.error[0] > self.angle_wrap/2.0:
                    self.error[0] -= self.angle_wrap
                    self.error_deriv = [0.0, 0.0, 0.0]
                    self.error_integral = 0.0

            # Calcolo del delta_t
            current_time = rospy.Time.now()
            if self.prev_time is not None:
                delta_t = (current_time - self.prev_time).to_sec()
                self.prev_time = current_time
                if delta_t == 0:
                    rospy.logerr("delta_t is 0, skipping this loop. Possible overloaded CPU at time: {}".format(current_time.to_sec()))
                    return
            else:
                rospy.loginfo("prev_time is 0, doing nothing")
                self.prev_time = current_time
                return

            # Integrazione dell'errore
            self.error_integral += self.error[0] * delta_t
            if self.error_integral > abs(self.windup_limit):
                self.error_integral = abs(self.windup_limit)
            if self.error_integral < -abs(self.windup_limit):
                self.error_integral = -abs(self.windup_limit)

            # Calcolo del filtro per la derivata, se cutoff_frequency è impostato
            if self.cutoff_frequency != -1:
                self.tan_filt = math.tan((self.cutoff_frequency * 2 * math.pi) * delta_t / 2.0)
                if self.tan_filt <= 0 and self.tan_filt > -0.01:
                    self.tan_filt = -0.01
                if self.tan_filt >= 0 and self.tan_filt < 0.01:
                    self.tan_filt = 0.01
                self.c = 1.0 / self.tan_filt

            denom = (1 + self.c*self.c + 1.414*self.c)
            self.filtered_error[2] = self.filtered_error[1]
            self.filtered_error[1] = self.filtered_error[0]
            self.filtered_error[0] = (1.0/denom) * (self.error[2] + 2*self.error[1] + self.error[0] -
                                                    ((self.c*self.c - 1.414*self.c + 1) * self.filtered_error[2]) -
                                                    ((-2*self.c*self.c + 2) * self.filtered_error[1]))

            self.error_deriv[2] = self.error_deriv[1]
            self.error_deriv[1] = self.error_deriv[0]
            self.error_deriv[0] = (self.error[0] - self.error[1]) / delta_t

            self.filtered_error_deriv[2] = self.filtered_error_deriv[1]
            self.filtered_error_deriv[1] = self.filtered_error_deriv[0]
            self.filtered_error_deriv[0] = (1.0/denom) * (self.error_deriv[2] + 2*self.error_deriv[1] + self.error_deriv[0] -
                                                         ((self.c*self.c - 1.414*self.c + 1) * self.filtered_error_deriv[2]) -
                                                         ((-2*self.c*self.c + 2) * self.filtered_error_deriv[1]))

            # Calcolo del controllo PID
            self.proportional = self.Kp * self.filtered_error[0]
            self.integral = self.Ki * self.error_integral
            self.derivative = self.Kd * self.filtered_error_deriv[0]
            self.control_effort = self.proportional + self.integral + self.derivative

            # Applicazione dei limiti di saturazione
            if self.control_effort > self.upper_limit:
                self.control_effort = self.upper_limit
            elif self.control_effort < self.lower_limit:
                self.control_effort = self.lower_limit

            # Pubblicazione del controllo se il PID è abilitato e il timeout del setpoint non è scaduto
            if self.pid_enabled and (self.setpoint_timeout == -1 or (rospy.Time.now() - self.last_setpoint_msg_time).to_sec() <= self.setpoint_timeout):
                msg = Float64()
                msg.data = self.control_effort
                self.control_effort_pub.publish(msg)

                debug_msg = Float64MultiArray()
                debug_msg.data = [self.plant_state, self.control_effort, self.proportional, self.integral, self.derivative]
                self.pid_debug_pub.publish(debug_msg)
            elif self.setpoint_timeout > 0 and (rospy.Time.now() - self.last_setpoint_msg_time).to_sec() > self.setpoint_timeout:
                rospy.logwarn("Setpoint message timed out, will stop publishing control effort messages")
                self.error_integral = 0.0
            else:
                self.error_integral = 0.0

            self.new_state_or_setpt = False

    def run(self):
        rate = rospy.Rate(1000)  # Frequenza di loop: 1000 Hz
        while not rospy.is_shutdown():
            self.doCalcs()
            rospy.sleep(0.001)

