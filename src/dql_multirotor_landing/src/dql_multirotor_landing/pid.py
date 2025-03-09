#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64, Float64MultiArray


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
        self.last_setpoint_msg_time: rospy.Time = rospy.Time.now()

        # Termini PID
        self.proportional = 0.0
        self.integral = 0.0
        self.derivative = 0.0

        # Filtro derivativo
        self.c = 1.0
        self.tan_filt = 1.0
        self.setpoint_timeout: float = -1.0  # type: ignore

        # Attendere che il tempo ROS sia inizializzato (non zero)
        while rospy.Time.now().to_sec() == 0 and not rospy.is_shutdown():
            rospy.loginfo("controller spinning, waiting for time to become non-zero")
            rospy.sleep(1)

        # Lettura dei parametri dal parameter server (nello spazio dei nomi privato "~")
        ns = "~"
        self.Kp: float = rospy.get_param(ns + "Kp", 1.0)  # type: ignore
        self.Ki: float = rospy.get_param(ns + "Ki", 0.0)  # type: ignore
        self.Kd: float = rospy.get_param(ns + "Kd", 0.0)  # type: ignore
        if not (
            (self.Kp <= 0 and self.Ki <= 0 and self.Kd <= 0)
            or (self.Kp >= 0 and self.Ki >= 0 and self.Kd >= 0)
        ):
            rospy.logwarn(
                "All three gains (Kp, Ki, Kd) should have the same sign for stability."
            )

        self.upper_limit: float = rospy.get_param(ns + "upper_limit", 1000.0)  # type: ignore
        self.lower_limit: float = rospy.get_param(ns + "lower_limit", -1000.0)  # type: ignore
        assert self.lower_limit < self.upper_limit, (
            "The lower saturation limit cannot be greater than the upper saturation limit."
        )

        self.windup_limit: float = rospy.get_param(ns + "windup_limit", 1000.0)  # type: ignore
        self.cutoff_frequency: float = rospy.get_param(ns + "cutoff_frequency", -1.0)  # type: ignore
        self.topic_from_controller: str = rospy.get_param(
            ns + "topic_from_controller", "control_effort"
        )  # type: ignore
        self.topic_from_plant: str = rospy.get_param(ns + "topic_from_plant", "state")  # type: ignore
        self.setpoint_topic: str = rospy.get_param(ns + "setpoint_topic", "setpoint")  # type: ignore

        self.max_loop_frequency: float = rospy.get_param(ns + "max_loop_frequency", 1.0)  # type: ignore
        self.min_loop_frequency: float = rospy.get_param(
            ns + "min_loop_frequency", 1000.0
        )  # type: ignore

        # Inizializzazione dei publisher e subscriber
        self.control_effort_pub = rospy.Publisher(
            self.topic_from_controller, Float64, queue_size=1
        )
        rospy.Subscriber(self.topic_from_plant, Float64, self.plant_state_callback)
        rospy.Subscriber(self.setpoint_topic, Float64, self.setpoint_callback)

        # Impostare il tempo precedente
        self.prev_time = rospy.Time.now()

        self.run()

    def setpoint_callback(self, msg):
        self.setpoint = msg.data
        self.last_setpoint_msg_time = rospy.Time.now()
        self.new_state_or_setpt = True

    def plant_state_callback(self, msg):
        self.plant_state = msg.data
        self.new_state_or_setpt = True

    def do_calcs(self):
        if self.new_state_or_setpt:
            # Aggiornamento degli errori: lo slot 0 contiene l'errore corrente
            self.error[2] = self.error[1]
            self.error[1] = self.error[0]
            self.error[0] = self.setpoint - self.plant_state

            # Calcolo del delta_t
            current_time = rospy.Time.now()
            if self.prev_time is not None:
                delta_t = (current_time - self.prev_time).to_sec()
                self.prev_time = current_time
                if delta_t == 0:
                    rospy.logerr(
                        "delta_t is 0, skipping this loop. Possible overloaded CPU at time: {}".format(
                            current_time.to_sec()
                        )
                    )
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

            denom = 1 + self.c * self.c + 1.414 * self.c
            self.filtered_error[2] = self.filtered_error[1]
            self.filtered_error[1] = self.filtered_error[0]
            self.filtered_error[0] = (1.0 / denom) * (
                self.error[2]
                + 2 * self.error[1]
                + self.error[0]
                - ((self.c * self.c - 1.414 * self.c + 1) * self.filtered_error[2])
                - ((-2 * self.c * self.c + 2) * self.filtered_error[1])
            )

            self.error_deriv[2] = self.error_deriv[1]
            self.error_deriv[1] = self.error_deriv[0]
            self.error_deriv[0] = (self.error[0] - self.error[1]) / delta_t

            self.filtered_error_deriv[2] = self.filtered_error_deriv[1]
            self.filtered_error_deriv[1] = self.filtered_error_deriv[0]
            self.filtered_error_deriv[0] = (1.0 / denom) * (
                self.error_deriv[2]
                + 2 * self.error_deriv[1]
                + self.error_deriv[0]
                - (
                    (self.c * self.c - 1.414 * self.c + 1)
                    * self.filtered_error_deriv[2]
                )
                - ((-2 * self.c * self.c + 2) * self.filtered_error_deriv[1])
            )

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

            msg = Float64()
            msg.data = self.control_effort
            self.control_effort_pub.publish(msg)

            debug_msg = Float64MultiArray()
            debug_msg.data = [
                self.plant_state,
                self.control_effort,
                self.proportional,
                self.integral,
                self.derivative,
            ]

            self.new_state_or_setpt = False

    def run(self):
        rate = 1000
        rospy.Rate(rate)  # Frequenza di loop: 1000 Hz
        while not rospy.is_shutdown():
            self.do_calcs()
            rospy.sleep(1 / rate)
