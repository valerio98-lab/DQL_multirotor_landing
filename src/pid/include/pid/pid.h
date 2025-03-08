

#ifndef PID_H
#define PID_H

#include "ros/ros.h"
#include <iostream>
#include <ros/time.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
#include <stdio.h>
#include <string>

namespace pid_ns
{
class PidObject
{
public:
  PidObject();

  // Primary output variable
  double control_effort_ = 0;        // output of pid controller

private:
  void doCalcs();
  void getParams(double in, double& value, double& scale);
  void pidEnableCallback(const std_msgs::Bool& pid_enable_msg);
  void plantStateCallback(const std_msgs::Float64& state_msg);
  void printParameters();
  // void reconfigureCallback(pid::PidConfig& config, uint32_t level);
  void setpointCallback(const std_msgs::Float64& setpoint_msg);
  bool validateParameters();

  // Primary PID controller input variables
  double plant_state_;               // current output of plant
  bool pid_enabled_ = true;          // PID is enabled to run
  bool new_state_or_setpt_ = false;  // Indicate that fresh calculations need to be run
  double setpoint_ = 0;              // desired output of plant

  ros::Time prev_time_;
  ros::Time last_setpoint_msg_time_;
  ros::Duration delta_t_;

  double error_integral_ = 0;
  double proportional_ = 0;  // proportional term of output
  double integral_ = 0;      // integral term of output
  double derivative_ = 0;    // derivative term of output

  // PID gains
  double Kp_ = 0, Ki_ = 0, Kd_ = 0;

  // Parameters for error calc. with disconinuous input
  bool angle_error_ = false;
  double angle_wrap_ = 2.0 * 3.14159;

  // Cutoff frequency for the derivative calculation in Hz.
  // Negative -> Has not been set by the user yet, so use a default.
  double cutoff_frequency_ = -1;
  
  // Setpoint timeout parameter to determine how long to keep publishing
  // control_effort messages after last setpoint message
  // -1 indicates publish indefinately, and positive number sets the timeout
  double setpoint_timeout_ = -1;

  // Used in filter calculations. Default 1.0 corresponds to a cutoff frequency
  // at
  // 1/4 of the sample rate.
  double c_ = 1.;

  // Used to check for tan(0)==>NaN in the filter calculation
  double tan_filt_ = 1.;

  // Upper and lower saturation limits
  double upper_limit_ = 1000, lower_limit_ = -1000;

  // Anti-windup term. Limits the absolute value of the integral term.
  double windup_limit_ = 1000;

  // Initialize filter data with zeros
  std::vector<double> error_, filtered_error_, error_deriv_, filtered_error_deriv_;

  // Topic and node names and message objects
  ros::Publisher control_effort_pub_;
  ros::Publisher pid_debug_pub_;

  std::string topic_from_controller_, topic_from_plant_, setpoint_topic_, pid_enable_topic_;
  std::string pid_debug_pub_name_;
  std_msgs::Float64 control_msg_, state_msg_;

  // Diagnostic objects
  double min_loop_frequency_ = 1, max_loop_frequency_ = 1000;
  int measurements_received_ = 0;
};
}  // end pid namespace

#endif
