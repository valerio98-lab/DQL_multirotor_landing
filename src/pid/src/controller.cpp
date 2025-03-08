

#include <pid/pid.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "controller");

  pid_ns::PidObject my_pid;

  return 0;
}