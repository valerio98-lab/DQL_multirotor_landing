#!/bin/bash

source devel/setup.bash

screen -dmS env_launch bash -c "roslaunch dql_multirotor_landing landing_simulation.launch; exec bash" 

echo "Waiting for the simulation to start."
sleep 5

roslaunch dql_multirotor_landing simulation.launch
rosnode kill -a
rosnode cleanup
killall rosmaster
clear