#!/bin/bash

source devel/setup.bash

screen -dmS env_launch bash -c "roslaunch dql_multirotor_landing landing_simulation.launch; exec bash" 

sleep 5

roslaunch dql_multirotor_landing test.launch
