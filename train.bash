#!/bin/bash

# Check if the catkin workspace has been built
if [ ! -d "devel" ]; then
    echo "Building the workspace with catkin_make..."
    catkin_make
fi

# Source the workspace setup
source devel/setup.bash

# Launch landing simulation in the background
echo "Starting landing simulation..."
exec roslaunch dql_multirotor_landing landing_simulation.launch 

# Wait a bit to ensure landing simulation is initialized
sleep 5

# Clear the screen before launching training
clear

# Run the training launch in the current terminal
echo "Starting training..."
roslaunch dql_multirotor_landing training.launch
