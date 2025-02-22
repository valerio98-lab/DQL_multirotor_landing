# DQL_multirotor_landing

Implementation of Double Q-Learning with Curriculum Learning for Autonomous UAV Landing on a Moving Platform 

## How to run

After each update just do

```bash
catkin_make
source devel/setup.bash
```

on the workspace root
# Autonomous Drone Landing on Moving Platform

Implementation of Double Q-Learning with Curriculum Learning for Autonomous UAV Landing on a Moving Platform 

## Prerequisites

1. **Operating System:** Ubuntu 20.04
2. **ROS Version:** ROS Noetic Full Desktop

## Installation Guide

### Step 1: Install ROS Noetic Full Desktop

Follow the official installation guide for ROS Noetic:  
[ROS Noetic Installation](http://wiki.ros.org/noetic/Installation/Ubuntu)

### Step 2: Clone the Repository

```bash
git clone https://github.com/valerio98-lab/DQL_multirotor_landing.git
```
```bash
cd DQL_multirotor_landing
```

### Step 3: Install Dependencies and Build
```bash
pip install -r requirements.txt
```

```bash
catkin_make
```

```bash
catkin_make
```

```bash
source devel/setup.bash
```

### Step 4: Training an Agent
```bash
chmod +x training.sh
```

```bash
./training.sh
```

### Step 5: Testing an Agent
```bash
chmod +x test.sh
```

```bash
./test.sh
```






