#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /home/kali/Documents/CubicDoggo_06Z/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export QT_QPA_PLATFORM=xcb

export ROS_DOMAIN_ID=63
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

exec ros2 launch my_robot_bringup cubic_doggo.with_lifecycle.launch.py




