# CubicDoggo 06Z: Homemade 12-DOF 4-Legged Robot Recipe with Simulation and Reinforcement Learning

Cubic Doggo 06Z Neucommu is upgraded from <a href="https://github.com/SphericalCowww/CubicDoggo_06R">Cubic Doggo 06R High Mobility</a>. The goal is to incorporate a simulation (Gazebo) and reinforcement learning (PyBullet) for walking gait control. 

## Gazebo

Simply:

    cd CubicDoggo_06Z
    colcon build
    source install/setup.bash
    ros2 launch my_robot_bringup cubic_doggo.gazebo.with_lifecycle.launch.py

## PyBullet

## Acknowledgements

- Some utility programs are adapted from the ROBOTIS DYNAMIXEL Workbench examples: https://github.com/ROBOTIS-GIT/dynamixel-workbench/tree/main/dynamixel_workbench_toolbox/examples/src. These files remain licensed under the Apache License 2.0. Modifications are documented in the source files.
