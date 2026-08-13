# CubicDoggo 06Z: Homemade 12-DOF 4-Legged Robot Recipe with Simulation and Reinforcement Learning

Cubic Doggo 06Z Neucommu is upgraded from <a href="https://github.com/SphericalCowww/CubicDoggo_06R">Cubic Doggo 06R High Mobility</a>. The goal is to integrate a simulation (Gazebo) and reinforcement learning (PyBullet) to control walking gait. 

## Moment of inertia from FreeCAD
    # View > Panels > Python console
    obj = App.ActiveDocument.getObject("Body")
    shape = obj.Shape
    print(shape.Mass)
    print(shape.CenterOfMass)
    print(shape.MatrixOfInertia)

## Gazebo

For keyboard control:

    sudo apt install xterm

Adjust the PID direction:

    roll, negative when tilting backward
    pitch, negative when tilting leftward

Reminder that ``~/.gz/sim/8/gui.config``contains GUI display options, such as camera location. Then run:

    cd CubicDoggo_06Z
    colcon build
    source install/setup.bash
    ros2 launch my_robot_bringup cubic_doggo.gazebo.with_lifecycle.launch.py
    ros2 run plotjuggler plotjuggler      # on another terminal

## PyBullet

## References:

- CFD Intech, FreeCAD Tutorial | Exercise 6: How to Calculate Moment of Inertia of Model and Paste to Spreadsheet (<a href="https://www.youtube.com/watch?v=h6S0lKXxD3s">YouTube</a>) 

## Acknowledgements

- Some utility programs are adapted from the ROBOTIS DYNAMIXEL Workbench examples: https://github.com/ROBOTIS-GIT/dynamixel-workbench/tree/main/dynamixel_workbench_toolbox/examples/src. These files remain licensed under the Apache License 2.0. Modifications are documented in the source files.
