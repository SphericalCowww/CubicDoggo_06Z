# CubicDoggo 06Z: Homemade 12-DOF 4-Legged Robot Recipe with Simulation and Reinforcement Learning

Cubic Doggo 06Z Neucommu is upgraded from [Cubic Doggo 06R High Mobility](https://github.com/SphericalCowww/CubicDoggo_06R). The goal is to integrate a simulation (Gazebo) and reinforcement learning (PyBullet) to control walking gait. 

## Moment of inertia from FreeCAD

To get the moment of inertia for a FreeCAD piece, run,

    # open FreeCAD > View > Panels > Python console
    obj = App.ActiveDocument.getObject("Body")
    shape = obj.Shape
    print(shape.Mass)
    print(shape.CenterOfMass)
    print(shape.MatrixOfInertia)

Then use this formula to convert to URDF value for each matrix element (I: moment of inertia, M: mass):

    I_urdf = I_freeCAD*(M_physical/M_freeCAD)*1.0E-6

The convex mesh can also be included for collision geometry; check out ``batteryHolder`` under ``CubicDoggo_06Z/src/my_robot_description/urdf/cubic_doggo.gazebo.xacro``.

Don't forget to check the basic geometry under rViz first:

    cd CubicDoggo_06Z
    colcon build
    source install/setup.bash
    ros2 launch my_robot_description cubic_doggo.rviz.launch.xacro.py

## Gazebo

To have a window for keyboard control:

    sudo apt install xterm

Reminder that ``~/.gz/sim/8/gui.config``contains GUI display options, such as camera location. Then run:

    cd CubicDoggo_06Z
    colcon build
    source install/setup.bash
    ros2 launch my_robot_bringup cubic_doggo.gazebo.with_lifecycle.launch.py
    ros2 run plotjuggler plotjuggler      # on another terminal

<img src="https://github.com/SphericalCowww/CubicDoggo_06Z/blob/main/fig_Gazebo.png" height="500">

## PyBullet

## References:

- CFD Intech, FreeCAD Tutorial | Exercise 6: How to Calculate Moment of Inertia of Model and Paste to Spreadsheet (<a href="https://www.youtube.com/watch?v=h6S0lKXxD3s">YouTube</a>) 

## Acknowledgements

- Some utility programs are adapted from the ROBOTIS DYNAMIXEL [Workbench examples](https://github.com/ROBOTIS-GIT/dynamixel-workbench/tree/main/dynamixel_workbench_toolbox/examples/sr) and [U2D2 Controller CAD](https://en.robotis.com/service/downloadpage.php?ca_id=70f0#). These files remain licensed under the Apache License 2.0. Modifications are documented in the source files.
- Raspberry Pi 5 CAD model sourced from [Printables](https://www.printables.com/model/607854-raspberry-pi-5).
- RPLIDAR A1M8 CAD model provided by [SLAMTEC](https://www.slamtec.com/en/support#rplidar-a-series).
