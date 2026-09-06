# CubicDoggo 06Z: Homemade 12-DOF 4-Legged Robot Recipe with Simulation and Reinforcement Learning

Cubic Doggo 06Z Neucommu is upgraded from [Cubic Doggo 06R High Mobility](https://github.com/SphericalCowww/CubicDoggo_06R). The goal is to integrate a simulation (Gazebo/Mujoco) and reinforcement learning (PyTorch) to control walking gait. 

Demos: Gazebo with Plotjuggler ([Reddit](https://www.reddit.com/r/ROS/comments/1vomb40/cubic_doggo_update_on_gazebo/)), Gazebo sim vs real ([Reddit](https://www.reddit.com/r/robotics/comments/1vpu2sd/cubic_doggo_found_a_nice_spot_on_the_ramp/)), MuJoCo sim ([Reddit](https://www.reddit.com/r/ROS/comments/1w8vm74/cubic_doggo_update_ros2_bridge_to_mujoco/)

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

In the figure, the robot walks up the ramp, stands, turns on the IMU, walks with IMU, and then just stands with IMU on:

  * top-left: Gazebo graphics
  * top-right: plotjuggler
    * left: all 12 servo positions, velocities, and loads set to 1.5N limit
    * right: IMU values (PID control to make roll (tilt left-sideways) and pitch (tilt backward) to 0) and their velocities
  * bottom-left: launch terminal outputs
  * bottom-center: xterm control command window

Adding the following line of 20 ms delay in IMU does cause oscillatory behavior in Gazebo sometimes (turn off for real robot!):

    vim CubicDoggo_06Z/src/my_robot_commander/src/cubic_doggo_lifecycle_imu.cpp 
    # std::this_thread::sleep_for(std::chrono::milliseconds(20)); // WARNING: deliberate delay for simulation!!!

<img src="https://github.com/SphericalCowww/CubicDoggo_06Z/blob/main/fig_Gazebo.webp" height="300">


## PyBullet

    cd CubicDoggo_06Z/
    cd ..                                                              # do NOT make the CubicDoggo_06Z_env/ inside CubicDoggo_06Z/, it will mess up colcon build
    python3 -m venv CubicDoggo_06Z_env/
    source CubicDoggo_06Z_env/bin/activate
    pip install --upgrade pip
    pip install "numpy<2"
    pip install setuptools jinja2 pyyaml typeguard
    pip install pybullet 
    pip install pin                                                     # for pinocchio
    pip install torch gymnasium stable-baselines3
    python3 -c "import torch; import pinocchio; import pybullet; import stable_baselines3; print('Installation Successful')"
    cd CubicDoggo_06Z/
    colcon build
    source install/setup.bash

Halfway through, realized MuJoCo is the modern way for robot simulation in python with CPU.

## MuJoCo

    cd CubicDoggo_06Z/
    cd ..                                                                # do NOT create the CubicDoggo_06Z_env/ inside CubicDoggo_06Z/, it will mess up colcon build
    python3 -m venv CubicDoggo_06Z_env/
    source CubicDoggo_06Z_env/bin/activate
    export PYTHONPATH=$PYTHONPATH:.../CubicDoggo_06Z_env/lib/python3.12/site-packages   # put it in .bashrc
    pip install --upgrade pip 
    pip install catkin_pkg empy lark
    pip install jinja2 pyyaml typeguard setuptools wheel
    pip install mujoco torch gymnasium stable-baselines3 pin             # pin for pinocchio
    python3 -c "import torch; import pinocchio; import mujoco; import stable_baselines3; print('Installation Successful')"
    cd CubicDoggo_06Z/
    rm -r install/ build/ log/
    colcon build
    source install/setup.bash

### Launch MuJoCo with ROS2

    cd CubicDoggo_06Z/
    source ../CubicDoggo_06Z_env/bin/activate
    colcon build --cmake-clean-first
    source install/setup.bash
    ros2 run my_robot_commander_py cubic_doggo_mujoco_read_urdf 
    # toggle 1 for visual, and toggle 0 for collision

Also to do a one-leg test, change ``CubicDoggo_06Z/src/my_robot_commander_py/my_robot_commander_py/cubic_doggo_mujoco_read_urdf.py``:

    xacro_path     = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.urdf.xacro')
    mjcf_path      = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.mujoco.xml')
    # to 
    xacro_path     = os.path.join(pkg_share_path, 'urdf', 'cubic_leg1.urdf.xacro')
    mjcf_path      = os.path.join(pkg_share_path, 'urdf', 'cubic_leg1.mujoco.xml')

To launch with commander/controller:
    
    ros2 launch my_robot_commander_py cubic_doggo.mujoco.with_lifecycle.launch.py
    ros2 run plotjuggler plotjuggler      # on another terminal

Note that this ROS2 to MuJoCo bridge does not use mujoco_ros2_control, but action server and joint state publisher instead. The reaction should rightfully be slower and less deterministic, but MuJoCo is plenty fast that its speed actually needs to be tuned by actuator PID under ``CubicDoggo_06Z/src/my_robot_description/urdf/cubic_doggo.mujoco.xml``.

## References:

- CFD Intech, FreeCAD Tutorial | Exercise 6: How to Calculate Moment of Inertia of Model and Paste to Spreadsheet (<a href="https://www.youtube.com/watch?v=h6S0lKXxD3s">YouTube</a>) 

## Acknowledgements

- Some utility programs are adapted from the ROBOTIS DYNAMIXEL [Workbench examples](https://github.com/ROBOTIS-GIT/dynamixel-workbench/tree/main/dynamixel_workbench_toolbox/examples/sr) and [U2D2 Controller CAD](https://en.robotis.com/service/downloadpage.php?ca_id=70f0#). These files remain licensed under the Apache License 2.0. Modifications are documented in the source files.
- Raspberry Pi 5 CAD model sourced from [Printables](https://www.printables.com/model/607854-raspberry-pi-5).
- RPLIDAR A1M8 CAD model provided by [SLAMTEC](https://www.slamtec.com/en/support#rplidar-a-series).
