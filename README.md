# CubicDoggo 06Z: Homemade 12-DOF 4-Legged Robot Recipe with Simulation and Reinforcement Learning

Cubic Doggo 06Z Neucommu is upgraded from <a href="https://github.com/SphericalCowww/CubicDoggo_06R">Cubic Doggo 06R High Mobility</a>. The goal is to incorporate a simulation and reinforcement learning for walking gait control; IMU is required.

## Setting up Isaac Sim & Issac Lab

### Hardware Reference

Planned hardware spec (<a href="https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html">requirement reference</a>):

  * AMD Ryzen 7 7800X3D - 8 x 4.2 GHz (TurboBoost bis 5.00 GHz, 8 Kerne / 16 Threads, 96MB Cache)
  * NVIDIA GeForce RTX 5080 (16GB GDDR7)
  * 64 GB DDR5 RAM
  * 2000 GB M.2 SSD
  * Gigabit Ethernet LAN, Wi-Fi 6
  * 6x USB 3.2, 4x USB 2.0, 2x HDMI 2.1a, 3x DisplayPort 1.4a, 1x RJ-45, 1x Mikrophon, 1x Kopfhörer, Line-In/ Line-Out/ Mikrofon
  * Xilence Performance A+ M705D => updated to: be quiet! Pure Loop 2 FX 240mm
  * 850 Watt
  * 210 x 430 x 444 mm

### Installation

For ROS2 Jazzy installation with IMU, check (<a href="https://github.com/SphericalCowww/CubicDoggo_06R">GitHub</a>). To install Isaac Sim & Issac Lab,

    sudo apt update
    sudo apt upgrade
    sudo apt install python3.12-venv python3.12-dev build-essential cmake -y
    python3.12 -m venv isaaclab_env
    source isaaclab_env/bin/activate
    pip install --upgrade pip uv
    pip install isaacsim[compatibility-check]   # do not install any other packages yet
    pip install packaging setuptools
    isaacsim isaacsim.exp.compatibility_check   # if see orange: Settings => Power => Power Mode => Performance
    uv pip install "isaacsim[all,extscache]==6.0.1" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow
    # installed under isaaclab_env/lib/python3.12/site-packages/isaacsim/
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    git clone https://github.com/isaac-sim/IsaacLab.git isaaclab
    cd isaaclab
    ./isaaclab.sh --install
    deactivate

    sudo snap install code --classic   # installing Visual Studio Code
    sudo apt install nvtop             # for monitoring GPU
    nvtop                              # for monitoring GPU
    sudo apt install psensor           # check psensor App, for monitoring temperatures 
    # check also APP "NVIDIA X Server Settings" to adjust GPU settings

And to check for the installation:

    source /opt/ros/jazzy/setup.bash
    source isaaclab_env/bin/activate
    python3 -c "import rclpy"               # check if connected to ROS2 Jazzy installed
    isaacsim                                # wait a bit if not responding
    # the following are Isaac Lab trainings
    python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --num_envs=1000
    python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless
    # check "Mean reward", wait 1 hr
    python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --num_envs=20
    # to find the saved trained files:
    cd isaaclab/logs/rsl_rl/anymal_c_rough
    
    # on another window
    nvtop                                  # checking if GPU is being used

### URDF Import

    ros2 run xacro xacro cubic_doggo.urdf.xacro > cubic_doggo.urdf
    # change: 
    ## package:/ => full path
    # remove: 
    ## </joint><link name="world"/> ## <joint name="world_base_link" type="fixed">
    ##   <parent link="world"/>
    ##   <child  link="base_link"/>
    ## <origin xyz="0 0 0" rpy="0 0 0"/>

    # rm ~/.cache/ # if needed
    isaacsim
    # File => Import => /home/cubicdoggo/Documents/CubicDoggo/src/my_robot_description/urdf/cubic_doggo.urdf
    # Model => Create in Stage
    # Links => Movable Base 
    # Joints & Drives => Joint Configuration => Stiffness
    # Collider Type => Convex Decomposition
    # Import
    # Check: Console (bottom left) or Terminal for errors

    # Stage Light (center-right top) => Grey Studio 
    # Create => Physics => Ground plane => pull it under World
    ## Rotate the robot upright
    # Tools => Physics => Physics Inspector => cubic_doggo (defaultPrim) => Refresh
    ## Move joints to the correct positions (although inaccurate) => click green tick to confirm
    # Run (left panel)
    # save as: /home/cubicdoggo/Documents/CubicDoggo/cubicdoggo_isaaclab/usd/cubic_doggo.usd

### Creating Isaac Lab Workflow

    isaaclab.sh --new
    # Task type: External
    # ...
    # Isaac Lab workflow: Manager-based | single-agent
    # RL library: rsl_rl

    # reference to /home/cubicdoggo/Documents/isaaclab/source/isaaclab_assets/isaaclab_assets/robots/cubic_doggo.py

<img src="https://github.com/SphericalCowww/ROS_leggedRobot_testBed/blob/main/IsaacSimImport.png" width="600">


## Acknowledgements

- Some utility programs are adapted from the ROBOTIS DYNAMIXEL Workbench examples: https://github.com/ROBOTIS-GIT/dynamixel-workbench/tree/main/dynamixel_workbench_toolbox/examples/src. These files remain licensed under the Apache License 2.0. Modifications are documented in the source files.
