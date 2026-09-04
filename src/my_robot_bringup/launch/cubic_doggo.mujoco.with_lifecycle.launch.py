from ament_index_python.packages import get_package_share_path, get_package_share_directory
from launch import LaunchDescription
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from moveit_configs_utils import MoveItConfigsBuilder
import os

######################################################################################################################
def generate_launch_description():
    pkg_share_path           = get_package_share_directory('my_robot_description')
    robot_description_path   = get_package_share_path('my_robot_description')
    robot_bringup_path       = get_package_share_path('my_robot_bringup')
    robot_moveit_config_path = get_package_share_path('cubic_doggo_moveit_config')   
 
    urdf_path          = os.path.join(robot_description_path,   'urdf',   'cubic_doggo.gazebo.xacro')
    moveit_config_path = os.path.join(robot_moveit_config_path, 'launch', 'move_group.launch.py')
    rviz_config_path   = os.path.join(robot_description_path,   'rviz',   'cubic_doggo.urdf_config.rviz')

    mujoco_ros2_bridge = Node(
        package="my_robot_commander_py",
        executable="cubic_doggo_mujoco_bridge", # Ensure this matches your setup.py entry point
        output="screen",
    )

    robot_description = ParameterValue(
        Command([
            'xacro ', 
            urdf_path]), 
        value_type=str) 
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {'robot_description': robot_description},
        ],
    )
    
    moveit_config = (
        MoveItConfigsBuilder("cubic_doggo", package_name="cubic_doggo_moveit_config")
        .robot_description(file_path=urdf_path)
        .to_moveit_configs()
    )
    moveit_launcher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(moveit_config_path),
        launch_arguments={}.items(),
    )
    lifecycle_node = Node(
        package="my_robot_commander",
        #executable="cubic_doggo_lifecycle",
        executable="cubic_doggo_lifecycle_imu",
        parameters=[
            moveit_config.robot_description,           # the URDF math
            moveit_config.robot_description_semantic,  # the SRDF 
            moveit_config.robot_description_kinematics,# the kinematics.yaml
            moveit_config.joint_limits,                # the joint_limits.yaml 
            {"jump_threshold": 0.15},                  # for computeCartesianPath
        ],
    )  
 
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_path],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
        ],
    )   

    joy_driver_node = Node(
        package="joy",
        executable="joy_node",
        parameters=[{
            "deadzone": 0.05,
            "autorepeat_rate": 20.0,
            # "device_id": 0            # force a specific controller if needed
        }]
    )
    joy_controller_node = Node(
        package="my_robot_controller",
        executable="cubic_doggo_joy_control",
        remappings=[("joy", "/joy")]
    )
    teleop_key_node = Node(
        package="my_robot_controller", 
        executable="cubic_doggo_teleop_key",
        output="screen",
        prefix="xterm -e", 
    )

    launch_entities = [
        SetParameter(name='use_sim_time', value=False),
        mujoco_ros2_bridge,
        robot_state_publisher_node,
        moveit_launcher,
        lifecycle_node,
        #rviz_node,
        joy_driver_node,
        joy_controller_node,
        teleop_key_node, 
    ]
    return LaunchDescription(launch_entities)

######################################################################################################################







