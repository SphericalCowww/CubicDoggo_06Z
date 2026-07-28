from ament_index_python.packages import get_package_share_path, get_package_share_directory
from launch import LaunchDescription
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

######################################################################################################################
def generate_launch_description():
    pkg_share_path           = get_package_share_directory('my_robot_description')
    robot_description_path   = get_package_share_path('my_robot_description')
    robot_bringup_path       = get_package_share_path('my_robot_bringup')
    robot_moveit_config_path = get_package_share_path('my_robot_moveit_config')   
 
    urdf_path          = os.path.join(robot_description_path,   'urdf',   'my_robot.gazebo.xacro')
    gazebo_config_path = os.path.join(robot_bringup_path,       'config', 'gazebo_bridge.yaml')
    robot_controllers  = os.path.join(robot_bringup_path,       'config', 'my_robot_controllers.yaml')
    moveit_config_path = os.path.join(robot_moveit_config_path, 'launch', 'move_group.launch.py')
    rviz_config_path   = os.path.join(robot_description_path,   'rviz',   'my_robot.urdf_config.rviz')

    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH', 
        value=os.path.join(pkg_share_path, '..'),
    )
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_path("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py"
            ),
        ),
        launch_arguments=[("gz_args", [" -r -v 0 empty.sdf"]),],
    )
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "robot_description"],
    )
    gz_ros2_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        parameters=[{"config_file": gazebo_config_path}],
    )
  
    robot_description = ParameterValue(Command(['xacro ', urdf_path]), value_type=str) 
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[
            {'robot_description': robot_description},
        ],
    )
    # check src/my_robot_bringup/config/my_robot_controllers.yaml 
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
        ],
    )
    leg1_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "leg1_controller",
        ],
    )
   
    moveit_launcher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(moveit_config_path),
        launch_arguments={}.items(),
    )
    commander_node = Node(
        package="my_robot_commander",
        executable="my_robot_commander",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_path],
    )
    
    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        set_gz_resource_path,
        gazebo,
        gz_spawn_entity,
        gz_ros2_bridge,
        robot_state_publisher_node,
        joint_state_broadcaster_spawner,
        leg1_controller_spawner,
        moveit_launcher,
        commander_node,
        #rviz_node,
    ])

######################################################################################################################
