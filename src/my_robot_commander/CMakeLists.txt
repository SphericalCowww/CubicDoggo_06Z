cmake_minimum_required(VERSION 3.8)
project(my_robot_commander)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(my_robot_interface REQUIRED)
find_package(control_toolbox REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(onnxruntime REQUIRED)

include_directories(include)

###########################################################################
add_executable(cubic_doggo_lifecycle src/cubic_doggo_lifecycle.cpp)
ament_target_dependencies(cubic_doggo_lifecycle rclcpp moveit_ros_planning_interface moveit_core example_interfaces my_robot_interface)

add_executable(cubic_doggo_lifecycle_imu src/cubic_doggo_lifecycle_imu.cpp)
ament_target_dependencies(cubic_doggo_lifecycle_imu rclcpp moveit_ros_planning_interface moveit_core example_interfaces my_robot_interface control_toolbox)

add_executable(cubic_doggo_lifecycle_ppo src/cubic_doggo_lifecycle_ppo.cpp)
ament_target_dependencies(cubic_doggo_lifecycle_ppo rclcpp moveit_ros_planning_interface moveit_core example_interfaces my_robot_interface control_toolbox nlohmann_json onnxruntime)
 
install(TARGETS cubic_doggo_lifecycle cubic_doggo_lifecycle_imu cubic_doggo_lifecycle_ppo
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()





