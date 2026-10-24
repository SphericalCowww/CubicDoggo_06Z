#include <memory>
#include <thread>
#include <atomic>
#include <string>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "moveit/move_group_interface/move_group_interface.hpp"
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_trajectory/robot_trajectory.hpp>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.hpp>
#include <moveit/trajectory_processing/ruckig_traj_smoothing.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/action/execute_trajectory.hpp>
 
using rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface;
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
using ExecuteTrajectory  = moveit_msgs::action::ExecuteTrajectory;
using namespace std::placeholders;                  // for using _1, _2
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//https://github.com/ros2/example_interfaces/tree/rolling/msg
#include "std_srvs/srv/set_bool.hpp"
#include "std_srvs/srv/trigger.hpp"
#include <example_interfaces/msg/string.hpp>
#include <example_interfaces/msg/float64_multi_array.hpp>    //variable size
#include "my_robot_interface/msg/cubic_doggo_leg_pose_target.hpp"
#include "my_robot_interface/msg/cubic_doggo_leg_feet_target.hpp"
#include <example_interfaces/msg/bool.hpp>
using ros_string        = example_interfaces::msg::String;
using ros_array         = example_interfaces::msg::Float64MultiArray;
using custom_pose_array = my_robot_interface::msg::CubicDoggoLegPoseTarget;
using custom_feet_array = my_robot_interface::msg::CubicDoggoLegFeetTarget;
using ros_bool          = example_interfaces::msg::Bool;

const double DEFAULT_VEL_SCALE = 0.2;
const double DEFAULT_ACC_SCALE = 0.05;

#include <control_toolbox/pid.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
static const std::string DEPLOY_DIR = ament_index_cpp::get_package_share_directory("my_robot_commander_py") 
                                    + "/deploy_model/";
static const std::string VEC_NORM_PATH   = DEPLOY_DIR + "policy1_force_privilegedHeight_vecNorm.json";
static const std::string ONNX_MODEL_PATH = DEPLOY_DIR + "policy1_force_privilegedHeight_model.onnx";
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class CubicDoggoLifecycleManager : public rclcpp_lifecycle::LifecycleNode {
public:
    explicit CubicDoggoLifecycleManager(const rclcpp::NodeOptions & options)
    : rclcpp_lifecycle::LifecycleNode("cubic_doggo_lifecycle", options) {
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:constructor(): %s", current_lifecycle_state_.c_str());

        callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        moveit_node_ = std::make_shared<rclcpp::Node>("moveit_interface_node", this->get_namespace(), options);
        exec_action_client_ = rclcpp_action::create_client<ExecuteTrajectory>(moveit_node_, "/execute_trajectory");
 
        current_lifecycle_state_ = "state_initialized";
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:constructor(): %s", current_lifecycle_state_.c_str());
    }
    CallbackReturn on_configure(const rclcpp_lifecycle::State &) override {
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:on_configure(): %s", current_lifecycle_state_.c_str());

        auto now = this->now();
        bool use_sim_time = false;
        this->get_parameter("use_sim_time", use_sim_time);    
        if (use_sim_time == true) {
            RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:on_configure(): "
                                      "simulation does not require clock synchronization");
        } else if (now.seconds() < 1577836800) { // timestamp for 2020
            RCLCPP_ERROR(get_logger(), "CubicDoggoLifecycleManager:on_configure(): system clock not synced " 
                                       "now at %.6f sec (%ld ns)", now.seconds(), now.nanoseconds());
            return CallbackReturn::FAILURE;
        } else {
            RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:on_configure(): system clock now synced "
                                      "at %.6f sec (%ld ns)", now.seconds(), now.nanoseconds());
        }
        if (!exec_action_client_->wait_for_action_server(std::chrono::seconds(2))) {
            RCLCPP_ERROR(get_logger(), "CubicDoggoLifecycleManager:on_configure(): "
                                       "ExecuteTrajectory action server not available");
            return CallbackReturn::FAILURE;
        }

        for (std::size_t legIdx = 0; legIdx < legN_; legIdx++) {
            leg_interface_[legIdx] = std::make_shared<MoveGroupInterface>(moveit_node_, planning_group_[legIdx]);
            if (!leg_interface_[legIdx]->startStateMonitor(2.0)) {
                RCLCPP_ERROR(get_logger(), "CubicDoggoLifecycleManager:on_configure(): "
                                           "MoveGroupInterface state %s invalide",
                             planning_group_[legIdx].c_str());
                return CallbackReturn::FAILURE;
            }
        }
        all_legs_interface_ = std::make_shared<MoveGroupInterface>(moveit_node_, all_legs_planning_group_);
        if (!all_legs_interface_->startStateMonitor(2.0)) {
            RCLCPP_ERROR(get_logger(), "CubicDoggoLifecycleManager:on_configure(): "
                                       "MoveGroupInterface state %s invalide", 
                         all_legs_planning_group_.c_str());
            return CallbackReturn::FAILURE;
        }
        for (std::size_t legIdx = 0; legIdx < legN_; legIdx++) {
            leg_interface_[legIdx]->setEndEffectorLink(endEffector_link_[legIdx]);
            leg_interface_[legIdx]->setGoalPositionTolerance(0.01);
            leg_interface_[legIdx]->setGoalOrientationTolerance(0.1);
            leg_interface_[legIdx]->setWorkspace(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0); // world size
            leg_interface_[legIdx]->setNumPlanningAttempts(10);
            leg_interface_[legIdx]->setPlanningTime(1.0);
        }
        all_legs_interface_->setGoalPositionTolerance(0.01);
        all_legs_interface_->setGoalOrientationTolerance(0.1);
        all_legs_interface_->setWorkspace(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0); // world size
        all_legs_interface_->setNumPlanningAttempts(10);
        all_legs_interface_->setPlanningTime(1.0);
        setDefaultVelAccScaler_(DEFAULT_VEL_SCALE, DEFAULT_ACC_SCALE);
 
        auto temp_node = std::make_shared<rclcpp::Node>("temp_param_fetcher");
        auto param_client = std::make_shared<rclcpp::SyncParametersClient>(temp_node, "/controller_manager");
        if (param_client->wait_for_service(std::chrono::seconds(3))) {
            if (param_client->has_parameter("update_rate")) {
                update_rate_ = param_client->get_parameter<int>("update_rate");
                RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:on_configure(): "
                                          "successfully synced update_rate: %d Hz", update_rate_);
            } else {
                RCLCPP_WARN(get_logger(), "CubicDoggoLifecycleManager:on_configure(): "
                                          "'update_rate' param missing in controller_manager. Using 100Hz.");
            }
        } else {
            RCLCPP_ERROR(get_logger(), "CubicDoggoLifecycleManager:on_configure(): controller_manager not found");
        }

        state_service_ = this->create_service<std_srvs::srv::Trigger>(
            "get_robot_state",
            std::bind(&CubicDoggoLifecycleManager::handleGetState_, this, _1, _2)); 
        walk_service_ = this->create_service<std_srvs::srv::SetBool>(
            "leg_walk_toggle", 
            std::bind(&CubicDoggoLifecycleManager::handleWalkRequest_, this, _1, _2), 
            rclcpp::ServicesQoS(), 
            callback_group_);
        imu_service_ = this->create_service<std_srvs::srv::SetBool>(
            "leg_imu_toggle",
            std::bind(&CubicDoggoLifecycleManager::handleIMUrequest_, this, _1, _2), 
            rclcpp::ServicesQoS(), 
            callback_group_);
        onnx_service_ = this->create_service<std_srvs::srv::SetBool>(
            "leg_onnx_toggle",
            std::bind(&CubicDoggoLifecycleManager::handleONNXrequest_, this, _1, _2), 
            rclcpp::ServicesQoS(),
            callback_group_);

        for (std::size_t legIdx = 0; legIdx < legN_; ++legIdx) {
            loadCurrentRobotState_(legIdx);
            RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:on_configure(): "
                                      "current end effector (x, y, z) = (%lf, %lf, %lf)",
                        endEffector_x_[legIdx], endEffector_y_[legIdx], endEffector_z_[legIdx]);
        }

        joint_publisher_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/all_legs_controller/joint_trajectory", 10);
        all_legs_robot_model_ = all_legs_interface_->getRobotModel();
        last_walk_state_ = std::make_shared<moveit::core::RobotState>(all_legs_robot_model_);
        last_walk_state_->setToDefaultValues();
        
        for (std::size_t legIdx = 0; legIdx < legN_; ++legIdx) {
            imu_z_corr_[legIdx] = 0.0;
        }

        if (initializeOnnxModel_() == false) {
            RCLCPP_ERROR(get_logger(), "CubicDoggoLifecycleManager:on_configure(): onnx init failed");
            return CallbackReturn::FAILURE;
        }

        current_lifecycle_state_ = "state_configured";
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:on_configure(): %s", current_lifecycle_state_.c_str());
        return CallbackReturn::SUCCESS;
    }
    CallbackReturn on_activate(const rclcpp_lifecycle::State &) override {
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:on_activation(): %s", current_lifecycle_state_.c_str());

        auto sub_options = rclcpp::SubscriptionOptions();
        sub_options.callback_group = callback_group_;
        leg_named_subscriber_ = create_subscription<ros_string>  ("/leg_set_named", 10,
            std::bind(&CubicDoggoLifecycleManager::legNamedCallback_, this, _1), sub_options);
        leg_joint_subscriber_ = create_subscription<ros_array>   ("/leg_set_joint", 10,
            std::bind(&CubicDoggoLifecycleManager::legJointCallback_, this, _1), sub_options);
        leg_pose_subscriber_  = create_subscription<custom_pose_array>("/leg_set_pose",  10,
            std::bind(&CubicDoggoLifecycleManager::legPoseCallback_,  this, _1), sub_options);
        leg_feet_subscriber_  = create_subscription<custom_feet_array>("/leg_set_feet",  10,
            std::bind(&CubicDoggoLifecycleManager::legFeetCallback_,  this, _1), sub_options);

        imu_subscriber_ = create_subscription<sensor_msgs::msg::Imu>(
            "/imu_broadcaster/imu", 10, [this](const sensor_msgs::msg::Imu::SharedPtr msg) {


            //std::this_thread::sleep_for(std::chrono::milliseconds(20)); // WARNING: deliberate delay for simulation!


            raw_gyro_x_.store(msg->angular_velocity.x, std::memory_order_relaxed);
            raw_gyro_y_.store(msg->angular_velocity.y, std::memory_order_relaxed);
            raw_gyro_z_.store(msg->angular_velocity.z, std::memory_order_relaxed);

            tf2::Quaternion qObj(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
            tf2::Matrix3x3 matrixObj(qObj);
            double roll, pitch, yaw;
            matrixObj.getRPY(roll, pitch, yaw);
            raw_roll_rad_ .store(roll,  std::memory_order_relaxed);
            raw_pitch_rad_.store(pitch, std::memory_order_relaxed);
            raw_yaw_rad_  .store(yaw, std::memory_order_relaxed);
            double raw_pitch = pitch*(180.0/M_PI);  // use if negative when tilting leftward
            //double raw_pitch =-pitch*(180.0/M_PI);  // use if negative when tilting rightward
            double raw_roll  = roll *(180.0/M_PI);  // use if negative when tilting forward
            //double raw_roll  =-roll *(180.0/M_PI);  // use if negative when tilting backward
            double alpha = 0.8;                     // [0.0, 1.0], lower the smoother but with lags
            double current_pitch = alpha*raw_pitch + (1.0 - alpha)*current_pitch_.load();
            double current_roll  = alpha*raw_roll  + (1.0 - alpha)*current_roll_.load();
            current_pitch_.store(current_pitch, std::memory_order_relaxed);
            current_roll_ .store(current_roll, std::memory_order_relaxed);

            double raw_pitch_vel = msg->angular_velocity.y*(180.0/M_PI);
            double raw_roll_vel  = msg->angular_velocity.x*(180.0/M_PI);
            double current_pitch_vel = alpha*raw_pitch_vel + (1.0 - alpha)*current_pitch_vel_.load();// share alpha
            double current_roll_vel  = alpha*raw_roll_vel  + (1.0 - alpha)*current_roll_vel_.load(); // for phase sync
            current_pitch_vel_.store(current_pitch_vel, std::memory_order_relaxed);
            current_roll_vel_ .store(current_roll_vel,  std::memory_order_relaxed);
        });

        joint_state_subscriber_ = create_subscription<sensor_msgs::msg::JointState>("/joint_states", 10,
            [this](sensor_msgs::msg::JointState::SharedPtr msg) { joint_state_buf_.writeFromNonRT(*msg); },
            sub_options);

        keep_running_thread_ = true;
        is_walking_          = false; 
        is_imu_              = false; 
        is_onnx_             = false;
        walking_thread_      = std::thread(&CubicDoggoLifecycleManager::controlLoop_, this);

        current_lifecycle_state_ = "state_stationary";
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:on_activation(): %s", current_lifecycle_state_.c_str());
        return CallbackReturn::SUCCESS;
    }
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override {
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:on_deactivation(): %s", 
                    current_lifecycle_state_.c_str());
        setDefaultVelAccScaler_(DEFAULT_VEL_SCALE, DEFAULT_ACC_SCALE);

        leg_named_subscriber_.reset();
        leg_joint_subscriber_.reset();
        leg_pose_subscriber_.reset();
        joint_state_subscriber_.reset();

        keep_running_thread_ = false;
        is_walking_          = false;
        is_imu_              = false;
        is_onnx_             = false;
        policy_idx_resolved_ = false;
        if (walking_thread_.joinable()) {
            walking_thread_.join();
        }

        for (std::size_t legIdx = 0; legIdx < legN_; legIdx++) {
            leg_interface_[legIdx]->stop();
        }
        all_legs_interface_->stop();
        current_lifecycle_state_ = "state_stopped";
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:on_deactivation(): %s", 
                                  current_lifecycle_state_.c_str());
        return CallbackReturn::SUCCESS;
    }
    CallbackReturn on_cleanup(const rclcpp_lifecycle::State &) override {
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:on_cleanup(): %s", current_lifecycle_state_.c_str());
        setDefaultVelAccScaler_(DEFAULT_VEL_SCALE, DEFAULT_ACC_SCALE);

        for (std::size_t legIdx = 0; legIdx < legN_; legIdx++) {
            leg_interface_[legIdx].reset();
        }
        all_legs_interface_.reset();
        leg_named_subscriber_.reset();
        leg_joint_subscriber_.reset();
        leg_pose_subscriber_.reset();
        joint_state_subscriber_.reset();
        ort_session_.reset(); ort_env_.reset(); 

        current_lifecycle_state_ = "state_configured";
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:on_clean(): %s", current_lifecycle_state_.c_str());
        return CallbackReturn::SUCCESS;
    }
    CallbackReturn on_shutdown(const rclcpp_lifecycle::State &) override {
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:on_shutdown(): %s", current_lifecycle_state_.c_str());
        setDefaultVelAccScaler_(DEFAULT_VEL_SCALE, DEFAULT_ACC_SCALE);

        keep_running_thread_ = false;
        is_walking_          = false;
        is_imu_              = false;
        is_onnx_             = false;
        policy_idx_resolved_ = false;
        if (walking_thread_.joinable()) {
            walking_thread_.join();
        }
        
        for (std::size_t legIdx = 0; legIdx < legN_; legIdx++) {
            leg_interface_[legIdx]->stop();
        }
        all_legs_interface_->stop();
    
        current_lifecycle_state_ = "state_stopped";
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:on_shutdown(): %s", current_lifecycle_state_.c_str());
        return CallbackReturn::SUCCESS;
    }
    rclcpp::Node::SharedPtr get_moveit_node() const { 
        return moveit_node_; 
    }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
private:
    void legNamedCallback_(const ros_string::SharedPtr msg) {
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:legNamedCallback(): command received");
        setDefaultVelAccScaler_(DEFAULT_VEL_SCALE, DEFAULT_ACC_SCALE);
        legNamedTarget_(msg->data);
    }
    void legJointCallback_(const ros_array::SharedPtr msg) {
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:legJointCallback(): command received");
        setDefaultVelAccScaler_(DEFAULT_VEL_SCALE, DEFAULT_ACC_SCALE);
        if (msg->data.size() != (1+jointNperLeg_)) {
            RCLCPP_WARN(get_logger(), "CubicDoggoLifecycleManager:legJointCallback(): message length mismatch");
            return;
        }
        legJointTarget_(msg->data);
    }
    void legPoseCallback_(const custom_pose_array::SharedPtr msg) {
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:legPoseCallback(): command received");
        setDefaultVelAccScaler_(DEFAULT_VEL_SCALE, DEFAULT_ACC_SCALE);
        legPoseTarget_(msg->leg_index, msg->x, msg->y, msg->z);
    }
    void legFeetCallback_(const custom_feet_array::SharedPtr msg) {
        if ((msg->x != target_x_stride_) || ((msg->y != target_y_stride_))) {
            RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:legFeetCallback(): command received, "
                                      "(x, y) = (%lf, %lf)", msg->x, msg->y);
            if (msg->x < -1.0) {
                target_x_stride_ = -1.0;
            } else if (1.0 < msg->x) {
                target_x_stride_ = 1.0;
            } else {
                target_x_stride_ = msg->x;
            }
            if (msg->y < -0.5) {
                target_y_stride_ = -0.5;
            } else if (1.0 < msg->y) {
                target_y_stride_ = 1.0;
            } else {
                target_y_stride_ = msg->y;
            }
        }
    }
    ///////////
    void legNamedTarget_(const std::string &name) {
        is_busy_ = true;
        if (name == "stand") {
            all_legs_interface_->setNamedTarget("stand_mid");
            idle_name_ = "stand_mid";
            planAndExecute_();
            RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:legNamedTarget_(): "
                                      "idle_name_ = %s",idle_name_.c_str());
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        all_legs_interface_->setNamedTarget(name);
        idle_name_ = name;
        planAndExecute_(); 
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:legNamedTarget_(): idle_name_ = %s",idle_name_.c_str());

        if (name == "stand") {
            control_initialized_ = false; 
        }
        is_busy_ = false;
    }
    void legJointTarget_(const std::vector<double> &joints) {
        is_busy_ = true;
        std::size_t legIdx = static_cast<std::size_t>(joints[0]);
        all_legs_current_robot_state_ = all_legs_interface_->getCurrentState(2.0);
        const std::vector<std::string>& leg_joint_names = leg_interface_[legIdx]->getJointNames();
        for (std::size_t jointIdx = 0; jointIdx < (joints.size()-1); jointIdx++) {
            double joint_val = joints[jointIdx + 1];
            all_legs_current_robot_state_->setJointPositions(leg_joint_names[jointIdx], &joint_val);
        }
        all_legs_interface_->setJointValueTarget(*all_legs_current_robot_state_);
        planAndExecute_();
        is_busy_ = false;
    }
    void legPoseTarget_(int legIdxInput, double x, double y, double z) {
        is_busy_ = true;
        std::size_t legIdx = static_cast<std::size_t>(legIdxInput);
        geometry_msgs::msg::Pose target_pose = endEffector_pose_[legIdx].pose;
        target_pose.position.x = x;
        target_pose.position.y = y;
        target_pose.position.z = z;
            
        success_ = leg_interface_[legIdx]->setApproximateJointValueTarget(target_pose, endEffector_link_[legIdx]);
        if (success_ == false) {
            RCLCPP_ERROR(get_logger(), "CubicDoggoLifecycleManager:legPoseTarget_(): "
                                       "failed to find IK solution for target!");
            return;
        }
        std::vector<double> leg_joints;
        leg_interface_[legIdx]->getJointValueTarget(leg_joints);
        leg_joints.insert(leg_joints.begin(), static_cast<double>(legIdx));
        legJointTarget_(leg_joints);
            
        loadCurrentRobotState_(legIdx);
        to_target_dist_ = std::sqrt(std::pow(endEffector_x_[legIdx] - x, 2) + 
                                    std::pow(endEffector_y_[legIdx] - y, 2) +
                                    std::pow(endEffector_z_[legIdx] - z, 2));
        if (to_target_dist_.load() > to_target_dist_thres_.load()) {
            RCLCPP_WARN(get_logger(), "CubicDoggoLifecycleManager:legPoseTarget_(): "
                                      "target unreachable (likely hit a joint limit). Settled %f meters away.", 
                                      to_target_dist_.load());
        }
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:legSetPoseTarget_(): "
                                  "current end effector (i, x, y, z) = (%zu, %lf, %lf, %lf)",
                                  legIdx, endEffector_x_[legIdx], endEffector_y_[legIdx], endEffector_z_[legIdx]);    
        is_busy_ = false;
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    moveit::core::RobotStatePtr sineWalkGait_(double gait_phase, double swing_fraction, 
                                              double lift, double x_stride, double y_stride, 
                                              double x_shift, double y_shift)
    // Note: full cycle makes 2 x stride
    {
        moveit::core::RobotStatePtr walk_state = std::make_shared<moveit::core::RobotState>(*last_walk_state_);
        for (std::size_t legIdx = 0; legIdx < legN_; legIdx++) {
            double target_x = home_x_[legIdx];
            double target_y = home_y_[legIdx];
            double target_z = home_z_[legIdx];

            if(is_imu_ == true) {
                target_z -= imu_z_corr_[legIdx];
            }

            bool is_group_a = ((legIdx == 0) || (legIdx == 3));
            bool is_group_b = ((legIdx == 1) || (legIdx == 2));
            bool is_group_backLeg = (legIdx == 2 || legIdx == 3);
            double local_phase = gait_phase;
            if (swing_fraction <= 0.25)
            {
                if (legIdx == 0) local_phase += 0.00;
                if (legIdx == 3) local_phase += 0.25;
                if (legIdx == 1) local_phase += 0.50;
                if (legIdx == 2) local_phase += 0.75;
            } else if (is_group_b) {
                local_phase += 0.5;
            }
            if (local_phase >= 1.0) {
                local_phase -= 1.0;
            }

            double x_offset = 0.0, y_offset = 0.0, z_offset = 0.0;
            if (local_phase < swing_fraction) {
                double swing_progress = local_phase/swing_fraction; 
                z_offset = lift*std::sin(swing_progress*M_PI);
                if (swing_fraction >= 0.5) {
                    x_offset = -x_stride + 2.0*x_stride*swing_progress;
                    y_offset = -y_stride + 2.0*y_stride*swing_progress;
                } else {
                    x_offset = -x_stride*std::cos(swing_progress*M_PI);
                    y_offset = -y_stride*std::cos(swing_progress*M_PI);
                }
            } else {
                double stance_progress = (local_phase - swing_fraction)/(1.0 - swing_fraction);
                z_offset = 0.0; 
                x_offset = x_stride - 2.0*x_stride*stance_progress;
                y_offset = y_stride - 2.0*y_stride*stance_progress;
            }
            if (is_group_backLeg == true) {
                target_x -= (x_offset + x_shift);
            } else {
                target_x += (x_offset + x_shift);
            }
            target_y += (y_offset + y_shift);
            target_z -= z_offset;

            geometry_msgs::msg::Pose leg_pose = endEffector_pose_[legIdx].pose;
            leg_pose.position.x = target_x;
            leg_pose.position.y = target_y;
            leg_pose.position.z = target_z;

            kinematics::KinematicsQueryOptions kinematics_options;
            kinematics_options.return_approximate_solution = true;
            auto leg_model_group = all_legs_robot_model_->getJointModelGroup(planning_group_[legIdx]);
            success_ = walk_state->setFromIK(leg_model_group, leg_pose, endEffector_link_[legIdx], 
                                             0.0, nullptr, kinematics_options);
            if (success_ == false) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, 
                                     "CubicDoggoLifecycleManager:sineWalkGait_(): "
                                     "IK failed for leg %zu. Skipping frame", legIdx);
            }
        }
        last_walk_state_ = walk_state;
        return walk_state;
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void handleGetState_(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        for (std::size_t legIdx = 0; legIdx < legN_; ++legIdx) {
            loadCurrentRobotState_(legIdx);
            response->message = "CubicDoggoLifecycleManager:handleGetState_(): current end effector (x, y, z) = ("
                               +std::to_string(endEffector_x_[legIdx]) + ", "
                               +std::to_string(endEffector_y_[legIdx]) + ", "
                               +std::to_string(endEffector_z_[legIdx]) + ")";
        }
        response->success = true;
    }
    void handleWalkRequest_(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                            std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
        is_walking_ = request->data;
        response->success = true;
        response->message = is_walking_ ? "walking started" : "walking stopped";
    }
    void handleIMUrequest_(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                           std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
        is_imu_ = request->data;
        response->success = true;
        response->message = is_imu_ ? "imu started" : "imu stopped";
    }
    void handleONNXrequest_(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                            std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
        if (request->data && (is_onnx_ == false)) {
            auto joint_state_obj = *joint_state_buf_.readFromRT();
            if (joint_state_obj.position.empty() == false) {
                if (policy_idx_resolved_ == false) {
                    policy_idx_resolved_ = resolvePolicyIndices_(joint_state_obj);
                }
                initial_ctrl_.assign(jointN_, 0.0);
                for (int jointIdx = 0; jointIdx < jointN_; jointIdx++) {
                    if (policy_idx_[jointIdx] >= 0) {
                        initial_ctrl_[jointIdx] = joint_state_obj.position[policy_idx_[jointIdx]];
                    }
                }
                last_action_.assign(jointN_, 0.0f);
                RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:handleONNXrequest_(): initial_ctrl_ captured");
            } else {
                response->success = false;
                response->message = "no joint_state";
                return;
            }
        }
        is_onnx_ = request->data;
        response->success = true;
        response->message = is_onnx_ ? "onnx started" : "onnx stopped";
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void controlLoop_() {
        rclcpp::get_logger("").set_level(rclcpp::Logger::Level::Warn);        // for silencing output

        // https://docs.ros.org/en/jazzy/p/control_toolbox/generated/structcontrol__toolbox_1_1AntiWindupStrategy.html
        control_toolbox::AntiWindupStrategy aw_strat;
        aw_strat.type = control_toolbox::AntiWindupStrategy::CONDITIONAL_INTEGRATION;
        aw_strat.i_max =  0.05;
        aw_strat.i_min = -0.05;
        //double kP = 0.0002, kI = 0.0008, kD = 0.00005;     // NOTE: PID 
        double kP = 0.0016, kI = 0.0001, kD = 0.0;     // NOTE: PID
        double corr_v_thres = 3.0, corr_d_thres = 1.0;              // threshold on pitch/roll and their vel 
        double corr_z_limit_kD = 0.01, corr_z_limit = 0.025;       // limit on +/- z
        //double pitch_shift = 1.0, roll_shift = -2.0;            // zero reference
        //double pitch_shift = 1.7, roll_shift = -2.7;              // basic standing reference
        double pitch_shift = 0.643, roll_shift = -2.917;              // basic on-support reference
        //double pitch_shift = 0.0, roll_shift = 0.0;

        auto loop_rate = rclcpp::WallRate(update_rate_);        // Hz, for consistent loop rate
        double maxVelScale = 1.0, maxAccScale = 1.0;
        int    waypoint_N_walk   = 40;                         // number of waypoints for each cycle
        double waypoint_dt_walk  = 0.5/double(update_rate_);    // second for each waypoint, to match loop rate
        int    waypoint_N_stand  = 1;                           // standing require immidiate reaction
        double waypoint_dt_stand = 1.0/double(update_rate_);    // standing require faster trajectory
        double IK_bufferTime     = 0.10;                        // time at end of cycle buffer for IK calc
        double swing_fraction    = 0.32;                        // creep < 0.25 < stable trot < 0.5 < trot
        double x_stride_max = 0.02, y_stride_max = 0.04;
        double lift0 = 0.04, x_shift0 = 0.0, y_shift0 = -0.007;
        double lift1 = 0.02, x_shift1 = 0.0, y_shift1 = -0.007;
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        pitch_pid_.set_gains(kP, kI, 0.0, aw_strat.i_max, aw_strat.i_min, aw_strat);     //P, I, D, upp, low
        roll_pid_ .set_gains(kP, kI, 0.0, aw_strat.i_max, aw_strat.i_min, aw_strat);
    
        all_legs_robot_model_ = all_legs_interface_->getRobotModel();
        auto joint_model_group = all_legs_robot_model_->getJointModelGroup(all_legs_planning_group_);
        std::vector<std::string> joint_names = joint_model_group->getActiveJointModelNames();

        double gait_phase = 0;
        double waypoint_dt = 1.0/double(update_rate_);
        double x_stride = 0.0, y_stride = 0.0;    
        double lift = lift0, x_shift = x_shift0, y_shift = y_shift0;
        rclcpp::Time previous_time = this->get_clock()->now();
        while (keep_running_thread_ && rclcpp::ok()) {
            rclcpp::Time current_time = this->get_clock()->now();
            double delta_t = (current_time - previous_time).seconds();            
            if (delta_t <= 0.0) {
                delta_t = waypoint_dt;
            }
            
            previous_time = current_time;
            RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:controlLoop_(): time = %lf\n"
                                      "is_walking_ = %d, control_initialized_ = %d, idle_name_ = %s", 
                                      current_time.seconds(), is_walking_.load(), control_initialized_.load(),
                                      idle_name_.c_str());
            RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:controlLoop_(): "
                                      "raw pitch = %lf, raw roll = %lf", 
                                      current_pitch_.load(std::memory_order_relaxed), 
                                      current_roll_ .load(std::memory_order_relaxed));

            if ((is_busy_ == true) || ((is_walking_ == false) && (idle_name_ != "stand"))) {
                waypoint_iter_ = 0;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            } else if (control_initialized_ == false) {
                if (idle_name_ != "stand") {
                    legNamedTarget_("stand");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                setDefaultVelAccScaler_(maxVelScale, maxAccScale);
                loadCurrentRobotState_();
                last_walk_state_ = std::make_shared<moveit::core::RobotState>(*all_legs_current_robot_state_);
                for (std::size_t legIdx = 0; legIdx < legN_; legIdx++) {
                    home_x_[legIdx] = endEffector_x_[legIdx];
                    home_y_[legIdx] = endEffector_y_[legIdx];
                    home_z_[legIdx] = endEffector_z_[legIdx];
                }
                pitch_pid_.reset();
                roll_pid_ .reset();
                control_initialized_ = true;    
                RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:controlLoop_(): home positions captured.");
            }
       
            moveit::core::RobotStatePtr target_state;
            bool onnx_now = is_onnx_.load();
            if (onnx_now == false) { 
                double current_pitch = current_pitch_.load(std::memory_order_relaxed) - pitch_shift; 
                double current_roll  = current_roll_ .load(std::memory_order_relaxed) - roll_shift; 
                double current_pitch_vel = current_pitch_vel_.load(std::memory_order_relaxed);
                double current_roll_vel  = current_roll_vel_ .load(std::memory_order_relaxed);
                current_pitch_vel *= std::clamp(std::fabs(current_pitch_vel)/corr_v_thres, 0.0, 1.0);
                current_roll_vel  *= std::clamp(std::fabs(current_roll_vel) /corr_v_thres, 0.0, 1.0);
                double pitch_corr = pitch_pid_.compute_command(current_pitch, delta_t);
                double roll_corr  = roll_pid_ .compute_command(current_roll,  delta_t);
                double pitch_corr_vel = std::clamp(kD*current_pitch_vel, -corr_z_limit_kD, corr_z_limit_kD);
                double roll_corr_vel  = std::clamp(kD*current_roll_vel,  -corr_z_limit_kD, corr_z_limit_kD); 
                pitch_corr += pitch_corr_vel;
                roll_corr  += roll_corr_vel;
                pitch_corr *= std::clamp(std::fabs(current_pitch)/corr_d_thres, 0.0, 1.0);
                roll_corr  *= std::clamp(std::fabs(current_roll) /corr_d_thres, 0.0, 1.0);
                pitch_corr  = std::clamp(pitch_corr, -corr_z_limit, corr_z_limit);
                roll_corr   = std::clamp(roll_corr,  -corr_z_limit, corr_z_limit);
                imu_z_corr_[0] = +pitch_corr + roll_corr;       // NOTE: FL, FR, BL, BR, watch out for direction!!!
                imu_z_corr_[1] = -pitch_corr + roll_corr;
                imu_z_corr_[2] = +pitch_corr - roll_corr;
                imu_z_corr_[3] = -pitch_corr - roll_corr;
                RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:controlLoop_():\n"
                                          "shifted pitch = %lf (vel = %lf), shifted roll = %lf (vel = %lf)\n"
                                          "full pitch corr = %lf (kD corr = %lf), "
                                          "full roll corr = %lf (kD corr = %lf)\n"
                                          "imu_z_corr = [%lf, %lf, %lf, %lf]",
                                          current_pitch, current_pitch_vel, current_roll, current_roll_vel,
                                          pitch_corr, pitch_corr_vel, roll_corr, roll_corr_vel,
                                          imu_z_corr_[0], imu_z_corr_[1], imu_z_corr_[2], imu_z_corr_[3]);
             
                if (is_walking_ == true) {
                    gait_phase = static_cast<double>(waypoint_iter_)/static_cast<double>(waypoint_N_walk);
                    waypoint_dt = waypoint_dt_walk;
                    x_stride = target_x_stride_*x_stride_max;
                    y_stride = target_y_stride_*y_stride_max;
                    if (is_imu_ == false) {
                        lift = lift0, x_shift = x_shift0, y_shift = y_shift0;
                    } else {
                        lift = lift1, x_shift = x_shift1, y_shift = y_shift1;
                    }
                    target_state = sineWalkGait_(gait_phase, swing_fraction, lift, x_stride,y_stride,x_shift,y_shift);
                    waypoint_iter_ = (waypoint_iter_ + 1) % waypoint_N_walk;
                } else {
                    waypoint_dt = waypoint_dt_stand; 
                    target_state = sineWalkGait_(0.0, swing_fraction, 0.0, 0.0, 0.0, 0.0, 0.0);
                    waypoint_iter_ = 0;
                }
            } else {
                if (is_walking_) {
                    // PASS, NOT YET IMPLEMENTED
                } else {            
                    std::vector<float> raw_obs;
                    if (buildObservation_(raw_obs) == false) {
                        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                                             "CubicDoggoLifecycleManager:controlLoop_(): loading raw_obs failed");
                    } else {
                        std::vector<float> norm_obs(raw_obs.size());
                        for (size_t obs_idx = 0; obs_idx < raw_obs.size(); obs_idx++) {
                            float obs_mean = (obs_idx < obs_mean_.size()) ? obs_mean_[obs_idx] : 0.0f;
                            float obs_var  = (obs_idx < obs_var_.size())  ? obs_var_[obs_idx]  : 1.0f; 
                            float normalized = (raw_obs[obs_idx] - obs_mean)/std::sqrt(obs_var + 1e-8f);
                            norm_obs[obs_idx] = std::clamp(normalized, -clip_obs_, clip_obs_);
                        }

                        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(norm_obs.size())};
                        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                            memory_info, norm_obs.data(), norm_obs.size(), input_shape.data(), input_shape.size());
                        auto output_tensors = ort_session_->Run(Ort::RunOptions{nullptr}, 
                            ort_input_names_.data(), &input_tensor, 1, ort_output_names_.data(), 1);
                        float* onnx_outputs = output_tensors[0].GetTensorMutableData<float>();
                        target_ctrl_ = processAction_(onnx_outputs);
                    }
                }
            }

            trajectory_msgs::msg::JointTrajectory traj_msg;
            traj_msg.joint_names = joint_names;
            traj_msg.header.stamp = rclcpp::Time(0, 0, this->get_clock()->get_clock_type());
            trajectory_msgs::msg::JointTrajectoryPoint joint_traj_pt;
            if (onnx_now == false) {
                target_state->copyJointGroupPositions(joint_model_group, joint_traj_pt.positions);
            } else {
                joint_traj_pt.positions.assign(joint_names.size(), 0.0);
                for (int jointIdx = 0; jointIdx < int(joint_names.size()); jointIdx++) {
                    auto jointID = std::find(policy_joint_names_.begin(), policy_joint_names_.end(), 
                                             joint_names[jointIdx]);
                    if (jointID != policy_joint_names_.end()) {
                        size_t ctrlIdx = std::distance(policy_joint_names_.begin(), jointID);
                        joint_traj_pt.positions[jointIdx] = target_ctrl_[ctrlIdx];
                    } else {
                        joint_traj_pt.positions[jointIdx] = all_legs_jointVals_[jointIdx];// hold pos if not in policy
                    }
                }
            }
            joint_traj_pt.time_from_start = rclcpp::Duration::from_seconds(1.0/double(update_rate_));
            traj_msg.points.push_back(joint_traj_pt);
            joint_publisher_->publish(traj_msg);
        }
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void setDefaultVelAccScaler_(double velScale, double accScale) {
        for (std::size_t legIdx = 0; legIdx < legN_; legIdx++) {
            leg_interface_[legIdx]->setMaxVelocityScalingFactor(velScale);
            leg_interface_[legIdx]->setMaxAccelerationScalingFactor(accScale);
        }
        all_legs_interface_->setMaxVelocityScalingFactor(velScale);
        all_legs_interface_->setMaxAccelerationScalingFactor(accScale);
    }
    void planAndExecute_(std::size_t legIdx) {
        planAndExecute_exe(legIdx);
    }
    void planAndExecute_() {
        planAndExecute_exe();
    }
    void planAndExecute_exe(std::size_t legIdx) {
        success_ = (leg_interface_[legIdx]->plan(move_plan_[legIdx]) == moveit::core::MoveItErrorCode::SUCCESS);
        if (success_) {
            leg_interface_[legIdx]->execute(move_plan_[legIdx]);
        } else {
            RCLCPP_WARN(get_logger(), "CubicDoggoLifecycleManager:planAndExecute_exe(%zu): planning failed", legIdx);
        }
    }
    void planAndExecute_exe() {
        success_ = ((all_legs_interface_->plan(all_legs_move_plan_) == moveit::core::MoveItErrorCode::SUCCESS));
        if (success_) {
            all_legs_interface_->execute(all_legs_move_plan_);
        } else {
            RCLCPP_WARN(get_logger(), "CubicDoggoLifecycleManager:planAndExecute_exe(): planning failed");
        }
    }
    void planAndExecute_pub(std::size_t legIdx) {
        if (leg_interface_[legIdx]->plan(move_plan_[legIdx]) == moveit::core::MoveItErrorCode::SUCCESS) {
            auto traj_msg = move_plan_[legIdx].trajectory.joint_trajectory;
            traj_msg.header.stamp = this->get_clock()->now();
            joint_publisher_->publish(traj_msg);
        } else {
            RCLCPP_WARN(get_logger(), "CubicDoggoLifecycleManager:planAndExecute_pub(%zu): planning failed", legIdx);
        }
    }
    void planAndExecute_pub() {
        if (all_legs_interface_->plan(all_legs_move_plan_) == moveit::core::MoveItErrorCode::SUCCESS) {
            auto traj_msg = all_legs_move_plan_.trajectory.joint_trajectory;
            traj_msg.header.stamp = this->get_clock()->now();
            joint_publisher_->publish(traj_msg);
        } else {
            RCLCPP_WARN(get_logger(), "CubicDoggoLifecycleManager:planAndExecute_pub(): planning failed");
        }
    }
    void loadCurrentRobotState_(std::size_t legIdx) {
        current_robot_state_[legIdx] = leg_interface_[legIdx]->getCurrentState(0.5);       // wait up to 0.5 seconds 
        if (!current_robot_state_[legIdx]) {
            RCLCPP_ERROR(get_logger(), "CubicDoggoLifecycleManager:loadCurrentRobotState_(%zu): "
                                       "timeout to fetch current robot state", legIdx);
            return;
        }
        endEffector_pose_[legIdx] = leg_interface_[legIdx]->getCurrentPose(endEffector_link_[legIdx]);
        endEffector_x_[legIdx] = endEffector_pose_[legIdx].pose.position.x;
        endEffector_y_[legIdx] = endEffector_pose_[legIdx].pose.position.y;
        endEffector_z_[legIdx] = endEffector_pose_[legIdx].pose.position.z;
        leg_interface_[legIdx]->setStartStateToCurrentState();
    }
    void loadCurrentRobotState_() {
        all_legs_current_robot_state_ = all_legs_interface_->getCurrentState(2.0);       // wait up to 2.0 seconds 
        if (!all_legs_current_robot_state_) {
            RCLCPP_ERROR(get_logger(), "CubicDoggoLifecycleManager:loadCurrentRobotState_(): "
                                       "timeout to fetch current robot state");
            return;
        }
        const moveit::core::JointModelGroup* joint_model_group = 
            all_legs_current_robot_state_->getJointModelGroup(all_legs_planning_group_);
        all_legs_current_robot_state_->copyJointGroupPositions(joint_model_group, all_legs_jointVals_);
        all_legs_interface_->setStartStateToCurrentState();
        for (std::size_t legIdx = 0; legIdx < legN_; legIdx++) {
            endEffector_pose_[legIdx] = leg_interface_[legIdx]->getCurrentPose(endEffector_link_[legIdx]);
            endEffector_x_[legIdx] = endEffector_pose_[legIdx].pose.position.x;
            endEffector_y_[legIdx] = endEffector_pose_[legIdx].pose.position.y;
            endEffector_z_[legIdx] = endEffector_pose_[legIdx].pose.position.z;
            leg_interface_[legIdx]->setStartStateToCurrentState();
        }
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    bool initializeOnnxModel_() {
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:initializeOnnxModel_(): "
                                  "loading vector normalization file: %s", VEC_NORM_PATH.c_str());
        std::ifstream ifstream_obj(VEC_NORM_PATH);
        if (!ifstream_obj.is_open()) {
            RCLCPP_ERROR(get_logger(), "CubicDoggoLifecycleManager:initializeOnnxModel_(): "
                                       "failed to load VEC_NORM_PATH");
            return false;
        }
        nlohmann::json json_data;
        ifstream_obj >> json_data;
        if (json_data.contains("obs_rms")) {
            obs_mean_ = json_data["obs_rms"]["mean"].get<std::vector<float>>();
            obs_var_  = json_data["obs_rms"]["var"].get<std::vector<float>>();
        } else {
            obs_mean_ = json_data["mean"].get<std::vector<float>>();
            obs_var_  = json_data["var"].get<std::vector<float>>();
        }
        if (json_data.contains("clip_obs")) {
            clip_obs_ = json_data["clip_obs"].get<float>();
        }

        ///////////////
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:initializeOnnxModel_(): "
                                  "loading ONNX model file: %s", ONNX_MODEL_PATH.c_str());
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CubicDoggoPPO");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        ort_session_ = std::make_unique<Ort::Session>(*ort_env_, ONNX_MODEL_PATH.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        ort_input_names_str_ .push_back(ort_session_->GetInputNameAllocated (0, allocator).get());
        ort_output_names_str_.push_back(ort_session_->GetOutputNameAllocated(0, allocator).get());
        ort_input_names_ .push_back(ort_input_names_str_ [0].c_str());
        ort_output_names_.push_back(ort_output_names_str_[0].c_str());  

        auto input_shape = ort_session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        onnx_input_dim_ = input_shape.back();
        RCLCPP_INFO(get_logger(), "CubicDoggoLifecycleManager:initializeOnnxModel_(): "
                                  "onnx input dim = %ld", long(onnx_input_dim_));

        ///////////////
        auto ort_meta_data = ort_session_->GetModelMetadata();
        auto ort_meta_map = [&](const char* key, const char* def) -> std::string {
            try { return ort_meta_data.LookupCustomMetadataMapAllocated(key, allocator).get(); }
            catch (...) { return std::string(def); }
        };
        auto parse_range = [](const std::string& str_val, double& str_low, double& str_high) {
            std::string str_cleaned;
            for (char char_val : str_val) if (!std::strchr("()[] ", char_val)) str_cleaned += char_val;
            auto str_comma = str_cleaned.find(',');
            str_low  = std::stod(str_cleaned.substr(0, str_comma));
            str_high = std::stod(str_cleaned.substr(str_comma + 1));
        };
        int joint_number = std::stoi(ort_meta_map("joint_number", std::to_string(jointN_)));
        if (joint_number != int(jointN_)) {
            RCLCPP_ERROR(get_logger(), "CubicDoggoLifecycleManager:initializeOnnxModel_(): "
                                       "ONNX joint_number mismatch");
            return false;
        }
        action_scale_        = std::stod(ort_meta_map("action_scale",        std::to_string(action_scale_)));
        action_filter_alpha_ = std::stod(ort_meta_map("action_filter_alpha", std::to_string(action_filter_alpha_)));
        parse_range(ort_meta_map("action_range",       "(-1.0, 1.0)"), action_range_[0],      action_range_[1]);
        parse_range(ort_meta_map("action_delta_range", "(-0.1, 0.1)"), action_delta_range_[0],action_delta_range_[1]);
        initial_ctrl_.assign(jointN_, 0.0);
        target_ctrl_ .assign(jointN_, 0.0);
        last_action_ .assign(jointN_, 0.0);
        return true;
    }
    bool resolvePolicyIndices_(const sensor_msgs::msg::JointState& joint_state_obj) {
        bool policy_idx_resolved = true; 
        for (int jointIdx = 0; jointIdx < jointN_; jointIdx++) {
            auto jointID = std::find(joint_state_obj.name.begin(), joint_state_obj.name.end(), 
                                     policy_joint_names_[jointIdx]);
            policy_idx_[jointIdx] = (jointID == joint_state_obj.name.end()) ? -1 : 
                                    int(std::distance(joint_state_obj.name.begin(), jointID));
            if (policy_idx_[jointIdx] < 0) {
                RCLCPP_ERROR(get_logger(), "resolvePolicyIndices_(): "
                                           "missing joint %s", policy_joint_names_[jointIdx].c_str());
                policy_idx_resolved = false;
            }
        }
        return policy_idx_resolved;
    }
    bool buildObservation_(std::vector<float>& raw_obs) {
        auto joint_state_obj = *joint_state_buf_.readFromRT();
        if (joint_state_obj.position.empty() == true) return false;
        if (policy_idx_resolved_ == false) {
            policy_idx_resolved_ = resolvePolicyIndices_(joint_state_obj);
            if (policy_idx_resolved_ == false) return false;
        }
        for (int jointIdx = 0; jointIdx < jointN_; jointIdx++) { 
            if (policy_idx_[jointIdx] < 0) return false;
        }

        raw_obs.clear();
        raw_obs.reserve(2*jointN_);
        for (int jointIdx = 0; jointIdx < jointN_; jointIdx++) {
            raw_obs.push_back( float(joint_state_obj.position[policy_idx_[jointIdx]]) );
        }
        if ((joint_state_obj.velocity.empty() == false) && 
            (joint_state_obj.velocity.size() == joint_state_obj.position.size())) {
            for (int jointIdx = 0; jointIdx < jointN_; jointIdx++) {
                raw_obs.push_back( float(joint_state_obj.velocity[policy_idx_[jointIdx]]) );
            }
        } else {
            auto now = this->get_clock()->now();
            double delta_t = (now - last_js_time_).seconds();
            if ( delta_t <= 0.0) delta_t = 1.0/double(update_rate_);
            for (int jointIdx = 0; jointIdx < jointN_; jointIdx++) {
                double pos_val = joint_state_obj.position[policy_idx_[jointIdx]];
                raw_obs.push_back( float((pos_val - last_js_pos_[jointIdx])/delta_t) );
                last_js_pos_[jointIdx] = pos_val;
            }
            last_js_time_ = now;
        }
        raw_obs.push_back( float(raw_gyro_x_.load(std::memory_order_relaxed)) );
        raw_obs.push_back( float(raw_gyro_y_.load(std::memory_order_relaxed)) );
        raw_obs.push_back( float(raw_gyro_z_.load(std::memory_order_relaxed)) );
        raw_obs.push_back( float(raw_roll_rad_ .load(std::memory_order_relaxed)) );
        raw_obs.push_back( float(raw_pitch_rad_.load(std::memory_order_relaxed)) );

        return (int64_t(raw_obs.size()) == onnx_input_dim_);
    }
    std::vector<double> processAction_(const float* onnx_outputs) {
        std::vector<float> action(jointN_);
        for (int jointIdx = 0; jointIdx < jointN_; jointIdx++) {
            action[jointIdx] = std::clamp(onnx_outputs[jointIdx], float(action_range_[0]), float(action_range_[1]));
        }
        for (int jointIdx = 0; jointIdx < jointN_; jointIdx++) {
            float action_delta = std::clamp(action[jointIdx] - last_action_[jointIdx],
                                            float(action_delta_range_[0]), float(action_delta_range_[1]));
            action[jointIdx] = last_action_[jointIdx] + action_delta;
        }
        std::vector<double> target_ctrl(jointN_, 0.0);
        for (int jointIdx = 0; jointIdx < jointN_; jointIdx++) {
            float filtered_action = action_filter_alpha_*action[jointIdx]
                                  + (1.0 - action_filter_alpha_)*last_action_[jointIdx];
            target_ctrl[jointIdx] = initial_ctrl_[jointIdx] + double(filtered_action*action_scale_);
        }
        last_action_ = action;
        return target_ctrl;
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    rclcpp::Node::SharedPtr moveit_node_;
    std::atomic<bool> success_{false};
    std::string current_lifecycle_state_ = "state_uninitialized";

    static constexpr std::size_t legN_         = 4;
    static constexpr std::size_t jointNperLeg_ = 3;
    static constexpr std::size_t jointN_       = legN_*jointNperLeg_;
    std::array<std::string, jointN_> policy_joint_names_ {
            "servo1_servo1_padding_FL", "servo2_servo2_padding_FL", "servo3_calfFeet_FL",
            "servo1_servo1_padding_FR", "servo2_servo2_padding_FR", "servo3_calfFeet_FR",
            "servo1_servo1_padding_BL", "servo2_servo2_padding_BL", "servo3_calfFeet_BL",
            "servo1_servo1_padding_BR", "servo2_servo2_padding_BR", "servo3_calfFeet_BR"};
    std::array<std::string, legN_> planning_group_  {"leg_FL",       "leg_FR",       "leg_BL",       "leg_BR"};
    std::array<std::string, legN_> endEffector_link_{"calfSphere_FL","calfSphere_FR","calfSphere_BL","calfSphere_BR"};
    std::array<std::shared_ptr<MoveGroupInterface>, legN_> leg_interface_;
    std::array<moveit::core::RobotStatePtr,         legN_> current_robot_state_;
    std::array<geometry_msgs::msg::PoseStamped,     legN_> endEffector_pose_;
    std::array<MoveGroupInterface::Plan,            legN_> move_plan_;
    std::array<double, legN_> endEffector_x_{};
    std::array<double, legN_> endEffector_y_{};
    std::array<double, legN_> endEffector_z_{};
    std::string                           all_legs_planning_group_ = "all_legs";
    std::vector<std::string>              all_legs_endEffector_links_;
    std::shared_ptr<MoveGroupInterface>   all_legs_interface_;
    moveit::core::RobotStatePtr           all_legs_current_robot_state_;
    std::vector<geometry_msgs::msg::Pose> all_legs_endEffector_pose_;
    MoveGroupInterface::Plan              all_legs_move_plan_;
    std::vector<double>                   all_legs_jointVals_;

    rclcpp::CallbackGroup::SharedPtr callback_group_;
    rclcpp::Subscription<ros_string>       ::SharedPtr leg_named_subscriber_;
    rclcpp::Subscription<ros_array>        ::SharedPtr leg_joint_subscriber_;
    rclcpp::Subscription<custom_pose_array>::SharedPtr leg_pose_subscriber_;
    rclcpp::Subscription<custom_feet_array>::SharedPtr leg_feet_subscriber_;
    std::atomic<double> to_target_dist_{0.0};
    std::atomic<double> to_target_dist_thres_{0.01};

    int update_rate_   = 100;  
    int waypoint_iter_ = 0;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr state_service_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr walk_service_; 
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr imu_service_; 
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr onnx_service_; 
    moveit::core::RobotStatePtr last_walk_state_;
    std::string       idle_name_ = "rest";
    std::atomic<bool> is_busy_   {false};
    std::atomic<bool> is_walking_{false};
    std::atomic<bool> is_imu_    {false};
    std::atomic<bool> is_onnx_   {false};
    std::atomic<bool> control_initialized_{false};
    std::atomic<bool> keep_running_thread_{false};
    std::thread walking_thread_;
    std::array<double, legN_> home_x_, home_y_, home_z_;
    std::atomic<double> target_x_stride_{0.0};
    std::atomic<double> target_y_stride_{0.0};
    moveit::core::RobotModelConstPtr                                    all_legs_robot_model_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_publisher_;
    rclcpp_action::Client<ExecuteTrajectory>::SharedPtr                 exec_action_client_;

    control_toolbox::Pid pitch_pid_, roll_pid_;
    std::atomic<double> raw_gyro_x_{0.0};
    std::atomic<double> raw_gyro_y_{0.0};
    std::atomic<double> raw_gyro_z_{0.0};
    std::atomic<double> raw_pitch_rad_{0.0};
    std::atomic<double> raw_roll_rad_ {0.0};
    std::atomic<double> raw_yaw_rad_  {0.0};
    std::atomic<double> current_pitch_{0.0};
    std::atomic<double> current_roll_ {0.0};
    std::atomic<double> current_pitch_vel_{0.0};
    std::atomic<double> current_roll_vel_ {0.0};
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
    std::array<double, legN_> imu_z_corr_{};

    std::unique_ptr<Ort::Env>     ort_env_;
    std::unique_ptr<Ort::Session> ort_session_;
    std::vector<const char*>      ort_input_names_;
    std::vector<const char*>      ort_output_names_;
    std::vector<std::string>      ort_input_names_str_;
    std::vector<std::string>      ort_output_names_str_;
    int64_t                       onnx_input_dim_ = 0;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber_;
    bool                                                          policy_idx_resolved_ = false; 
    std::array<int, jointN_>                                      policy_idx_{};
    realtime_tools::RealtimeBuffer<sensor_msgs::msg::JointState>  joint_state_buf_;
    std::vector<double>                                           initial_ctrl_;
    std::vector<double>                                           target_ctrl_;
    std::array<double, jointN_>                                   last_js_pos_{};
    rclcpp::Time                                                  last_js_time_{0, 0, RCL_ROS_TIME};

    std::vector<float> obs_mean_;
    std::vector<float> obs_var_;
    float              clip_obs_ = 10.0f;
    double              action_scale_        = 1.0;
    std::vector<double> action_range_        = {-1.0, 1.0};
    std::vector<double> action_delta_range_  = {-0.1, 0.1};
    double              action_filter_alpha_ = 0.5;
    std::vector<float> last_action_;
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);

    auto lifecycle_manager_node = std::make_shared<CubicDoggoLifecycleManager>(node_options);
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(lifecycle_manager_node->get_node_base_interface());
    executor.add_node(lifecycle_manager_node->get_moveit_node());
    
    std::thread executor_thread([&executor]() { executor.spin(); });
    using lifecycle_msgs::msg::Transition;
    using lifecycle_msgs::msg::State;
    lifecycle_manager_node->trigger_transition(rclcpp_lifecycle::Transition(Transition::TRANSITION_CONFIGURE));
    while (rclcpp::ok() && lifecycle_manager_node->get_current_state().id() != State::PRIMARY_STATE_ACTIVE) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (lifecycle_manager_node->get_current_state().id() == State::PRIMARY_STATE_INACTIVE) {
            lifecycle_manager_node->trigger_transition(
                rclcpp_lifecycle::Transition(Transition::TRANSITION_ACTIVATE));
        }
    }
    while (rclcpp::ok()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    executor.cancel();
    if (executor_thread.joinable()) {
        executor_thread.join();
    }
    rclcpp::shutdown();
    return 0;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

