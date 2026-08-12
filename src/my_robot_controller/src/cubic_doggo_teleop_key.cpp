#include <iostream>
#include <cstring>
#include <termios.h>
#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "example_interfaces/msg/string.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "my_robot_interface/msg/cubic_doggo_leg_feet_target.hpp"
using custom_feet_array = my_robot_interface::msg::CubicDoggoLegFeetTarget;

#define KEYCODE_R 0x43
#define KEYCODE_L 0x44
#define KEYCODE_U 0x41
#define KEYCODE_D 0x42
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
char getKey() {
    char buf = 0;
    struct termios old;
    std::memset(&old, 0, sizeof(struct termios));

    if (tcgetattr(0, &old) < 0) perror("tcsetattr()");
    old.c_lflag &= ~ICANON;
    old.c_lflag &= ~ECHO;
    old.c_cc[VMIN] = 1;
    old.c_cc[VTIME] = 0;
    if (tcsetattr(0, TCSANOW, &old) < 0) perror("tcsetattr ICANON");
    if (read(0, &buf, 1) < 0) perror ("read()");
    old.c_lflag |= ICANON;
    old.c_lflag |= ECHO;
    if (tcsetattr(0, TCSADRAIN, &old) < 0) perror ("tcsetattr ~ICANON");
    return (buf);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class CubicDoggoTeleopKey : public rclcpp::Node {
public:
    CubicDoggoTeleopKey() : Node("cubic_doggo_teleop_key") {
        command_pub_ = this->create_publisher<example_interfaces::msg::String>("/leg_set_named", 10);
        walk_client_ = this->create_client<std_srvs::srv::SetBool>            ("/leg_walk_toggle");
        imu_client_  = this->create_client<std_srvs::srv::SetBool>            ("/leg_imu_toggle");
        feet_pub_    = this->create_publisher<custom_feet_array>              ("/leg_set_feet", 10);
        RCLCPP_INFO(this->get_logger(), "CubicDoggoTeleopKey:constructor()"
                                        "controller node started, listening on /joy...");      
 
        std::cout << "Reading from keyboard\n";
        std::cout << "---------------------------\n";
        std::cout << "Press 'w' to stand\n";
        std::cout << "Press 'a' to bow\n";
        std::cout << "Press 'd' to sit\n";
        std::cout << "Press 's' to rest\n";
        std::cout << "Press 'z' to walk, and arrow keys for the directions\n";
        std::cout << "Press 'x' to start imu\n";
        std::cout << "Press 'q' to quit\n";
    }

    void run() {
        auto feet_msg = custom_feet_array();
        while (rclcpp::ok()) {
            char keyPressed = getKey();
            if (keyPressed == '\033') { 
                getKey();
                keyPressed = getKey();
            }
            auto msg = example_interfaces::msg::String();

            if (keyPressed == 'w') {
                send_pose_("stand");
            } else if (keyPressed == 'a') {
                send_pose_("bow");
            } else if (keyPressed == 'd') {
                send_pose_("sit");
            } else if (keyPressed == 's') {
                send_pose_("rest");
            } else if (keyPressed == 'z') {
                call_walk_(!is_walking_);
                is_walking_ = !is_walking_;
            } else if (keyPressed == KEYCODE_U) {
                current_x_ = 0.0;
                current_y_ = 1.0;
            } else if (keyPressed == KEYCODE_D) {
                current_x_ = 0.0;
                current_y_ = -1.0;
            } else if (keyPressed == KEYCODE_L) {
                current_x_ = 1.0;
                current_y_ = 0.0;
            } else if (keyPressed == KEYCODE_R) { 
                current_x_ = -1.0;
                current_y_ = 0.0;
            } else if (keyPressed == 'x') {
                call_imu_(!is_imu_);
                is_imu_ = !is_imu_;
            } else if (keyPressed == 'q') {
                RCLCPP_INFO(this->get_logger(), "exiting");
                break;
            } else {
                current_x_ = 0.0;
                current_y_ = 0.0;
            }

            if (is_walking_ == true) {
                feet_msg.x = current_x_;
                feet_msg.y = current_y_;
                feet_pub_->publish(feet_msg);
            }
        }
    }

private:
    void send_pose_(std::string pose_name) {
        example_interfaces::msg::String out_msg;
        out_msg.data = pose_name;
        command_pub_->publish(out_msg);
        RCLCPP_INFO(this->get_logger(), "CubicDoggoTeleopKey:send_pose_(): '%s' command sent", pose_name.c_str());
    }
    void call_walk_(bool walk_state) {
        if (!walk_client_->wait_for_service(std::chrono::milliseconds(500))) {
            RCLCPP_WARN(this->get_logger(), "CubicDoggoTeleopKey:call_walk_(): "
                                            "service /leg_walk_toggle not available!");
            return;
        }
        auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
        request->data = walk_state;
        auto result = walk_client_->async_send_request(request);
        std::string walk_state_str = walk_state ? "true" : "false";
        RCLCPP_INFO(this->get_logger(), "CubicDoggoTeleopKey:call_walk_(): walk state '%s' sent", 
                                        walk_state_str.c_str());
    }
    void call_imu_(bool imu_state) {
        if (!imu_client_->wait_for_service(std::chrono::milliseconds(500))) {
            RCLCPP_WARN(this->get_logger(), "CubicDoggoTeleopKey:call_walk_(): "
                                            "service /leg_walk_toggle not available!");
            return;
        }
        auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
        request->data = imu_state;
        auto result = imu_client_->async_send_request(request);
        std::string imu_state_str = imu_state ? "true" : "false";
        RCLCPP_INFO(this->get_logger(), "CubicDoggoTeleopKey:call_imu_(): imu state '%s' sent",
                                        imu_state_str.c_str());
    }

    rclcpp::Publisher<example_interfaces::msg::String>::SharedPtr command_pub_;
    rclcpp::Publisher<custom_feet_array>::SharedPtr               feet_pub_;
    rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr             walk_client_;
    rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr             imu_client_;
    bool   is_walking_ = false;
    double current_x_  = 0.0;
    double current_y_  = 0.0;
    bool   is_imu_     = false;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CubicDoggoTeleopKey>();
    node->run();
    rclcpp::shutdown();
    return 0;
}
