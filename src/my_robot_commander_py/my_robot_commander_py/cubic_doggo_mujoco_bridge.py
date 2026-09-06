from ament_index_python.packages import get_package_share_directory
import os, re, time, copy, math
import tempfile
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import Imu, JointState
from control_msgs.action import FollowJointTrajectory

import xacro
import mujoco, mujoco.viewer
MUJOCO_ROOT_DOF = 7
#############################################################################################################################
class CubicDoggoMuJoCoBridge(Node):
    def __init__(self, model, data):
        super().__init__('mujoco_ros_bridge')
        self.model = model
        self.data  = data

        self.action_server = ActionServer(self, FollowJointTrajectory, '/all_legs_controller/follow_joint_trajectory',
                                          self.execute_callback)
        self.sub = self.create_subscription(JointTrajectory, '/all_legs_controller/joint_trajectory', self.joint_callback,10)
        self.imu_pub = self.create_publisher(Imu, '/imu_broadcaster/imu', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)

        self.ctrl_joint_names = []
        self.ctrl_joint_map   = {}
        self.ctrl_qpos_addrs  = []
        self.ctrl_qvel_addrs  = []
        for joint_idx in range(self.model.nu):
            joint_id   = self.model.actuator_trnid[joint_idx, 0]
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_idx)
            qpos_addr  = self.model.jnt_qposadr[joint_id]
            qvel_addr  = self.model.jnt_dofadr[joint_id] 
            self.ctrl_joint_names.append(joint_name)
            self.ctrl_joint_map[joint_name] = joint_idx
            self.ctrl_qpos_addrs.append(qpos_addr)
            self.ctrl_qvel_addrs.append(qvel_addr)
            self.data.ctrl[joint_idx] = self.data.qpos[qpos_addr]

    def apply_joint_positions(self, ros_joint_names, positions):
        for joint_idx, name in enumerate(ros_joint_names):
            if name in self.ctrl_joint_map:
                ctrl_joint_id = self.ctrl_joint_map[name]
                self.data.ctrl[ctrl_joint_id] = positions[joint_idx]

    async def execute_callback(self, goal_handle):
        self.get_logger().info('CubicDoggoMuJoCoBridge:execute_callback(): action received, executing')
        trajectory = goal_handle.request.trajectory
        
        target_point = trajectory.points[-1]
        self.apply_joint_positions(trajectory.joint_names, target_point.positions)
        
        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        return result

    def joint_callback(self, msg):
        #self.get_logger().info('CubicDoggoMuJoCoBridge:joint_callback(): action received, executing')
        self.apply_joint_positions(msg.joint_names, msg.points[-1].positions)    

    def publish_data(self):
        time_now = self.get_clock().now().to_msg()

        joint_state_msg = JointState()
        joint_state_msg.header.stamp = time_now
        joint_state_msg.name = self.ctrl_joint_names
        joint_state_msg.position = [float(self.data.qpos[qpos_addr]) for qpos_addr in self.ctrl_qpos_addrs] 
        joint_state_msg.velocity = [float(self.data.qvel[qvel_addr]) for qvel_addr in self.ctrl_qvel_addrs]
        joint_state_msg.effort   = [float(self.data.actuator_force[joint_idx]) for joint_idx in range(self.model.nu)]
        self.joint_state_pub.publish(joint_state_msg)

        msg = Imu()
        msg.header.stamp = time_now
        msg.header.frame_id = "imu_link"
        accel_data = self.data.sensor('accel').data
        msg.linear_acceleration.x = accel_data[0]
        msg.linear_acceleration.y = accel_data[1]
        msg.linear_acceleration.z = accel_data[2]
        gyro_data = self.data.sensor('gyro').data
        msg.angular_velocity.x = gyro_data[0]
        msg.angular_velocity.y = gyro_data[1]
        msg.angular_velocity.z = gyro_data[2]
        quat_data = self.data.sensor('quat').data
        msg.orientation.x = quat_data[1]
        msg.orientation.y = quat_data[2]
        msg.orientation.z = quat_data[3]
        msg.orientation.w = quat_data[0]

        self.imu_pub.publish(msg)
#############################################################################################################################
def main():
    rclpy.init()
    
    pkg_share_path = get_package_share_directory('my_robot_description')
    xacro_path     = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.urdf.xacro')
    mjcf_path      = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.mujoco.xml')
    usdf_file      =                                      'cubic_doggo.mujoco.urdf'

    xacro_raw = xacro.process_file(xacro_path)
    urdf_raw  = xacro_raw.toxml()
    urdf_content = urdf_raw.replace('package://my_robot_description', pkg_share_path)
    urdf_content = urdf_content.replace('</robot>', '<mujoco><compiler discardvisual="false"/></mujoco></robot>')

    urdf_content_temp = mujoco.MjModel.from_xml_string(urdf_content)
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml') as tempfileObj:
        mujoco.mj_saveLastXML(tempfileObj.name, urdf_content_temp)
        tempfileObj.seek(0)
        urdf_content = tempfileObj.read()
    robot_assets = re.search(r'<asset>(.*?)</asset>',         urdf_content, re.DOTALL).group(1)
    robot_bodies = re.search(r'<worldbody>(.*?)</worldbody>', urdf_content, re.DOTALL).group(1)

    with open(mjcf_path, 'r') as fileObj:
        mjcf_content = fileObj.read()
    mjcf_content = mjcf_content.replace('$MY_ROBOT_DESCRIPTION_PATH', pkg_share_path)
    mjcf_content = mjcf_content.replace('</asset>', robot_assets + '\n    </asset>')
    mjcf_content = mjcf_content.replace('<include file=\"'+usdf_file+'\"/>', robot_bodies)
    mjcf_content = mjcf_content.replace('name="calfSphere_FL"', 'name="calfSphere_FL" class="foot_friction"')
    mjcf_content = mjcf_content.replace('name="calfSphere_FR"', 'name="calfSphere_FR" class="foot_friction"')
    mjcf_content = mjcf_content.replace('name="calfSphere_BL"', 'name="calfSphere_BL" class="foot_friction"')
    mjcf_content = mjcf_content.replace('name="calfSphere_BR"', 'name="calfSphere_BR" class="foot_friction"')

    model  = mujoco.MjModel.from_xml_string(mjcf_content)
    data   = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)     #prevent moveit from setting every joint 0
    bridge = CubicDoggoMuJoCoBridge(model, data)

    wall_time_start = time.time()
    sim_time_start  = data.time
    last_log_time   = copy.copy(wall_time_start)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[0] = 0
        while rclpy.ok() and viewer.is_running():
            wall_time_elapsed = time.time() - wall_time_start
            while (data.time - sim_time_start) < wall_time_elapsed:
                mujoco.mj_step(model, data)
            rclpy.spin_once(bridge, timeout_sec=0)
            bridge.publish_data()
            if (time.time() - last_log_time) >= 1.0:
                bridge.get_logger().info(f"CubicDoggoMuJoCoBridge:main(): "
                                         f"MuJoCo Time: {data.time:.2f}s, Real Time: {time.time() - wall_time_start:.2f}s")
                last_log_time = time.time()
            viewer.sync()
            time.sleep(0.001)
#############################################################################################################################
if __name__ == '__main__': main()







