from ament_index_python.packages import get_package_share_directory
import os, re, time, math
import tempfile
import numpy as np

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
from sensor_msgs.msg import Imu, JointState

import xacro
import mujoco, mujoco.viewer
MUJOCO_ROOT_DOF = 7
#############################################################################################################################
class CubicDoggoMuJoCoBridge(Node):
    def __init__(self, model, data):
        super().__init__('mujoco_ros_bridge')
        self.model = model
        self.data  = data

        self.sub = self.create_subscription(JointTrajectory, '/all_legs_controller/joint_trajectory', self.joint_callback,10)
        self.imu_pub = self.create_publisher(Imu,        '/imu_broadcaster/imu', 10)
        self.js_pub  = self.create_publisher(JointState, '/joint_states',        10)

        self.ctrl_joint_names = []
        self.ctrl_joint_map   = {}
        self.ctrl_qpos_addrs  = []
        for joint_idx in range(self.model.nu):
            joint_id = self.model.actuator_trnid[joint_idx, 0]
            join_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_idx)
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.ctrl_joint_names.append(join_name)
            self.ctrl_joint_map[join_name] = joint_idx
            self.ctrl_qpos_addrs.append(qpos_addr)
            self.data.ctrl[joint_idx] = self.data.qpos[qpos_addr]

    def joint_callback(self, msg):
        target_point = msg.points[-1] 
        for joint_idx, name in enumerate(msg.joint_names):
            if name in self.ctrl_joint_names:
                ctrl_joint_id = self.ctrl_joint_map[name]
                self.data.ctrl[ctrl_joint_id] = target_point.positions[joint_idx]

    def publish_data(self):
        time_now = self.get_clock().now().to_msg()

        joint_state_msg = JointState()
        joint_state_msg.header.stamp = time_now
        joint_state_msg.name = self.ctrl_joint_names
        joint_state_msg.position = [float(self.data.qpos[qpos_addr]) for qpos_addr in self.ctrl_qpos_addrs] 
        self.js_pub.publish(joint_state_msg)

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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[0] = 0
        while rclpy.ok() and viewer.is_running():
            step_start = time.time()
            
            mujoco.mj_step(model, data)
            rclpy.spin_once(bridge, timeout_sec=0)
            bridge.publish_data()
            
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
#############################################################################################################################
if __name__ == '__main__': main()







