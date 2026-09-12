from ament_index_python.packages import get_package_share_directory
import os, re, time, math
import tempfile
import numpy as np

import xacro
import mujoco, mujoco.viewer
import pinocchio

from ._GlobalFuncs import *
#############################################################################################################################
def solve_leg_ik(model, data, frame_ids, target_positions, joint_angle_init, max_iter=100, eps=1e-4):
    joint_angle    = joint_angle_init.copy()
    delta_time     = 0.1
    damping_factor = 1e-6

    for _ in range(max_iter):
        pinocchio.forwardKinematics(model, data, joint_angle)
        pinocchio.updateFramePlacements(model, data)
        
        delta_positions, trans_Jacobs = [], []
        for frame_id, target_position in zip(frame_ids, target_positions):
            curr_position = data.oMf[frame_id].translation
            delta_positions.append(curr_position - target_position)
            
            trans_Jacob = pinocchio.computeFrameJacobian(model, data, joint_angle, frame_id, 
                                                         pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            trans_Jacobs.append(trans_Jacob)

        delta_position_full = np.concatenate(delta_positions)
        if np.linalg.norm(delta_position_full) < eps:
            break

        trans_Jacob_all = np.vstack(trans_Jacobs)
        joint_velocity = -trans_Jacob_all.T @ np.linalg.inv(
            trans_Jacob_all @ trans_Jacob_all.T + damping_factor*np.eye(trans_Jacob_all.shape[0])) @ delta_position_full
        joint_angle = pinocchio.integrate(model, joint_angle, joint_velocity*delta_time)
    return joint_angle
#############################################################################################################################
def main():
    leg_prefixes = ['FL', 'FR', 'BL', 'BR']
    joint_names = []
    for leg_prefix in leg_prefixes:
        joint_names.append('servo1_servo1_padding_'+leg_prefix)
        joint_names.append('servo2_servo2_padding_'+leg_prefix)
        joint_names.append('servo3_calfFeet_'      +leg_prefix)
    target_feet_standing = [
        np.array([ 0.09,  0.07, -0.12]), # FL
        np.array([ 0.09, -0.07, -0.12]), # FR
        np.array([-0.09,  0.07, -0.12]), # BL
        np.array([-0.09, -0.07, -0.12])  # BR
    ]

    pkg_share_path = get_package_share_directory('my_robot_description')
    xacro_path     = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.urdf.xacro')
    mjcf_path      = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.mujoco.xml')
    usdf_file      =                                      'cubic_doggo.mujoco.urdf'

    xacro_raw = xacro.process_file(xacro_path)
    urdf_raw  = xacro_raw.toxml()
    urdf_content = urdf_raw.replace('package://my_robot_description', pkg_share_path)
    urdf_content = urdf_content.replace('</robot>', '<mujoco><compiler discardvisual="false"/></mujoco></robot>')

    urdf_pinocchio = re.sub(r'<joint name="world_base_link".*?</joint>', '', urdf_content, flags=re.DOTALL)
    urdf_pinocchio = urdf_pinocchio.replace('<link name="world"/>', '')
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.urdf') as tempfileObj:
        tempfileObj.write(urdf_pinocchio)
        tempfileObj.flush()
        pinocchio_model = pinocchio.buildModelFromUrdf(tempfileObj.name, pinocchio.JointModelFreeFlyer())
        pinocchio_data  = pinocchio_model.createData()
    foot_frame_ids = [pinocchio_model.getFrameId('calfSphere_'+leg_prefix) for leg_prefix in leg_prefixes]
    joint_angle_neutral = pinocchio.neutral(pinocchio_model)
    joint_angle_target  = solve_leg_ik(pinocchio_model, pinocchio_data, foot_frame_ids, target_feet_standing, 
                                       joint_angle_neutral)

    mujoco_joint_targets = []
    for joint_name in joint_names:
        joint_id = pinocchio_model.getJointId(joint_name)
        if joint_id >= len(pinocchio_model.joints):
            raise KeyError('cubic_doggo_mujoco_stand(): joint '+joint_name+' not found in Pinocchio model')
        joint_angle_idx = pinocchio_model.joints[joint_id].idx_q
        mujoco_joint_targets.append(joint_angle_target[joint_angle_idx])
    mujoco_joint_targets = np.array(mujoco_joint_targets)

    urdf_mujoco = mujoco.MjModel.from_xml_string(urdf_content)
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml') as tempfileObj:
        mujoco.mj_saveLastXML(tempfileObj.name, urdf_mujoco)
        tempfileObj.seek(0)
        urdf_mujoco = tempfileObj.read().decode('utf-8')
    robot_assets = re.search(r'<asset>(.*?)</asset>',         urdf_mujoco, re.DOTALL).group(1)
    robot_bodies = re.search(r'<worldbody>(.*?)</worldbody>', urdf_mujoco, re.DOTALL).group(1)
    with open(mjcf_path, 'r') as fileObj:
        mjcf_mujoco = fileObj.read()
    mjcf_mujoco = mjcf_mujoco.replace('$MY_ROBOT_DESCRIPTION_PATH', pkg_share_path)
    mjcf_mujoco = mjcf_mujoco.replace('</asset>', robot_assets + '\n    </asset>')
    #mjcf_mujoco = mjcf_mujoco.replace('<include file=\"'+usdf_file+'\"/>', robot_bodies)
    mjcf_mujoco = re.sub(r'<include\s+file=["\']' + re.escape(usdf_file) + r'["\']\s*/>', robot_bodies, mjcf_mujoco)
    mjcf_mujoco = mjcf_mujoco.replace('name="calfSphere_FL"', 'name="calfSphere_FL" class="foot_friction"')
    mjcf_mujoco = mjcf_mujoco.replace('name="calfSphere_FR"', 'name="calfSphere_FR" class="foot_friction"')
    mjcf_mujoco = mjcf_mujoco.replace('name="calfSphere_BL"', 'name="calfSphere_BL" class="foot_friction"')
    mjcf_mujoco = mjcf_mujoco.replace('name="calfSphere_BR"', 'name="calfSphere_BR" class="foot_friction"')
    print(mjcf_mujoco)

    mujoco_model = mujoco.MjModel.from_xml_string(mjcf_mujoco)
    mujoco_data  = mujoco.MjData(mujoco_model)
    if mujoco_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mujoco_model, mujoco_data, 0)
    with mujoco.viewer.launch_passive(mujoco_model, mujoco_data) as viewer:
        viewer.opt.geomgroup[0] = 0
        while viewer.is_running():
            step_start = time.time()
            mujoco_data.ctrl[:] = mujoco_joint_targets

            mujoco.mj_step(mujoco_model, mujoco_data)
            accel_data = mujoco_data.sensor('accel').data
            gyro_data  = mujoco_data.sensor('gyro').data
            quat_data  = mujoco_data.sensor('quat').data
            roll_rad, pitch_rad, yaw_rad = quat2euler(*quat_data)
            roll_deg  = math.degrees(roll_rad)
            pitch_deg = math.degrees(pitch_rad)
            yaw_deg   = math.degrees(yaw_rad)
            print(f"Roll: {roll_deg:6.1f} | Pitch: {pitch_deg:6.1f} | Yaw: {yaw_deg:6.1f}", end='\r')           
 
            viewer.sync()
            time_until_next_step = mujoco_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
#############################################################################################################################
if __name__ == '__main__': main()

