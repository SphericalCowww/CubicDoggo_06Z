from ament_index_python.packages import get_package_share_directory
import os, re, time, math, copy
import tempfile
import numpy as np

import xacro
import mujoco, mujoco.viewer
import pinocchio

from ._GlobalFuncs import *
#############################################################################################################################
def sineWalkGait_getTarget(home_positions, gait_phase, swing_fraction, lift, x_stride, y_stride, x_shift, y_shift,
                           phase_offsets=[0.00, 0.50, 0.75, 0.25]):          # FL, FR, BL, BR
    target_feet = []
    for leg_idx in range(len(home_positions)):
        target_x = home_positions[leg_idx][0]
        target_y = home_positions[leg_idx][1]
        target_z = home_positions[leg_idx][2]
        
        is_group_a       = (leg_idx == 0 or leg_idx == 3)
        is_group_b       = (leg_idx == 1 or leg_idx == 2)
        is_group_backLeg = (leg_idx == 2 or leg_idx == 3)
        
        local_phase = copy.deepcopy(gait_phase)
        if swing_fraction <= 0.25:
            local_phase += phase_offsets[leg_idx]
        elif is_group_b:
            local_phase += 0.5
        if local_phase >= 1.0:
            local_phase -= 1.0

        x_offset, y_offset, z_offset = 0.0, 0.0, 0.0
        if local_phase < swing_fraction:
            swing_progress = local_phase/swing_fraction
            z_offset = lift*np.sin(swing_progress*np.pi)
            if swing_fraction >= 0.5:
                x_offset = -x_stride + 2.0*x_stride*swing_progress
                y_offset = -y_stride + 2.0*y_stride*swing_progress
            else:
                x_offset = -x_stride*np.cos(swing_progress*np.pi)
                y_offset = -y_stride*np.cos(swing_progress*np.pi)
        else:
            stance_progress = (local_phase - swing_fraction)/(1.0 - swing_fraction)
            z_offset = 0.0
            x_offset = x_stride - 2.0*x_stride*stance_progress
            y_offset = y_stride - 2.0*y_stride*stance_progress
        if is_group_backLeg:
            target_x -= (x_offset + x_shift)
        else:
            target_x += (x_offset + x_shift)
            
        target_y += (y_offset + y_shift)
        target_z -= z_offset
        target_feet.append(np.array([target_x, target_y, target_z]))
    return target_feet
#############################################################################################################################
def main():
    leg_prefixes = ['FL', 'FR', 'BL', 'BR']
    joint_names = []
    for leg_prefix in leg_prefixes:
        joint_names.append('servo1_servo1_padding_'+leg_prefix)
        joint_names.append('servo2_servo2_padding_'+leg_prefix)
        joint_names.append('servo3_calfFeet_'      +leg_prefix)
    feet_stand_targets = [
        np.array([  0.096,  0.152, 0.15]), # FL
        np.array([ -0.096,  0.152, 0.15]), # FR
        np.array([  0.096, -0.078, 0.15]), # BL
        np.array([ -0.096, -0.078, 0.15])  # BR
    ]

    pkg_share_path = get_package_share_directory('my_robot_description')
    xacro_path     = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.urdf.xacro')
    mjcf_path      = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.mujoco_walk.xml')
    usdf_file      =                                      'cubic_doggo.mujoco.urdf'

    xacro_raw = xacro.process_file(xacro_path)
    urdf_raw  = xacro_raw.toxml()
    urdf_content = urdf_raw.replace('package://my_robot_description', pkg_share_path)
    urdf_content = urdf_content.replace('</robot>', '<mujoco><compiler discardvisual="false"/></mujoco></robot>')

    ################
    mujoco_model, mujoco_data = None, None
    urdf_mujoco = mujoco.MjModel.from_xml_string(urdf_content)
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml') as tempfileObj:
        mujoco.mj_saveLastXML(tempfileObj.name, urdf_mujoco)
        tempfileObj.seek(0)
        urdf_mujoco = tempfileObj.read()
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
    mujoco_model = mujoco.MjModel.from_xml_string(mjcf_mujoco)
    mujoco_data  = mujoco.MjData(mujoco_model)
    if mujoco_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mujoco_model, mujoco_data, 0)

    pinocchio_model, pinocchio_data = None, None
    urdf_pinocchio = re.sub(r'<joint name="world_base_link".*?</joint>', '', urdf_content, flags=re.DOTALL)
    urdf_pinocchio = urdf_pinocchio.replace('<link name="world"/>', '')
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.urdf') as tempfileObj:
        tempfileObj.write(urdf_pinocchio)
        tempfileObj.flush()
        pinocchio_model = pinocchio.buildModelFromUrdf(tempfileObj.name, pinocchio.JointModelFreeFlyer())
        pinocchio_data  = pinocchio_model.createData()

    ################
    pinocchio_joint_inits = pinocchio.neutral(pinocchio_model)
    for joint_name in joint_names:
        mujoco_joint_id    = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        pinocchio_joint_id = pinocchio_model.getJointId(joint_name)
        if mujoco_joint_id != -1 and pinocchio_joint_id < len(pinocchio_model.joints):
            mujoco_joint_idx    = mujoco_model.jnt_qposadr[mujoco_joint_id]
            pinocchio_joint_idx = pinocchio_model.joints[pinocchio_joint_id].idx_q
            pinocchio_joint_inits[pinocchio_joint_idx] = copy.deepcopy(mujoco_data.qpos[mujoco_joint_idx])
    pinocchio.forwardKinematics(pinocchio_model, pinocchio_data, pinocchio_joint_inits)
    pinocchio.updateFramePlacements(pinocchio_model, pinocchio_data)

    pinocchio_leg_ids = [pinocchio_model.getFrameId('calfSphere_'+leg_prefix) for leg_prefix in leg_prefixes]
    pinocchio_base_id = pinocchio_model.getFrameId('base_link')
    pinocchio_joint_targets = getLegIK(pinocchio_model, pinocchio_data, pinocchio_joint_inits,
                                       pinocchio_leg_ids, feet_stand_targets)

    mujoco_ctrl_targets = []
    for joint_name in joint_names:
        pinocchio_joint_id = pinocchio_model.getJointId(joint_name)
        if pinocchio_joint_id < len(pinocchio_model.joints):
            pinocchio_joint_idx = pinocchio_model.joints[pinocchio_joint_id].idx_q
            mujoco_ctrl_targets.append(pinocchio_joint_targets[pinocchio_joint_idx])
    
    #########################################################################################################################
    text_update_time   = 0.1                         # s
    action_update_time = 3.0                         # s
    action_delay_time  = 3.0                         # s

    delta_t = mujoco_model.opt.timestep
    swing_fraction = 0.5
    gait_frequency = 1.5                            # Hz
    lift, x_shift, y_shift = 0.03, 0.0, -0.007      # m
    x_stride_range, y_stride_range = [-0.03, 0.03], [0.0, 0.04]
    #########################################################################################################################

    last_text_update   = 0.0
    last_action_update = 0.0
    is_standing = False
    gait_phase, x_stride, y_stride = 0.0, 0.0, 0.0
    ray_geomid    = np.zeros(1, dtype=np.int32)
    ray_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    with mujoco.viewer.launch_passive(mujoco_model, mujoco_data) as viewer:
        viewer.opt.geomgroup[0] = 0
        while viewer.is_running():
            step_start = time.time()

            feet_currs = []
            pinocchio_joint_currs = pinocchio.neutral(pinocchio_model)
            for joint_name in joint_names:
                mujoco_joint_id    = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                pinocchio_joint_id = pinocchio_model.getJointId(joint_name)
                if mujoco_joint_id != -1 and pinocchio_joint_id < len(pinocchio_model.joints):
                    mujoco_joint_idx    = mujoco_model.jnt_qposadr[mujoco_joint_id]
                    pinocchio_joint_idx = pinocchio_model.joints[pinocchio_joint_id].idx_q
                    pinocchio_joint_currs[pinocchio_joint_idx] = copy.deepcopy(mujoco_data.qpos[mujoco_joint_idx])
            pinocchio.forwardKinematics(pinocchio_model, pinocchio_data, pinocchio_joint_currs)
            pinocchio.updateFramePlacements(pinocchio_model, pinocchio_data)
            pinocchio_base_frame = pinocchio_data.oMf[pinocchio_base_id]
            for leg_prefix, leg_id in zip(leg_prefixes, pinocchio_leg_ids):
                feet_currs.append(pinocchio_base_frame.actInv(pinocchio_data.oMf[leg_id]).translation)
            kinematic_height  = np.average([feet_curr[2] for feet_curr in feet_currs])

            if mujoco_data.time > action_delay_time:
                if is_standing == False:
                    feet_errors = [np.linalg.norm(np.array(curr) - np.array(target)) 
                                   for curr, target in zip(feet_currs, feet_stand_targets)]
                    if np.max(feet_errors) < 0.01:
                        gait_phase = 0.0
                        is_standing = True
                        print('\nStand gait complete, walk gait starting...')
                else:
                    if (mujoco_data.time - last_action_update) > action_update_time:
                        if x_stride != 0.0:
                            x_stride = 0.0
                        else:
                            x_stride = np.random.uniform(*x_stride_range)
                        y_stride = np.random.uniform(*y_stride_range)
                        last_action_update = copy.deepcopy(mujoco_data.time)

                    gait_phase += gait_frequency*delta_t
                    if gait_phase >= 1.0:
                        gait_phase -= 1.0
                    feet_walk_targets = sineWalkGait_getTarget(feet_stand_targets, gait_phase, swing_fraction,
                                                               lift, x_stride, y_stride, x_shift, y_shift)
                    pinocchio_joint_targets = getLegIK(pinocchio_model, pinocchio_data, pinocchio_joint_inits, 
                                                       pinocchio_leg_ids, feet_walk_targets)
                    mujoco_ctrl_targets = []
                    for joint_name in joint_names:
                        pin_id = pinocchio_model.getJointId(joint_name)
                        if pin_id < len(pinocchio_model.joints):
                            pin_idx = pinocchio_model.joints[pin_id].idx_q
                            mujoco_ctrl_targets.append(pinocchio_joint_targets[pin_idx])
                mujoco_data.ctrl[:] = mujoco_ctrl_targets           
            mujoco.mj_step(mujoco_model, mujoco_data)
           
            mujoco_base_id   = mujoco_model.body('robot_root').id
            mujoco_base_curr = mujoco_data.xpos[mujoco_base_id] 
            rayCast_distance = mujoco.mj_ray(m=mujoco_model, d=mujoco_data, pnt=mujoco_base_curr, vec=ray_direction,
                                             geomgroup=None, flg_static=1, bodyexclude=mujoco_base_id, geomid=ray_geomid)
            privileged_height = rayCast_distance if rayCast_distance >= 0 else mujoco_base_curr[2]
            
            accel_data = mujoco_data.sensor('accel').data
            gyro_data  = mujoco_data.sensor('gyro').data
            quat_data  = mujoco_data.sensor('quat').data
            roll_rad, pitch_rad, yaw_rad = quat2euler(*quat_data)
            roll_deg  = math.degrees(roll_rad)
            pitch_deg = math.degrees(pitch_rad)
            yaw_deg   = math.degrees(yaw_rad)
            if (mujoco_data.time - last_text_update) > text_update_time:
                telemetry_str  = f"X Stride:{x_stride:7.3f}m | Y Stride:{y_stride:7.3f}m | "
                telemetry_str += f"Roll:{roll_deg:4.1f}deg | Pitch:{pitch_deg:4.1f}deg | Yaw:{yaw_deg:4.1f}deg\n"
                telemetry_str += f"Privileged Height:{privileged_height:8.4f}m | Kinematic Height:{kinematic_height:8.4f}m"
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT, 
                                  "TELEMETRY", telemetry_str))
                last_text_update = copy.deepcopy(mujoco_data.time)
                print(telemetry_str.replace('\n', ' | '), end='\r')

            viewer.sync()
            time_until_next_step = delta_t - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
#############################################################################################################################
if __name__ == '__main__': main()

