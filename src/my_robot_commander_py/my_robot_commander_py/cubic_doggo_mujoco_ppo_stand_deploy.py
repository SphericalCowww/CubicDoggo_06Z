from ament_index_python.packages import get_package_share_directory
import os, re, time, math, copy
import tempfile, pickle, json
import numpy as np

import xacro
import mujoco, mujoco.viewer
import pinocchio
import onnxruntime as ort

from ._GlobalFuncs import *
##################################################################################################################################
def export_vec_norm_to_json(vec_norm_pkl_path, json_save_path):
    print("cubic_doggo_mujoco_ppo_stand_deploy(): reading vector normalization PKL file:", vec_norm_pkl_path)
    with open(vec_norm_pkl_path, "rb") as file_obj:
        vec_norm = pickle.load(file_obj)
    norm_data = {
        "obs_mean": vec_norm.obs_rms.mean.tolist(),
        "obs_var":  vec_norm.obs_rms.var.tolist(),
        "clip_obs": float(getattr(vec_norm, "clip_obs", 10.0)),
        "epsilon":  1e-8}
    with open(json_save_path, "w") as file_obj:
        json.dump(norm_data, file_obj, indent=4)
    print("cubic_doggo_mujoco_ppo_stand_deploy(): exporting vector normalization to JSON:", json_save_path)
##################################################################################################################################
def main():
    leg_prefixes = ['FL', 'FR', 'BL', 'BR']
    joint_names = []
    for leg_prefix in leg_prefixes:
        joint_names.append('servo1_servo1_padding_'+leg_prefix)
        joint_names.append('servo2_servo2_padding_'+leg_prefix)
        joint_names.append('servo3_calfFeet_'      +leg_prefix)

    reinforcement_path = get_package_share_directory('my_robot_commander_py')
    norm_path  = os.path.join(reinforcement_path, 'deploy_model', 'cubic_doggo_stand_261009_2_19_vec_norm.pkl')
    onnx_path  = os.path.join(reinforcement_path, 'deploy_model', 'cubic_doggo_stand_261009_2_19.onnx')
    export_vec_norm_to_json(norm_path, norm_path.replace('install/my_robot_commander_py/share/', 'src/').replace('.pkl', '.json'))

    pkg_share_path = get_package_share_directory('my_robot_description')
    xacro_path     = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.urdf.xacro')
    mjcf_path      = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.mujoco.ppo_stand.xml')
    usdf_file      =                                      'cubic_doggo.mujoco.urdf'
    
    ################
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name  = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    with open(norm_path, 'rb') as file_obj:
        vec_norm = pickle.load(file_obj)
    
    obs_mean = vec_norm.obs_rms.mean
    obs_var  = vec_norm.obs_rms.var
    clip_obs = getattr(vec_norm, 'clip_obs', 10.0)

    def normalize_obs(raw_obs):
        norm_obs = (raw_obs - obs_mean) / np.sqrt(obs_var + 1e-8)
        return np.clip(norm_obs, -clip_obs, clip_obs).astype(np.float32)
    ################
    xacro_raw = xacro.process_file(xacro_path)
    urdf_raw  = xacro_raw.toxml()
    urdf_content = urdf_raw.replace('package://my_robot_description', pkg_share_path)
    urdf_content = urdf_content.replace('</robot>', '<mujoco><compiler discardvisual="false"/></mujoco></robot>')

    mujoco_model, mujoco_data = None, None
    urdf_mujoco = mujoco.MjModel.from_xml_string(urdf_content)
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml') as tempfileObj:
        mujoco.mj_saveLastXML(tempfileObj.name, urdf_mujoco)
        tempfileObj.seek(0)
        urdf_mujoco = tempfileObj.read()
    robot_assets = re.search(r'<asset>(.*?)</asset>',         urdf_mujoco, re.DOTALL).group(1)
    robot_bodies = re.search(r'<worldbody>(.*?)</worldbody>', urdf_mujoco, re.DOTALL).group(1)
    with open(mjcf_path, 'r') as file_obj:
        mjcf_mujoco = file_obj.read()
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
    pinocchio_leg_ids = [pinocchio_model.getFrameId('calfSphere_' + leg_prefix) for leg_prefix in leg_prefixes]
    pinocchio_base_id = pinocchio_model.getFrameId('base_link')
    pinocchio_joint_pos = pinocchio.neutral(pinocchio_model)
    pinocchio_joint_vel = np.zeros(pinocchio_model.nv)
    pinocchio_joint_mappings = []
    for joint_name in joint_names:
        mujoco_joint_id    = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        pinocchio_joint_id = pinocchio_model.getJointId(joint_name)
        if mujoco_joint_id != -1 and pinocchio_joint_id < len(pinocchio_model.joints):
            mujoco_joint_q_idx = mujoco_model.jnt_qposadr[mujoco_joint_id]
            mujoco_joint_v_idx = mujoco_model.jnt_dofadr[mujoco_joint_id]
            pinocchio_joint_q_idx = pinocchio_model.joints[pinocchio_joint_id].idx_q
            pinocchio_joint_v_idx = pinocchio_model.joints[pinocchio_joint_id].idx_v
            pinocchio_joint_mappings.append((mujoco_joint_q_idx, mujoco_joint_v_idx, pinocchio_joint_q_idx,pinocchio_joint_v_idx))
    ################
    joint_number = 12
    action_scale = 0.2
    action_range = [-1.0, 1.0]
    action_delta_range = [-0.05, 0.05]
    action_filter_alpha = 0.2
    stepN_per_action = 4

    text_update_time   = 0.1                         # s
    action_delay_time  = 1.0                         # s
    delta_t = mujoco_model.opt.timestep
    ##############################################################################################################################

    initial_ctrl = copy.deepcopy(mujoco_data.ctrl)
    last_action  = np.zeros(joint_number, dtype=np.float32)
    ray_geomid         = np.zeros(1, dtype=np.int32)
    last_text_update   = 0.0
    is_standing = False
    ray_geomid    = np.zeros(1, dtype=np.int32)
    ray_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    with mujoco.viewer.launch_passive(mujoco_model, mujoco_data) as viewer:
        viewer.opt.geomgroup[0] = 0
        while viewer.is_running():
            step_start = time.time()

            joint_pos = mujoco_data.qpos[-joint_number:]
            joint_vel = mujoco_data.qvel[-joint_number:]
            imu_gyro  = mujoco_data.sensor('gyro').data
            quat_data = mujoco_data.sensor('quat').data
            roll_rad, pitch_rad, yaw_rad = quat2euler(*quat_data)
            roll_deg, pitch_deg, yaw_deg = math.degrees(roll_rad),  math.degrees(pitch_rad),  math.degrees(yaw_rad)

            feet_pos = []
            for m_q, m_v, p_q, p_v in pinocchio_joint_mappings:
                pinocchio_joint_pos[p_q] = mujoco_data.qpos[m_q]
                pinocchio_joint_vel[p_v] = mujoco_data.qvel[m_v]
            pinocchio.forwardKinematics(pinocchio_model, pinocchio_data, pinocchio_joint_pos, pinocchio_joint_vel)
            pinocchio.updateFramePlacements(pinocchio_model, pinocchio_data)
            pinocchio_base_frame = pinocchio_data.oMf[pinocchio_base_id]
            for leg_id in pinocchio_leg_ids:
                feet_pos.append(pinocchio_base_frame.actInv(pinocchio_data.oMf[leg_id]).translation)
            kinematic_height = np.average([f[2] for f in feet_pos])

            mujoco_base_id   = mujoco_model.body('robot_root').id
            mujoco_base_curr = mujoco_data.xpos[mujoco_base_id] 
            rayCast_distance = mujoco.mj_ray(m=mujoco_model, d=mujoco_data, pnt=mujoco_base_curr, vec=ray_direction,
                                             geomgroup=None, flg_static=1, bodyexclude=mujoco_base_id, geomid=ray_geomid)
            privileged_height = rayCast_distance if rayCast_distance >= 0 else mujoco_base_curr[2]
            ###########################################
            raw_obs = np.concatenate([joint_pos, joint_vel, imu_gyro, [roll_rad, pitch_rad, kinematic_height]]).astype(np.float32)
            if mujoco_data.time > action_delay_time:
                norm_obs = normalize_obs(raw_obs)
                onnx_outputs = ort_session.run([output_name], {input_name: norm_obs.reshape(1, -1)})
                action = np.clip(onnx_outputs[0][0], *action_range)

                action_delta = np.clip(action - last_action, *action_delta_range)
                action = last_action + action_delta

                filtered_action = action_filter_alpha*action + (1.0 - action_filter_alpha)*last_action
                target_ctrl = initial_ctrl + filtered_action*action_scale
                mujoco_data.ctrl[:] = target_ctrl
                last_action = action.copy()
            ###########################################
            for _ in range(stepN_per_action):
                mujoco.mj_step(mujoco_model, mujoco_data)
           
            if (mujoco_data.time - last_text_update) > text_update_time:
                telemetry_str  = f"Roll:{roll_deg:4.1f}deg | Pitch:{pitch_deg:4.1f}deg | Yaw:{yaw_deg:4.1f}deg\n"
                telemetry_str += f"Privileged Height:{privileged_height:9.5f}m | Kinematic Height:{kinematic_height:9.5f}m"
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT, 
                                  "TELEMETRY", telemetry_str))
                last_text_update = copy.deepcopy(mujoco_data.time)
                print(telemetry_str.replace('\n', ' | '), end='\r')

            viewer.sync()
            time_until_next_step = delta_t*stepN_per_action - (time.time() - step_start) 
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
##################################################################################################################################
if __name__ == '__main__': main()

