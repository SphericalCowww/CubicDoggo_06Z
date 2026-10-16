import os, sys, pathlib, time, re, glob, math, copy
import tempfile, pickle, json
import numpy as np
import mujoco, mujoco.viewer
import pinocchio
import torch
import onnx
import onnxruntime as ort
import ast

#############################################################################################################################
def quat2euler(w, x, y, z):
    sinr_cosp = 2*(w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2*(w*y - z*x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi/2, sinp) # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

#############################################################################################################################
def getLegIK(pinocchio_model, pinocchio_data, pinocchio_joint_inits, pinocchio_leg_ids, target_positions, 
             damping_factor=1e-6, delta_time=0.1, max_iter=100, eps=1e-4):
    joint_angle = pinocchio_joint_inits.copy()
    for _ in range(max_iter):
        pinocchio.forwardKinematics(pinocchio_model, pinocchio_data, joint_angle)
        pinocchio.updateFramePlacements(pinocchio_model, pinocchio_data)
        pinocchio_base_frame = pinocchio_data.oMf[pinocchio_model.getFrameId('base_link')] 
 
        delta_positions, joint_Jacobs = [], []
        for leg_id, target_position in zip(pinocchio_leg_ids, target_positions):
            #curr_position = pinocchio_data.oMf[leg_id].translation                               # world frame
            curr_position = pinocchio_base_frame.actInv(pinocchio_data.oMf[leg_id]).translation   # base frame
            delta_positions.append(curr_position - target_position)
            
            full_Jacob = pinocchio.computeFrameJacobian(pinocchio_model, pinocchio_data, joint_angle, leg_id, 
                                                        pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            joint_Jacobs.append(full_Jacob[:, 6:])  # with only the leg Jacobian

        delta_position_full = np.concatenate(delta_positions)
        if np.linalg.norm(delta_position_full) < eps:
            break

        joint_Jacob_all = np.vstack(joint_Jacobs)
        joint_velocity = np.zeros(pinocchio_model.nv)
        joint_velocity[6:] = -joint_Jacob_all.T @ np.linalg.inv(
            joint_Jacob_all @ joint_Jacob_all.T + damping_factor*np.eye(joint_Jacob_all.shape[0])) @ delta_position_full
    
        joint_angle = pinocchio.integrate(pinocchio_model, joint_angle, joint_velocity*delta_time)
    return joint_angle
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
def renderForceArrow(viewer, mujoco_data, mujoco_body_id, push_force_vec, push_force_max):
    if viewer is None:
        return
    arrow_radius  = 0.008        #m
    max_arrow_len = 1.0         #m
    force_xyz = np.array(push_force_vec, dtype=np.float64)
    force_mag = float(np.linalg.norm(force_xyz))
    force_vec = max_arrow_len*force_xyz/push_force_max

    viewer_scn = viewer.user_scn
    viewer_scn.ngeom = 0

    start_pos = np.array(mujoco_data.xpos[mujoco_body_id], dtype=np.float64)
    if float(np.linalg.norm(force_vec)) < 1e-6:
        return
    end_pos = start_pos + force_vec
        
    viewer_geom = viewer_scn.geoms[viewer_scn.ngeom]
    mujoco.mjv_connector(viewer_geom, mujoco.mjtGeom.mjGEOM_ARROW, arrow_radius, start_pos, end_pos)
    viewer_geom.rgba[:] = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    viewer_scn.ngeom += 1
#############################################################################################################################
def exportPPO_ONNX(ppo_model, obs_dim, onnx_save_path, config=None):
    policy_net = ppo_model.policy
    policy_net.to("cpu")
    policy_net.eval()
    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32, device="cpu")
    class OnnxPolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
        def forward(self, obs):
            features = self.policy.extract_features(obs)
            latent_pi, _ = self.policy.mlp_extractor(features)
            actions = self.policy.action_net(latent_pi)
            return actions
    onnx_wrapper = OnnxPolicyWrapper(policy_net)
    onnx_wrapper.eval()
    torch.onnx.export(
        onnx_wrapper,
        dummy_input,
        onnx_save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action':      {0: 'batch_size'}},
        dynamo=False)
    if config is not None:
        onnx_model = onnx.load(onnx_save_path)
        for key, val in config.items():
            meta = onnx_model.metadata_props.add()
            meta.key = str(key)
            meta.value = str(val)
        onnx.save(onnx_model, onnx_save_path)
def extractONNXconfig(onnx_path):
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    env_config = {}
    for key, val in ort_session.get_modelmeta().custom_metadata_map.items():
        try:
            env_config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            env_config[key] = val 
    return env_config
def findSaveFile(policy_model_path, policy_model_name, file_type):
    ppo_files = glob.glob(os.path.join(policy_model_path, policy_model_name+"*"+file_type))
    if len(ppo_files) != 0:
        ppo_iters = [int(ppo_file.split(policy_model_name)[1].split(file_type)[0]) for ppo_file in ppo_files]
        return os.path.join(policy_model_path, policy_model_name+str(max(ppo_iters))+file_type), max(ppo_iters)
    else:
        return None, 0
def exportPKLvecNorm_JSON(vec_norm_pkl_path, json_save_path):
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
#############################################################################################################################

def main(): pass
if __name__ == '__main__': main()
 


