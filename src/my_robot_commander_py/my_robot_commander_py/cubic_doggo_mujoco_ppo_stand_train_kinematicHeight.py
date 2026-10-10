from ament_index_python.packages import get_package_share_directory
import os, re, time, datetime, math, copy
import tempfile
import numpy as np

import xacro
import mujoco, mujoco.viewer
import pinocchio
import torch
torch.use_deterministic_algorithms(True)
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, CSVOutputFormat, HumanOutputFormat, TensorBoardOutputFormat
 
from ._GlobalFuncs import *
PKG_SHARE_PATH = get_package_share_directory('my_robot_description')
##################################################################################################################################
class CubicDoggoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.viewer      = None

        self.joint_number = 12
        self.gyro_number  = 3 
        self.obs_dim      = self.joint_number + self.joint_number + self.gyro_number + 2 + 1

        self.time_per_step     = 0.003   # s, default = 0.002
        self.stepN_per_action  = 4       # steps
        self.skip_first_stepN  = 4       # steps 
        self.truncate_max_time = 20.0    # s
        self.text_update_time  = 1.0     # s
        self.last_text_update  = 0.0

        self.leg_prefixes = ['FL', 'FR', 'BL', 'BR']
        self.joint_names = []
        for leg_prefix in self.leg_prefixes:
            self.joint_names.append('servo1_servo1_padding_'+leg_prefix)
            self.joint_names.append('servo2_servo2_padding_'+leg_prefix)
            self.joint_names.append('servo3_calfFeet_'      +leg_prefix)

        xacro_path = os.path.join(PKG_SHARE_PATH, 'urdf', 'cubic_doggo.urdf.xacro')
        mjcf_path  = os.path.join(PKG_SHARE_PATH, 'urdf', 'cubic_doggo.mujoco.ppo_stand.xml')
        usdf_file  =                                      'cubic_doggo.mujoco.urdf'

        xacro_raw = xacro.process_file(xacro_path)
        urdf_raw  = xacro_raw.toxml()
        urdf_content = urdf_raw.replace('package://my_robot_description', PKG_SHARE_PATH)
        urdf_content = urdf_content.replace('</robot>', '<mujoco><compiler discardvisual="false"/></mujoco></robot>')

        ################
        urdf_mujoco = mujoco.MjModel.from_xml_string(urdf_content)
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml') as tempfileObj:
            mujoco.mj_saveLastXML(tempfileObj.name, urdf_mujoco)
            tempfileObj.seek(0)
            urdf_mujoco = tempfileObj.read()
        robot_assets = re.search(r'<asset>(.*?)</asset>',         urdf_mujoco, re.DOTALL).group(1)
        robot_bodies = re.search(r'<worldbody>(.*?)</worldbody>', urdf_mujoco, re.DOTALL).group(1)
        with open(mjcf_path, 'r') as fileObj:
            mjcf_mujoco = fileObj.read()
        mjcf_mujoco = mjcf_mujoco.replace('$MY_ROBOT_DESCRIPTION_PATH', PKG_SHARE_PATH)
        mjcf_mujoco = mjcf_mujoco.replace('</asset>', robot_assets + '\n    </asset>')
        mjcf_mujoco = re.sub(r'<include\s+file=["\']' + re.escape(usdf_file) + r'["\']\s*/>', robot_bodies, mjcf_mujoco)
        for leg_prefix in self.leg_prefixes:
            #mjcf_mujoco = mjcf_mujoco.replace('name="calfSphere_'+leg_prefix+'"', 
            #                                  'name="calfSphere_'+leg_prefix+'" class="foot_friction"')
            body_name = f'calfSphere_{leg_prefix}'
            pattern = rf'(<body[^>]*name="{body_name}"[^>]*>\s*<geom)'
            mjcf_mujoco = re.sub(pattern, rf'\1 name="{body_name}" class="foot_friction"', mjcf_mujoco)
        self.mujoco_model = mujoco.MjModel.from_xml_string(mjcf_mujoco)
        self.mujoco_data  = mujoco.MjData(self.mujoco_model)
        self.mujoco_model.opt.timestep = self.time_per_step
        if self.mujoco_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.mujoco_model, self.mujoco_data, 0)

        self.mujoco_root_body_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, 'robot_root')
        self.mujoco_foot_body_ids, self.mujoco_foot_ids = [], []
        for leg_prefix in self.leg_prefixes:
            feet_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, 'calfFeet_'+leg_prefix)
            first_geom = self.mujoco_model.body_geomadr[feet_id]
            num_geoms  = self.mujoco_model.body_geomnum[feet_id]
            for geom_id in range(first_geom, first_geom + num_geoms):
                if self.mujoco_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    self.mujoco_foot_body_ids.append(geom_id)                   # getting calfSphere body_id by shape matching
                    break
            self.mujoco_foot_ids.append(first_geom + (num_geoms - 1))           # getting calfSphere geom_id by order matching

        self.mujoco_floor_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.ray_geom_id     = np.zeros(1, dtype=np.int32)
        self.init_cond_id    = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_KEY, "initial_condition")
        base_q = self.mujoco_model.key_qpos[self.init_cond_id, 3:7]
        self.base_q = pinocchio.Quaternion(base_q[0], base_q[1], base_q[2], base_q[3])
        ###
        urdf_pinocchio = re.sub(r'<joint name="world_base_link".*?</joint>', '', urdf_content, flags=re.DOTALL)
        urdf_pinocchio = urdf_pinocchio.replace('<link name="world"/>', '')
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.urdf') as tempfileObj:
            tempfileObj.write(urdf_pinocchio)
            tempfileObj.flush()
            self.pinocchio_model = pinocchio.buildModelFromUrdf(tempfileObj.name, pinocchio.JointModelFreeFlyer())
            self.pinocchio_data  = self.pinocchio_model.createData()

        self.pinocchio_leg_ids        = [self.pinocchio_model.getFrameId('calfSphere_'+leg_prefix) 
                                         for leg_prefix in self.leg_prefixes]
        self.pinocchio_base_id        = self.pinocchio_model.getFrameId('base_link')
        self.pinocchio_joint_pos      = pinocchio.neutral(self.pinocchio_model)
        self.pinocchio_joint_vel      = np.zeros(self.pinocchio_model.nv)
        self.pinocchio_joint_mappings = []
        for joint_name in self.joint_names:
            mujoco_joint_id    = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            pinocchio_joint_id = self.pinocchio_model.getJointId(joint_name)
            if mujoco_joint_id != -1 and pinocchio_joint_id < len(self.pinocchio_model.joints):
                mujoco_joint_q_idx = self.mujoco_model.jnt_qposadr[mujoco_joint_id]
                mujoco_joint_v_idx = self.mujoco_model.jnt_dofadr [mujoco_joint_id]
                pinocchio_joint_q_idx = self.pinocchio_model.joints[pinocchio_joint_id].idx_q
                pinocchio_joint_v_idx = self.pinocchio_model.joints[pinocchio_joint_id].idx_v
                self.pinocchio_joint_mappings.append((mujoco_joint_q_idx, mujoco_joint_v_idx, 
                                                      pinocchio_joint_q_idx, pinocchio_joint_v_idx))
        self.feet_pos, self.feet_vel = [], []
        ###
        self.obs_noise_schedule              = 0.0
        self.init_var_schedule               = 0.0
        self.rew_pen_schedule                = 1.0
        self.penalty_joint_pos_init_schedule = 1.0
        self.push_force_schedule             = 0.0
        self.push_interval  = 5                    # s 
        self.push_duration  = 0.1                  # s
        self.push_force_max = 15.0                 # N 
        self.push_force_vec = np.zeros(6)                
        self.last_push_time = 0
        ###
        self.action_range        = [-1.0,  1.0]
        self.action_delta_range  = [-0.05, 0.05]
        self.action_filter_alpha = 0.2
        self.action_scale        = 0.2
        self.num_actions = len(self.joint_names)
        self.action_space      = spaces.Box(low=-1.0,    high=1.0,    shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.last_joint_vel = np.zeros(self.joint_number)
        self.last_action    = np.zeros(self.action_space.shape, dtype=np.float32)
        self.initial_ctrl   = copy.deepcopy(self.mujoco_data.ctrl)
        self.term_time = 0
        self.term_keys = ["roll", "pitch", "height_low", "height_high"]
        self.term_data = {term_key:False for term_key in self.term_keys}
        self.reset_counters = np.zeros(1 + len(self.term_keys), dtype=np.int64)
    def _get_obs(self):
        joint_pos = self.mujoco_data.qpos[-self.joint_number:]
        joint_vel = self.mujoco_data.qvel[-self.joint_number:]
        imu_gyro  = self.mujoco_data.sensor('gyro').data
        quat_data = self.mujoco_data.sensor('quat').data
        imu_roll_rad, imu_pitch_rad, _ = quat2euler(*quat_data)
        ############################################################################## observable domain randomization
        if self.obs_noise_schedule > 0:
            dom_rand_joint_pos_scale      = np.pi/180
            dom_rand_joint_vec_scale      = np.pi/30
            dom_rand_imu_gyro_scale       = np.pi/90
            dom_rand_imu_roll_pitch_scale = np.pi/180
        ##############################################################################
            joint_pos += self.np_random.normal(0.0, dom_rand_joint_pos_scale*self.obs_noise_schedule, size=self.joint_number)
            joint_vel += self.np_random.normal(0.0, dom_rand_joint_vec_scale*self.obs_noise_schedule, size=self.joint_number)
            imu_gyro  += self.np_random.normal(0.0, dom_rand_imu_gyro_scale *self.obs_noise_schedule, size=self.gyro_number)
            imu_roll_rad  += self.np_random.normal(0.0, dom_rand_imu_roll_pitch_scale*self.obs_noise_schedule)
            imu_pitch_rad += self.np_random.normal(0.0, dom_rand_imu_roll_pitch_scale*self.obs_noise_schedule)
    
        self.feet_pos, self.feet_vel = [], []
        for mujoco_joint_q_idx, mujoco_joint_v_idx, pinocchio_joint_q_idx, pinocchio_joint_v_idx \
        in self.pinocchio_joint_mappings:
            self.pinocchio_joint_pos[pinocchio_joint_q_idx] = self.mujoco_data.qpos[mujoco_joint_q_idx]
            self.pinocchio_joint_vel[pinocchio_joint_v_idx] = self.mujoco_data.qvel[mujoco_joint_v_idx]
        pinocchio.forwardKinematics(self.pinocchio_model, self.pinocchio_data, 
                                    self.pinocchio_joint_pos, self.pinocchio_joint_vel)
        pinocchio.updateFramePlacements(self.pinocchio_model, self.pinocchio_data)
        pinocchio_base_frame = self.pinocchio_data.oMf[self.pinocchio_base_id]
        for leg_id in self.pinocchio_leg_ids:
            self.feet_pos.append(pinocchio_base_frame.actInv(self.pinocchio_data.oMf[leg_id]).translation)
            self.feet_vel.append(pinocchio.getFrameVelocity(self.pinocchio_model, self.pinocchio_data, 
                                                            leg_id, pinocchio.ReferenceFrame.WORLD).linear)
        kinematic_height = np.average([foot_pos[2] for foot_pos in self.feet_pos])
        ###
        mujoco_base_id   = self.mujoco_model.body('robot_root').id
        mujoco_base_curr = self.mujoco_data.xpos[mujoco_base_id]
        rayCast_distance = mujoco.mj_ray(m=self.mujoco_model, d=self.mujoco_data, pnt=mujoco_base_curr, vec=[0.0, 0.0, -1.0],
                                         geomgroup=None, flg_static=1, bodyexclude=mujoco_base_id, geomid=self.ray_geom_id)
        privileged_height = rayCast_distance if rayCast_distance >= 0 else mujoco_base_curr[2]

        observations = np.concatenate([joint_pos, joint_vel, imu_gyro, 
                                       [imu_roll_rad, imu_pitch_rad, kinematic_height]]).astype(np.float32)
        return observations
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #print('CubicDoggoEnv(): reset(): restarting robot after time:', self.mujoco_data.time)
        self.term_time = self.mujoco_data.time
        self.reset_counters[-1] += 1
        for term_idx, term_key in enumerate(self.term_keys):
            #print('  '+term_key+':', self.term_data[term_key])    
            if self.term_data[term_key] == True:
                self.reset_counters[term_idx] += 1
            self.term_data[term_key] = False
        self.last_text_update = 0.0
        self.last_joint_vel   = np.zeros(self.joint_number)
        self.last_action      = np.zeros(self.action_space.shape, dtype=np.float32)
        mujoco.mj_resetData(self.mujoco_model, self.mujoco_data)
        if self.mujoco_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.mujoco_model, self.mujoco_data, 0)
        else:
            self.mujoco_data.qpos[:] = self.mujoco_model.qpos0
            self.mujoco_data.qvel[:] = 0.0
        self.last_push_time                              = 0.0
        self.push_force_vec[:]                           = 0.0
        self.mujoco_data.xfrc_applied[self.mujoco_root_body_id] = 0.0
        ############################################################################## initial state domain randomization
        if self.init_var_schedule > 0:
            init_joint_pos_var_range   = np.array([-np.pi/18, np.pi/18])
            init_rot_var_range         = np.array([-np.pi,    np.pi])
        ##############################################################################
            # pos_var            
            pos_shift = self.np_random.uniform(*(self.init_var_schedule*init_joint_pos_var_range), size=self.joint_number)
            self.mujoco_data.qpos[-self.joint_number:] += pos_shift
            # rot_var
            yaw_angle = self.np_random.uniform(*(self.init_var_schedule*init_rot_var_range))
            angle_R = pinocchio.rpy.rpyToMatrix(0.0, 0.0, yaw_angle)
            angle_q = pinocchio.Quaternion(angle_R)
            angle_q = angle_q*self.base_q
            self.mujoco_data.qpos[3:7] = np.array([angle_q.w, angle_q.x, angle_q.y, angle_q.z], dtype=np.float64)

        if self.skip_first_stepN > 0:
            for _ in range(self.skip_first_stepN):
                self.mujoco_data.ctrl[:] = self.initial_ctrl
                mujoco.mj_step(self.mujoco_model, self.mujoco_data)
        mujoco.mj_kinematics(self.mujoco_model, self.mujoco_data)
        mujoco.mj_forward(   self.mujoco_model, self.mujoco_data)
        if (self.render_mode == "human") and (self.viewer is not None):
            self.viewer.sync() 
        return self._get_obs(), {}
    def step(self, action):
        infos_dict = {}
        current_time = self.mujoco_data.time
        action = np.clip(action, *self.action_range)
        action_delta = np.clip(action - self.last_action, *self.action_delta_range)
        action = self.last_action + action_delta
        filtered_action = self.action_filter_alpha*action + (1.0 - self.action_filter_alpha)*self.last_action   #low pass filter
        target_ctrl = self.initial_ctrl + filtered_action*self.action_scale
        self.mujoco_data.ctrl[:] = target_ctrl

        if self.push_force_schedule > 0:
            if (current_time - self.last_push_time) >= self.push_interval:
                force_angle     = self.np_random.uniform(0, 2*np.pi)
                force_magnitude = self.np_random.uniform(0.0, self.push_force_max)*self.push_force_schedule
                self.push_force_vec[0] = force_magnitude*np.cos(force_angle)
                self.push_force_vec[1] = force_magnitude*np.sin(force_angle)
                self.push_force_vec[2] = 0.0 
                self.last_push_time = current_time
            if (current_time - self.last_push_time) >= self.push_duration:
                self.push_force_vec[:] = 0.0
            self.mujoco_data.xfrc_applied[self.mujoco_root_body_id] = self.push_force_vec
    
        for _ in range(self.stepN_per_action):
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)
        ###
        obs_dict = {}
        observations = self._get_obs()
        data_joint_pos = observations[0:self.joint_number]
        data_joint_vel = observations[self.joint_number:(2*self.joint_number)]
        data_gyro      = observations[(2*self.joint_number):(2*self.joint_number+self.gyro_number)]
        data_roll, data_pitch, data_height = observations[(2*self.joint_number+self.gyro_number):] 

        sim_lin_vel      = self.mujoco_data.sensor('linvel').data[:2]       # XY velocity
        sim_joint_torque = self.mujoco_data.actuator_force[:]
        sim_leg_torque   = np.sum(np.abs(sim_joint_torque.reshape(4, 3)), axis=1)
        
        for obs_idx in range(len(data_joint_pos)):
            obs_dict["y_data_joint_pos"+str(obs_idx)] = data_joint_pos[obs_idx]
        for obs_idx in range(len(data_joint_vel)):
            obs_dict["y_data_joint_vel"+str(obs_idx)] = data_joint_vel[obs_idx]
        for obs_idx in range(len(data_gyro)):
            obs_dict["data_gyro"+str(obs_idx)] = data_gyro[obs_idx]
        obs_dict["data_roll"]   = data_roll
        obs_dict["data_pitch"]  = data_pitch
        obs_dict["data_height"] = data_height
        for obs_idx in range(len(sim_lin_vel)):
            obs_dict["x_sim_lin_vel"+str(obs_idx)] = sim_lin_vel[obs_idx]
        for obs_idx in range(len(sim_joint_torque)):
            obs_dict["x_sim_joint_torque"+str(obs_idx)] = sim_joint_torque[obs_idx]
        for obs_idx in range(len(sim_leg_torque)):
            obs_dict["x_sim_leg_torque"+str(obs_idx)] = sim_leg_torque[obs_idx]
        for obs_idx in range(len(self.push_force_vec)):
            obs_dict["x_sim_push_force"+str(obs_idx)] = self.push_force_vec[obs_idx]
        ############################################################################## reward/penalty parameters
        target_roll, target_pitch = 0.0, 0.0
        target_height             = 0.14984        #0.15689 for previleged

        reward_roll_pitch_scale = 1.0
        reward_roll_pitch_sigma = 0.03          /self.rew_pen_schedule
        reward_height_scale     = 1.0
        reward_height_sigma     = 0.05          /self.rew_pen_schedule
        penalty_joint_vel_scale = 1.0E-3
        penalty_joint_acc_scale = 1.0E-5
        penalty_ang_vel_scale   = 0.5           *self.rew_pen_schedule
        penalty_yaw_rate_scale  = 1.0           *self.rew_pen_schedule

        penalty_lin_vel_scale        = 10.0     *self.rew_pen_schedule
        penalty_joint_torque_scale   = 1.0E-4
        penalty_joint_power_scale    = 2.0E-4     
        penalty_leg_torque_scale     = 0.5      *self.rew_pen_schedule
        penalty_slip_scale           = 0.5
        penalty_joint_pos_init_scale = 0.1

        penalty_action_scale      = 0.005 
        penalty_action_rate_scale = 1.0
        ############################################################################## termination conditions
        self.term_data["roll"]        = (np.pi/6 < abs(data_roll))
        self.term_data["pitch"]       = (np.pi/6 < abs(data_pitch))
        self.term_data["height_low"]  = (data_height < target_height-0.05)
        self.term_data["height_high"] = (target_height+0.05 < data_height)
        terminated = bool(self.term_data["roll"]       or self.term_data["pitch"] or 
                          self.term_data["height_low"] or self.term_data["height_high"])
        truncated = bool(self.mujoco_data.time >= self.truncate_max_time)               # truncated is termination without penalty
        ##############################################################################        

        residual_roll        = np.square(data_roll - target_roll) 
        residual_pitch       = np.square(data_pitch - target_pitch)
        residual_orientation = residual_roll + residual_pitch
        residual_height      = np.square(data_height - target_height)
        residual_pos         = np.square(data_joint_pos - self.initial_ctrl)
        residual_leg_torque  = np.std(sim_leg_torque)
        residual_vel         = 0.0
        residual_action      = 0.0
        residual_slip        = 0.0
        if np.sum(self.last_joint_vel) != 0:
            joint_acc = (data_joint_vel - self.last_joint_vel)/(self.mujoco_model.opt.timestep*self.stepN_per_action)
            residual_vel = np.square(joint_acc)
        if np.sum(self.last_action) != 0:
            residual_action = np.square(action - self.last_action)
        
        foot_contacts = np.zeros(len(self.leg_prefixes), dtype=bool)
        for contact_id in range(self.mujoco_data.ncon):
            geom_contact = self.mujoco_data.contact[contact_id]
            for foot_idx, foot_id in enumerate(self.mujoco_foot_ids):
                if (geom_contact.geom1 == self.mujoco_floor_id and geom_contact.geom2 == foot_id) or \
                   (geom_contact.geom2 == self.mujoco_floor_id and geom_contact.geom1 == foot_id): 
                    foot_contacts[foot_idx] = True
        for foot_vel, in_contact in zip(self.feet_vel, foot_contacts):
            if in_contact == True:
                residual_slip += np.sum(np.square(foot_vel[:2]))
        #print("cubic_doggo_mujoco_ppo_stand_train(): contact geom_ids:", 
        #      [[self.mujoco_data.contact[contact_id].geom1, self.mujoco_data.contact[contact_id].geom2] 
        #        for contact_id in range(self.mujoco_data.ncon)])

        obs_dict["z_residual_roll"]        = residual_roll
        obs_dict["z_residual_pitch"]       = residual_pitch
        obs_dict["z_residual_orientation"] = residual_orientation
        obs_dict["z_residual_height"]      = residual_height
        obs_dict["z_residual_pos"]         = residual_pos
        obs_dict["z_residual_vel"]         = residual_vel
        obs_dict["z_residual_action"]      = residual_action
        obs_dict["z_residual_slip"]        = residual_slip
        infos_dict["obs_dict"] = obs_dict
        ### 
        reward_dict = {}
        reward_dict["reward_roll_pitch"] = reward_roll_pitch_scale*np.exp(-residual_orientation/reward_roll_pitch_sigma)
        reward_dict["reward_height"]     = reward_height_scale    *np.exp(-residual_height     /reward_height_sigma)
        reward_dict["penalty_joint_vel"]    = -penalty_joint_vel_scale   *np.sum(np.square(data_joint_vel))
        reward_dict["penalty_joint_acc"]    = -penalty_joint_acc_scale   *np.sum(residual_vel)
        reward_dict["penalty_ang_vel"]      = -penalty_ang_vel_scale     *np.sum(np.square(data_gyro))
        reward_dict["penalty_yaw_rate"]     = -penalty_yaw_rate_scale    *np.square(data_gyro[2])
        reward_dict["penalty_lin_vel"]      = -penalty_lin_vel_scale     *np.sum(np.square(sim_lin_vel))
        reward_dict["penalty_joint_torque"] = -penalty_joint_torque_scale*np.sum(np.square(sim_joint_torque))
        reward_dict["penalty_joint_power"]  = -penalty_joint_power_scale *np.sum(np.abs(sim_joint_torque*data_joint_vel))
        reward_dict["penalty_leg_torque"]   = -penalty_leg_torque_scale  *residual_leg_torque
        reward_dict["penalty_slip"]         = -penalty_slip_scale        *residual_slip
        reward_dict["penalty_action"]       = -penalty_action_scale      *np.sum(np.square(action))
        reward_dict["penalty_action_rate"]  = -penalty_action_rate_scale *np.sum(residual_action)
        reward_dict["penalty_joint_pos_init"]  = -penalty_joint_pos_init_scale*np.sum(residual_pos)
        reward_dict["penalty_joint_pos_init"] *= self.penalty_joint_pos_init_schedule        
        reward = float(sum(reward_dict.values()))
        infos_dict["reward_dict"] = reward_dict

        reward_ratio_dict = {}
        reward_ratio_dict["reward_roll_pitch_ratio"] = reward_dict["reward_roll_pitch"]/reward_roll_pitch_scale
        reward_ratio_dict["reward_height_ratio"]     = reward_dict["reward_height"]    /reward_height_scale
        reward_ratio_dict["reward_full_ratio"]       = reward/(reward_roll_pitch_scale + reward_height_scale)
        infos_dict["reward_ratio_dict"] = reward_ratio_dict

        self.last_joint_vel = data_joint_vel.copy()
        self.last_action    = action.copy()
        ###
        if (self.render_mode == "human") and (self.viewer is None):
            self.viewer = mujoco.viewer.launch_passive(self.mujoco_model, self.mujoco_data)
            self.camera_initialized = False
            self.last_text_update = 0.0 
        if (self.mujoco_data.time - self.last_text_update) > self.text_update_time: 
            telemetry_str  = f"Roll:{data_roll:9.5f}deg | Pitch:{data_pitch:9.5f}deg | Height:{data_height:9.5f}m | "
            telemetry_str += f"Time:{self.mujoco_data.time:9.5f}s"
            if self.render_mode == "human":
                self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                                      "TELEMETRY", telemetry_str))
            self.last_text_update = copy.deepcopy(self.mujoco_data.time)
        if self.render_mode == "human":
            self._render_force_arrow()
            self.viewer.sync()
        return observations, reward, terminated, truncated, infos_dict
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    def _render_force_arrow(self):
        if self.viewer is None:
            return
        arrow_radius  = 0.008        #m
        max_arrow_len = 1.0         #m
        force_xyz = np.array(self.push_force_vec[:3], dtype=np.float64)
        force_mag = float(np.linalg.norm(force_xyz))
        force_vec = max_arrow_len*force_xyz/self.push_force_max

        viewer_scn = self.viewer.user_scn
        viewer_scn.ngeom = 0

        start_pos = np.array(self.mujoco_data.xpos[self.mujoco_root_body_id], dtype=np.float64)
        if float(np.linalg.norm(force_vec)) < 1e-6:
            return
        end_pos = start_pos + force_vec
        
        viewer_geom = viewer_scn.geoms[viewer_scn.ngeom]
        mujoco.mjv_connector(viewer_geom, mujoco.mjtGeom.mjGEOM_ARROW, arrow_radius, start_pos, end_pos)
        viewer_geom.rgba[:] = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        viewer_scn.ngeom += 1
    ###
    def get_term_time(self):
        return self.term_time
    def get_reset_counters(self) -> np.ndarray:
        return self.reset_counters
    def set_reset_counters(self, counters: list):
        self.reset_counters = np.array(counters, dtype=np.int64)
    ###
    def set_obs_noise_schedule(self, new_scale: float):
        self.obs_noise_schedule = new_scale
    def set_init_var_schedule(self, new_scale: float):
        self.init_var_schedule = new_scale
    def set_rew_pen_schedule(self, new_scale: float):
        self.rew_pen_schedule = new_scale
    def set_penalty_joint_pos_init_schedule(self, new_scale: float):
        self.penalty_joint_pos_init_schedule = new_scale
    def set_push_force_schedule(self, new_scale: float):
        self.push_force_schedule = new_scale
class CurriculumSchedulingCallback(BaseCallback):
    def __init__(self, total_timesteps, scale_thres=1.0E-6, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.scale_thres     = scale_thres
    def _on_step(self) -> bool:
        progress_ratio = self.num_timesteps/self.total_timesteps


        ############################################################################## curriculum scheduling
        # obs_noise
        obs_noise_scale = min(1.0, 2*progress_ratio)
        self.training_env.env_method("set_obs_noise_schedule", obs_noise_scale)
        # init_var
        init_var_scale = min(1.0, 2*progress_ratio)
        self.training_env.env_method("set_init_var_schedule", init_var_scale)
        # reward_penalty
        y_final_val       = 2.0
        x_mid_point       = 0.4
        sigmoid_steepness = 12.0
        rew_pen_scale = 1.0 + (y_final_val - 1.0)/(1.0 + np.exp(-sigmoid_steepness*(progress_ratio - x_mid_point)))
        self.training_env.env_method("set_rew_pen_schedule", rew_pen_scale)
        # penalty_joint_pos_init
        decay_rate = 5.0
        pos_init_scale = np.exp(-decay_rate*progress_ratio)
        pos_init_scale = pos_init_scale if (pos_init_scale > self.scale_thres) else 0.0
        self.training_env.env_method("set_penalty_joint_pos_init_schedule", pos_init_scale)      
        # push_force
        push_force_scale = 0.0#max(0.0, min(1.0, (progress_ratio - 0.2)/0.5))
        self.training_env.env_method("set_push_force_schedule", push_force_scale)
        ##############################################################################
        self.logger.record("z_schedulers/obs_noise",              obs_noise_scale)
        self.logger.record("z_schedulers/init_var",               init_var_scale)
        self.logger.record("z_schedulers/rew_pen",                rew_pen_scale)
        self.logger.record("z_schedulers/penalty_joint_pos_init", pos_init_scale)
        self.logger.record("z_schedulers/push_force",             push_force_scale)
        ###
        term_time = np.mean(self.training_env.env_method("get_term_time"))
        self.logger.record("rollout/ep_len_time", term_time)
        ###
        reset_counters = sum(self.training_env.env_method("get_reset_counters"))
        total_resets = reset_counters[-1] if reset_counters[-1] > 0 else 1
        self.logger.record("z_reward_ratios/termination_roll",        float(reset_counters[0])/total_resets)
        self.logger.record("z_reward_ratios/termination_pitch",       float(reset_counters[1])/total_resets)
        self.logger.record("z_reward_ratios/termination_height_low",  float(reset_counters[2])/total_resets)
        self.logger.record("z_reward_ratios/termination_height_high", float(reset_counters[3])/total_resets)
        self.logger.record("z_reward_ratios/termination_total",       reset_counters[-1])
        ###
        infos = self.locals.get("infos")
        if infos and "obs_dict" in infos[0]:
            for reward_key, reward_val in infos[0]["obs_dict"].items():
                self.logger.record("z_observables/"+reward_key, reward_val)
        if infos and "reward_dict" in infos[0]:
            for reward_key, reward_val in infos[0]["reward_dict"].items():
                self.logger.record("z_rewards/"+reward_key, reward_val) 
        if infos and "reward_ratio_dict" in infos[0]:
            for reward_key, reward_val in infos[0]["reward_ratio_dict"].items():
                self.logger.record("z_reward_ratios/"+reward_key, reward_val)
        return True
##################################################################################################################################
def export_ppo_to_onnx(ppo_model, obs_dim, onnx_save_path):
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
##################################################################################################################################
def main():
    render_mode = None
    policy_model_name = "cubic_doggo_stand_"+str(datetime.date.today().strftime("%y%m%d"))+"_1_"
    #render_mode = "human"
    #policy_model_name = "cubic_doggo_stand_261009_3_"

    policy_model_path = PKG_SHARE_PATH.replace("install/my_robot_description/share/my_robot_description", "ppo_tensorboards/")
    os.makedirs(policy_model_path, exist_ok=True)
    policy_model_file, policy_model_idx = findSaveFile(policy_model_path, policy_model_name, ".zip")
    n_envs = min(os.cpu_count(), 16)
    ############################################################################## neural network
    #default_policy_kwargs = dict(activation_fn=torch.nn.Tanh,
    #                             net_arch=[dict(pi=[64, 64], vf=[64, 64])])
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         ortho_init=True,
                         net_arch=dict(pi=[256, 256, 128],                  # Policy/Actor network layers
                                       vf=[256, 256, 128]))                 # Value/Critic network layers
    ############################################################################## visualization or headless
    rand_seed       = 1
    ppo_checkpointN = 40
    ppo_stepN       = 1_000_000                             # minimum is n_envs*n_steps, 16*2048 = 32768
    if render_mode == "human":
        ppo_env = make_vec_env(lambda: CubicDoggoEnv(render_mode=render_mode), n_envs=1)
        reset_num_timesteps = False#True
    else:
        ppo_env = make_vec_env(lambda: CubicDoggoEnv(render_mode=render_mode), n_envs=n_envs)
        reset_num_timesteps = False
    ##############################################################################
    if policy_model_file == None:
        print("cubic_doggo_mujoco_ppo_stand_train(): initializing PPO model")
        ppo_env = VecNormalize(ppo_env,
                               norm_obs=True,
                               norm_reward=True,
                               clip_obs=10.0,
                               clip_reward=10.0,
                               gamma=0.99)
        ppo_model = PPO("MlpPolicy", 
                        ppo_env,
                        device="cpu",
                        verbose=1,
                        seed=rand_seed,
                        policy_kwargs=policy_kwargs,
                        learning_rate=1.0E-5,
                        target_kl=0.03,
                        vf_coef=1.0,
                        max_grad_norm=0.5,
                        n_steps=2048,
                        batch_size=128,
                        n_epochs=10,
                        gamma=0.99,
                        gae_lambda=0.95)
    else:
        print("cubic_doggo_mujoco_ppo_stand_train(): continuing PPO model:", policy_model_file)
        ppo_env = VecNormalize.load(policy_model_file.replace(".zip", "_vec_norm.pkl"), ppo_env)
        ppo_model = PPO.load(policy_model_file, 
                             env=ppo_env, 
                             device="cpu",
                             verbose=1,
                             seed=rand_seed)
        if hasattr(ppo_model.policy, "user_data") and "reset_counters" in ppo_model.policy.user_data:
            saved_counters = ppo_model.policy.user_data["reset_counters"]
            counters_per_env = (np.array(saved_counters) // ppo_env.num_envs).tolist()
            ppo_env.env_method("set_reset_counters", counters_per_env)

    print("cubic_doggo_mujoco_ppo_stand_train(): start training, with "+str(ppo_env.num_envs)+" CPU cores...")
    start_time = time.time()
    for ppo_idx in range(ppo_checkpointN-policy_model_idx+int(render_mode == "human")):
        ppo_checkpoint_idx = ppo_idx + policy_model_idx + 1 
        ppo_model_save_name = policy_model_path + policy_model_name + str(ppo_checkpoint_idx)

        log_writers = [HumanOutputFormat(sys.stdout),
                       TensorBoardOutputFormat(policy_model_path + policy_model_name),
                       CSVOutputFormat(policy_model_path + policy_model_name + "_log.csv")]
        ppo_model.set_logger(Logger(folder=policy_model_path, output_formats=log_writers))
        curriculum_callback = CurriculumSchedulingCallback(total_timesteps=(ppo_stepN*ppo_checkpointN))
        ppo_model.learn(total_timesteps=ppo_stepN, 
                        callback=curriculum_callback,
                        reset_num_timesteps=reset_num_timesteps,
                        tb_log_name=policy_model_name)
        elapsed_time = time.time() - start_time
        if getattr(ppo_env, "render_mode", None) != "human":
            total_counters = sum(ppo_env.env_method("get_reset_counters"))
            ppo_model.policy.user_data = {"reset_counters": total_counters.tolist()}
            ppo_model.save(ppo_model_save_name)
            print("cubic_doggo_mujoco_ppo_stand_train(): model checkpoint saved:", ppo_model_save_name+".zip")
            ppo_env.save(ppo_model_save_name+"_vec_norm.pkl")
            print("cubic_doggo_mujoco_ppo_stand_train(): vector normalization saved:", ppo_model_save_name+"_vec_norm.pkl")
            export_ppo_to_onnx(ppo_model, ppo_env.observation_space.shape[0], ppo_model_save_name+".onnx")
            print("cubic_doggo_mujoco_ppo_stand_train(): model export onnx saved:", ppo_model_save_name+".onnx")
        print("cubic_doggo_mujoco_ppo_stand_train(): total time used", elapsed_time, "s") 
        print("---------------------------------------------------------------------------------------------------------------\n")
    print("cubic_doggo_mujoco_ppo_stand_train(): end of code")

##################################################################################################################################
if __name__ == '__main__': main()




