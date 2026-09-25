from ament_index_python.packages import get_package_share_directory
import os, re, time, math, copy
import tempfile
import numpy as np

import xacro
import mujoco, mujoco.viewer
import pinocchio
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
 
from ._GlobalFuncs import *
PKG_SHARE_PATH = get_package_share_directory('my_robot_description')
#############################################################################################################################
class CubicDoggoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode      = render_mode
        self.viewer           = None
        self.text_update_time = 0.001     #s
        self.last_text_update = 0.0

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
            mjcf_mujoco = mjcf_mujoco.replace('name="calfSphere_'+leg_prefix+'"', 
                                              'name="calfSphere_'+leg_prefix+'" class="foot_friction"')
        self.mujoco_model = mujoco.MjModel.from_xml_string(mjcf_mujoco)
        self.mujoco_data  = mujoco.MjData(self.mujoco_model)
        if self.mujoco_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.mujoco_model, self.mujoco_data, 0)

        self.mujoco_foot_ids = [mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, 'calfSphere_'+leg_prefix)
                                for leg_prefix in self.leg_prefixes]
        self.mujoco_floor_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.ray_geomid = np.zeros(1, dtype=np.int32)
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
        self.num_actions = len(self.joint_names)
        dim_observations = 12 + 12 + 3 + 2 + 1     # joint_pos (12), joint_vel (12), gyro (3), roll/pitch (2), height (1)
        self.action_space      = spaces.Box(low=-1.0,    high=1.0,    shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim_observations,), dtype=np.float32)

        self.action_scale   = 0.2
        self.last_joint_vel = None
        self.last_action    = None
        self.initial_pose = copy.deepcopy(self.mujoco_data.qpos[-12:])
    def _get_obs(self):
        joint_pos = self.mujoco_data.qpos[-12:]
        joint_vel = self.mujoco_data.qvel[-12:]
        imu_gyro  = self.mujoco_data.sensor('gyro').data
        quat_data = self.mujoco_data.sensor('quat').data
        imu_roll_rad, imu_pitch_rad, _ = quat2euler(*quat_data)

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
                                         geomgroup=None, flg_static=1, bodyexclude=mujoco_base_id, geomid=self.ray_geomid)
        privileged_height = rayCast_distance if rayCast_distance >= 0 else mujoco_base_curr[2]

        return np.concatenate([joint_pos, joint_vel, imu_gyro, 
                               [imu_roll_rad, imu_pitch_rad, privileged_height]]).astype(np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #print('CubicDoggoEnv(): reset(): restarting robot after time:', self.mujoco_data.time)
        
        self.last_text_update = 0.0
        self.last_joint_vel   = None
        self.last_action      = None
        mujoco.mj_resetData(self.mujoco_model, self.mujoco_data)
        if self.mujoco_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.mujoco_model, self.mujoco_data, 0)


        ############################################################################## initial state randomization

        #self.mujoco_data.qpos[-12:] += self.np_random.uniform(-np.pi/18, np.pi/18, size=12)

        ##############################################################################


        mujoco.mj_forward(self.mujoco_model, self.mujoco_data)
        return self._get_obs(), {}
    def step(self, action):
        target_ctrl = self.initial_pose + action*self.action_scale
        self.mujoco_data.ctrl[:] = target_ctrl
        
        mujoco_step_per_action = 4                                      ############## mujoco step fineness
        for _ in range(mujoco_step_per_action):
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)
        observations = self._get_obs()
        data_joint_pos = observations[0:12]
        data_joint_vel = observations[12:24]
        data_gyro      = observations[24:27]
        data_roll, data_pitch, data_height = observations[27:] 

        sim_lin_vel      = self.mujoco_data.sensor('linvel').data[:2]       # XY velocity
        sim_joint_torque = self.mujoco_data.actuator_force
        


        ############################################################################## reward/penalty parameters
        target_roll, target_pitch = 0.0, 0.0
        target_height             = 0.15

        reward_roll_pitch_scale = 2.0
        reward_roll_pitch_sigma = 0.05
        reward_height_scale     = 2.0
        reward_height_sigma     = 0.05
        penalty_joint_vel_scale = 0.0001
        penalty_joint_acc_scale = 2.5e-7
        penalty_ang_vel_scale   = 0.05

        penalty_lin_vel_scale      = 0.5
        penalty_joint_torque_scale = 1.0e-4
        penalty_joint_power_scale  = 2.0e-4     
        penalty_slip_scale         = 0.0

        penalty_action_scale      = 0.001 
        penalty_action_rate_scale = 0.01
        ############################################################################## termination conditions
        terminated = bool((np.pi/6 < abs(data_roll)) or (np.pi/6 < abs(data_pitch)) or
                          ((data_height < 0.1) or (0.2 < data_height)))
        truncated  = False
        ##############################################################################        



        residual_orientation = np.square(data_roll  - target_roll) + np.square(data_pitch - target_pitch)       
        residual_height      = np.square(data_height - target_height)
        residual_vel         = 0.0
        residual_action      = 0.0
        residual_slip        = 0.0
        if hasattr(self, 'last_joint_vel') and self.last_joint_vel is not None:
            joint_acc = (data_joint_vel - self.last_joint_vel)/(self.mujoco_model.opt.timestep*mujoco_step_per_action)
            residual_vel = np.square(joint_acc)
        if self.last_action is not None:
            residual_action = np.square(action - self.last_action)
        ###
        foot_contacts = [False]*4
        for contact_id in range(self.mujoco_data.ncon):
            contact = self.mujoco_data.contact[contact_id]
            for mujoco_foot_idx, mujoco_foot_id in enumerate(self.mujoco_foot_ids):
                if (contact.geom1 == mujoco_foot_id and contact.geom2 == self.mujoco_floor_id) or \
                   (contact.geom2 == mujoco_foot_id and contact.geom1 == self.mujoco_floor_id): 
                    foot_contacts[mujoco_foot_idx] = True
        for foot_vel, in_contact in zip(self.feet_vel, foot_contacts):
            if in_contact == True:
                residual_slip += np.sum(np.square(foot_vel[:2]))

        reward  = reward_roll_pitch_scale*np.exp(-residual_orientation/reward_roll_pitch_sigma) 
        reward += reward_height_scale    *np.exp(-residual_height     /reward_height_sigma)
        reward -= penalty_joint_vel_scale   *np.sum(np.square(data_joint_vel))
        reward -= penalty_joint_acc_scale   *np.sum(residual_vel)
        reward -= penalty_ang_vel_scale     *np.sum(np.square(data_gyro))
        reward -= penalty_lin_vel_scale     *np.sum(np.square(sim_lin_vel))
        reward -= penalty_joint_torque_scale*np.sum(np.square(sim_joint_torque))
        reward -= penalty_joint_power_scale *np.sum(np.abs(sim_joint_torque*data_joint_vel))
        reward -= penalty_slip_scale        *       residual_slip
        reward -= penalty_action_scale      *np.sum(np.square(action))
        reward -= penalty_action_rate_scale *np.sum(residual_action)
        self.last_joint_vel = data_joint_vel.copy()
        self.last_action    = action.copy()
  
        if (self.mujoco_data.time - self.last_text_update) > self.text_update_time: 
            telemetry_str  = f"Roll:{data_roll:9.5f}deg | Pitch:{data_pitch:9.5f}deg | Height:{data_height:9.5f}m | "
            telemetry_str += f"Time:{self.mujoco_data.time:9.5f}s"
            self.last_text_update = copy.deepcopy(self.mujoco_data.time)
            #print(telemetry_str)
        if self.render_mode == "human":
            if self.viewer is None:
               self.viewer = mujoco.viewer.launch_passive(self.mujoco_model, self.mujoco_data)
               self.last_text_update = 0.0
            if (self.mujoco_data.time - self.last_text_update) > self.text_update_time:
                self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                                      "TELEMETRY", telemetry_str))
            self.viewer.sync()
        return observations, reward, terminated, truncated, {}
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
#############################################################################################################################
def main():
    policy_model_path = PKG_SHARE_PATH.replace("install/my_robot_description/share/my_robot_description", 
                                               "ppo_tensorboards/")
    policy_model_name = "ppo_cubic_doggo_stand"
    policy_model_file = os.path.join(policy_model_path, policy_model_name+".zip")
    n_envs = min(os.cpu_count(), 16)
    
    ############################################################################## neural network
    #default_policy_kwargs = dict(activation_fn=torch.nn.Tanh,
    #                             net_arch=[dict(pi=[64, 64], vf=[64, 64])])
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=[256, 256, 128],                  # Policy/Actor network layers
                                       vf=[256, 256, 128]))                 # Value/Critic network layers
    ############################################################################## visualization or headless
    #ppo_env = CubicDoggoEnv(render_mode="human")
    ppo_env = make_vec_env(CubicDoggoEnv, n_envs=n_envs)
    ##############################################################################

    if os.path.exists(policy_model_file):
        print("cubic_doggo_mujoco_stand_ppo(): continuing PPO model:", policy_model_file) 
        ppo_model = PPO.load(
            policy_model_file, 
            env=ppo_env, 
            device="cpu",
            verbose=1,
            tensorboard_log=policy_model_path
        )
    else:
        print("cubic_doggo_mujoco_stand_ppo(): initializing PPO model")
        ppo_model = PPO("MlpPolicy", 
                        ppo_env,
                        device="cpu",
                        verbose=1,
                        policy_kwargs=policy_kwargs,
                        learning_rate=3e-4,
                        n_steps=2048,
                        batch_size=128,
                        n_epochs=10,
                        gamma=0.99,
                        gae_lambda=0.95,
                        tensorboard_log=policy_model_path)
    
    print("cubic_doggo_mujoco_stand_ppo(): start training, with "+str(n_envs)+" CPU cores...")
    ppo_model.learn(total_timesteps=1_000_000)
    ppo_model.save(policy_model_path+policy_model_name)
    print("cubic_doggo_mujoco_stand_ppo(): model saved:", policy_model_path+policy_model_name+".zip")
    

#############################################################################################################################
if __name__ == '__main__': main()




