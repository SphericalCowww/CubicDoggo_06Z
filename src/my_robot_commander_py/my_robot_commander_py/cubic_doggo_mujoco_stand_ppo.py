from ament_index_python.packages import get_package_share_directory
import os, re, time, math, copy
import tempfile
import numpy as np

import xacro
import mujoco, mujoco.viewer
import pinocchio
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
        self.render_mode = render_mode

        self.leg_prefixes = ['FL', 'FR', 'BL', 'BR']
        self.joint_names = []
        for leg_prefix in self.leg_prefixes:
            self.joint_names.append('servo1_servo1_padding_'+leg_prefix)
            self.joint_names.append('servo2_servo2_padding_'+leg_prefix)
            self.joint_names.append('servo3_calfFeet_'      +leg_prefix)

        xacro_path = os.path.join(PKG_SHARE_PATH, 'urdf', 'cubic_doggo.urdf.xacro')
        mjcf_path  = os.path.join(PKG_SHARE_PATH, 'urdf', 'cubic_doggo.mujoco_stand.xml')
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

        urdf_pinocchio = re.sub(r'<joint name="world_base_link".*?</joint>', '', urdf_content, flags=re.DOTALL)
        urdf_pinocchio = urdf_pinocchio.replace('<link name="world"/>', '')
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.urdf') as tempfileObj:
            tempfileObj.write(urdf_pinocchio)
            tempfileObj.flush()
            self.pinocchio_model = pinocchio.buildModelFromUrdf(tempfileObj.name, pinocchio.JointModelFreeFlyer())
            self.pinocchio_data  = self.pinocchio_model.createData()

        self.pinocchio_leg_ids = [self.pinocchio_model.getFrameId('calfSphere_'+leg_prefix) 
                                  for leg_prefix in self.leg_prefixes]
        self.pinocchio_base_id = self.pinocchio_model.getFrameId('base_link')
        self.ray_geomid        = np.zeros(1, dtype=np.int32)
 
        self.num_actions = len(self.joint_names)
        dim_observations = 12 + 12 + 3 + 2 + 1     # joint_pos (12), joint_vel (12), gyro (3), roll/pitch (2), height (1)
        self.action_space      = spaces.Box(low=-1.0,    high=1.0,    shape=(self.num_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim_observations,), dtype=np.float32)

        self.action_scale = 0.2
        self.last_action  = None
        self.initial_pose = copy.deepcopy(self.mujoco_data.qpos[-12:])
    def _get_obs(self):
        joint_pos = self.mujoco_data.qpos[-12:]
        joint_vel = self.mujoco_data.qvel[-12:]
        imu_gyro  = self.mujoco_data.sensor('gyro').data
        quat_data = self.mujoco_data.sensor('quat').data
        imu_roll_rad, imu_pitch_rad, _ = quat2euler(*quat_data)

        '''
        feet_currs = []
        pinocchio_joint_currs = pinocchio.neutral(pinocchio_model)
        for joint_name in self.joint_names:
            mujoco_joint_id    = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            pinocchio_joint_id = pinocchio_model.getJointId(joint_name)
            if mujoco_joint_id != -1 and pinocchio_joint_id < len(self.pinocchio_model.joints):
                mujoco_joint_idx    = self.mujoco_model.jnt_qposadr[mujoco_joint_id]
                pinocchio_joint_idx = self.pinocchio_model.joints[pinocchio_joint_id].idx_q
                pinocchio_joint_currs[pinocchio_joint_idx] = copy.deepcopy(self.mujoco_data.qpos[mujoco_joint_idx])
        pinocchio.forwardKinematics(self.pinocchio_model, self.pinocchio_data, pinocchio_joint_currs)
        pinocchio.updateFramePlacements(self.pinocchio_model, self.pinocchio_data)
        pinocchio_base_frame = self.pinocchio_data.oMf[pinocchio_base_id]
        for leg_prefix, leg_id in zip(self.leg_prefixes, self.pinocchio_leg_ids):
            feet_currs.append(pinocchio_base_frame.actInv(self.pinocchio_data.oMf[leg_id]).translation)
        kinematic_height = np.average([feet_curr[2] for feet_curr in feet_currs])
        '''

        mujoco_base_id   = self.mujoco_model.body('robot_root').id
        mujoco_base_curr = self.mujoco_data.xpos[mujoco_base_id]
        rayCast_distance = mujoco.mj_ray(m=self.mujoco_model, d=self.mujoco_data, pnt=mujoco_base_curr, vec=[0.0, 0.0, -1.0],
                                         geomgroup=None, flg_static=1, bodyexclude=mujoco_base_id, geomid=self.ray_geomid)
        privileged_height = rayCast_distance if rayCast_distance >= 0 else mujoco_base_curr[2]

        return np.concatenate([joint_pos, joint_vel, imu_gyro, 
                               [imu_roll_rad, imu_pitch_rad, privileged_height]]).astype(np.float32)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.mujoco_model, self.mujoco_data)
        if self.mujoco_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.mujoco_model, self.mujoco_data, 0)


        ############################################################################## variation in conditions

        self.mujoco_data.qpos[-12:] += self.np_random.uniform(-0.05, 0.05, size=12)

        ##############################################################################


        mujoco.mj_forward(self.mujoco_model, self.mujoco_data)
        return self._get_obs(), {}
    def step(self, action):
        target_ctrl = self.initial_pose + action*self.action_scale
        self.mujoco_data.ctrl[:] = target_ctrl
 
        ############################################################################## variable targets
        target_pitch, target_roll = 0.0, 0.0
        target_height             = 0.15

        reward_exp_factor         = 0.01
        reward_pitch_roll_scale   = 1.4
        reward_height_scale       = 2.0
        penalty_joint_vel_scale   = 0.001
        penalty_action_scale      = 0.01 
        penalty_action_rate_scale = 0.05
        
        mujoco_step_per_action = 4
        ##############################################################################
        for _ in range(mujoco_step_per_action):
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)

        observations = self._get_obs()
        data_pos  = observations[0:12]
        data_vel  = observations[12:24]
        data_giro = observations[24:27]
        data_pitch, data_roll, data_height = observations[27:]

        residual_orientation = -np.square(data_pitch - target_pitch) - np.square(data_roll - target_roll)       
        residual_height      = -np.square(data_height - target_height)

        reward_orientation  = np.exp(-residual_orientation /(reward_exp_factor*np.square(2)))
        reward_height       = np.exp(-residual_height      / reward_exp_factor)
        penalty_action      = -np.sum(np.square(action))
        penalty_action_rate = 0
        if self.last_action is not None:
            penalty_action_rate = -np.sum(np.square(action - self.last_action))
        penalty_joint_vel = -np.sum(np.square(data_vel))

        reward  = reward_pitch_roll_scale*reward_orientation + reward_height_scale*reward_height
        reward += penalty_joint_vel_scale*penalty_joint_vel
        reward +=  penalty_action_scale*penalty_action + penalty_action_rate_scale*penalty_action_rate
        terminated = bool((0.6 < abs(data_pitch)) or (0.6 < abs(data_roll)) or (data_height < 0.08) or (0.18 < data_height))
        truncated  = False

        self.last_action = action.copy()
        return observations, reward, terminated, truncated, {}
#############################################################################################################################
def main():
    ppo_env = make_vec_env(CubicDoggoEnv, n_envs=8)
    ppo_model = PPO("MlpPolicy", 
                    ppo_env,
                    verbose=1,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=128,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    tensorboard_log=(PKG_SHARE_PATH.replace("src/my_robot_description/urdf", "")+"/ppo_tensorboards/"))
    
    print("cubic_doggo_mujoco_stand_ppo(): start training...")
    ppo_model.learn(total_timesteps=1_000_000)
    ppo_model.save("ppo_cubic_doggo_stand")
    print("cubic_doggo_mujoco_stand_ppo(): model saved:")
    

#############################################################################################################################
if __name__ == '__main__': main()




