from ament_index_python.packages import get_package_share_directory
import os, re, time, math, copy
import threading
import tempfile, pickle, json
import numpy as np

import xacro
import mujoco, mujoco.viewer
import pygame
import pinocchio
import onnxruntime as ort
import ast

from ._GlobalFuncs import *
##################################################################################################################################
def main():
    leg_prefixes = ['FL', 'FR', 'BL', 'BR']
    joint_names = []
    for leg_prefix in leg_prefixes:
        joint_names.append('servo1_servo1_padding_'+leg_prefix)
        joint_names.append('servo2_servo2_padding_'+leg_prefix)
        joint_names.append('servo3_calfFeet_'      +leg_prefix)

    #model_name = 'policy_stand0'
    model_name = 'policy_stand1_force'
    reinforcement_path = get_package_share_directory('my_robot_commander_py')
    norm_path = os.path.join(reinforcement_path, 'deploy_model', model_name+'_vecNorm.pkl')
    onnx_path = os.path.join(reinforcement_path, 'deploy_model', model_name+'_model.onnx')
#    norm_path = "/home/cubicdoggo/Documents/CubicDoggo_06Z/ppo_tensorboards/cubic_doggo_stand_261016_6_40_vec_norm.pkl"
#    onnx_path = "/home/cubicdoggo/Documents/CubicDoggo_06Z/ppo_tensorboards/cubic_doggo_stand_261016_6_40.onnx"

    pkg_share_path = get_package_share_directory('my_robot_description')
    xacro_path     = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.urdf.xacro')
    mjcf_path      = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.mujoco.ppo_stand.xml')
#    mjcf_path      = os.path.join(pkg_share_path, 'urdf', 'cubic_doggo.mujoco.ppo_stand_slope.xml')
    usdf_file      =                                      'cubic_doggo.mujoco.urdf'

    
    ################
    print("cubic_doggo_mujoco_ppo_stand_deploy(): loading model ONNX file:", onnx_path)
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name  = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    meta_map = ort_session.get_modelmeta().custom_metadata_map
    env_config = {}
    for key, val in meta_map.items():
        try:
            env_config[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            env_config[key] = val
        print('  '+key+':', env_config[key])
    
    height_def = env_config["height_def"] if "height_def" in env_config else "kinematic"
    joint_number = env_config["joint_number"]
    action_scale = env_config["action_scale"]
    action_range = env_config["action_range"]
    action_delta_range = env_config["action_delta_range"]
    action_filter_alpha = env_config["action_filter_alpha"]
    stepN_per_action = env_config["stepN_per_action"]

    exportPKLvecNorm_JSON(norm_path, norm_path.replace('install/my_robot_commander_py/share/', 'src/').replace('.pkl', '.json'))
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
    force_max = 5.0                                     # N 
    force_ctrl = ArrowKeyForceController(force_max=force_max)
    push_force_vec = np.zeros(6, dtype=np.float64)
    mujoco_root_body_id = mujoco_model.body('robot_root').id
    ##############################################################################################################################
    text_update_time   = 0.1                            # s
    action_delay_time  = 1.0                            # s
    delta_t = mujoco_model.opt.timestep
    ##############################################################################################################################

    initial_ctrl = copy.deepcopy(mujoco_data.ctrl)
    last_action  = np.zeros(joint_number, dtype=np.float32)
    ray_geomid       = np.zeros(1, dtype=np.int32)
    last_text_update = 0.0
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
            raw_obs = None
            if height_def == "kinematic":
                raw_obs = np.concatenate([joint_pos, joint_vel, imu_gyro, 
                                          [roll_rad, pitch_rad, kinematic_height]]).astype(np.float32)
            elif height_def == "privileged":
                raw_obs = np.concatenate([joint_pos, joint_vel, imu_gyro, [roll_rad, pitch_rad]]).astype(np.float32)
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
            force_x, force_y, force_mag, force_ang = force_ctrl.get_force()
            push_force_vec[:] = 0.0
            push_force_vec[0] = force_x
            push_force_vec[1] = force_y
            mujoco_data.xfrc_applied[mujoco_root_body_id] = push_force_vec
            renderForceArrow(viewer, mujoco_data, mujoco_root_body_id, push_force_vec[:3], force_max)
            ###########################################
            for _ in range(stepN_per_action):
                mujoco.mj_step(mujoco_model, mujoco_data)
          
            if (mujoco_data.time - last_text_update) > text_update_time:
                telemetry_str  = f"Roll:{roll_deg:5.1f}deg | Pitch:{pitch_deg:5.1f}deg | Yaw:{yaw_deg:5.1f}deg\n"
                telemetry_str += f"Privileged Height:{privileged_height:9.5f}m | Kinematic Height:{kinematic_height:9.5f}m\n"
                telemetry_str += f"Model: {model_name}\n"
                telemetry_str += f"Force Magnitude: {force_mag:.1f}N | Force Angle: {force_ang:3.1f}deg"
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPRIGHT, 
                                  "TELEMETRY", telemetry_str))
                last_text_update = copy.deepcopy(mujoco_data.time)
                print(telemetry_str.replace('\n', ' | '), end='\r')
            elif (last_text_update - mujoco_data.time) > text_update_time:
                last_text_update = copy.deepcopy(mujoco_data.time)

            viewer.sync()
            time_until_next_step = delta_t*stepN_per_action - (time.time() - step_start) 
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    force_ctrl.stop()
##################################################################################################################################
class ArrowKeyForceController:
    def __init__(self, force_max=5.0):
        self.mag_scale       = 0.5
        self.force_ang_res   = 3.0              # deg
        self.force_ang_swipe = 30               # Hz
        self.force_max       = force_max        # N
        self.force_x       = 0.0                # N
        self.force_y       = 0.0                # N
        self.force_mag_val = 0.0                # N
        self.force_ang_val = 0.0                # deg
        self.last_ang_tick = 0.0
        self._running = True
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    def _run(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = '1800, 600'
        pygame.init()
        pygame.display.set_caption("Keyboard Control")
        pygame_screen = pygame.display.set_mode((280, 140))
        pygame_font   = pygame.font.Font(None, 24)
        pygame_clock  = pygame.time.Clock()

        self.last_ang_tick = time.monotonic()
        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        self.mag_scale = min(1.0, self.mag_scale + 0.1)
                    elif event.key == pygame.K_s:
                        self.mag_scale = max(0.0, self.mag_scale - 0.1)
                    elif event.key == pygame.K_LEFT:
                        self.force_ang_val = (self.force_ang_val - self.force_ang_res)%360.0
                        self.last_ang_tick = time.monotonic()
                    elif event.key == pygame.K_RIGHT:
                        self.force_ang_val = (self.force_ang_val + self.force_ang_res)%360.0
                        self.last_ang_tick = time.monotonic()
            pygame_now  = time.monotonic()
            pygame_keys = pygame.key.get_pressed()
            left_held  = pygame_keys[pygame.K_LEFT]
            right_held = pygame_keys[pygame.K_RIGHT]
            hold_period = 1.0/self.force_ang_swipe
            if (left_held or right_held) and (pygame_now - self.last_ang_tick) >= hold_period:
                if left_held:
                    self.force_ang_val = (self.force_ang_val - self.force_ang_res)%360.0
                if right_held:
                    self.force_ang_val = (self.force_ang_val + self.force_ang_res)%360.0
                self.last_ang_tick = pygame_now
            up_held   = pygame_keys[pygame.K_UP]
            down_held = pygame_keys[pygame.K_DOWN]
            force_sign = 0
            if up_held and not down_held:
                force_sign = +1
            elif down_held and not up_held:
                force_sign = -1
            
            force_mag   = force_sign*self.force_max*self.mag_scale
            force_theta = math.radians(self.force_ang_val)
            force_x     = -force_mag*math.cos(force_theta)
            force_y     = -force_mag*math.sin(force_theta)
            with self._lock:
                self.force_x, self.force_y = force_x, force_y
            self.force_mag_val = force_mag 
            
            pygame_display1 = pygame_font.render("w: increases force magnitude",           True, (255, 255, 255))
            pygame_display2 = pygame_font.render("s: decreases force magnitude",           True, (255, 255, 255))
            pygame_display3 = pygame_font.render("left/right: change force direction",     True, (255, 255, 255))
            pygame_display4 = pygame_font.render("up/down: applying force",                True, (255, 255, 255))
            pygame_screen.fill((0, 0, 0))
            pygame_screen.blit(pygame_display1, (5, 5))
            pygame_screen.blit(pygame_display2, (5, 25))
            pygame_screen.blit(pygame_display3, (5, 45))
            pygame_screen.blit(pygame_display4, (5, 65))
            
            pygame.display.flip()
            pygame_clock.tick(60)
        pygame.quit()
    def get_force(self):
        with self._lock:
            return self.force_x, self.force_y, self.force_mag_val, self.force_ang_val
    def stop(self):
        self._running = False
##################################################################################################################################
if __name__ == '__main__': main()





