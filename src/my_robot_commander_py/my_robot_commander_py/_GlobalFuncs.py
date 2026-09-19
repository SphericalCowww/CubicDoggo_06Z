import os, sys, pathlib, time, re, glob, math, copy
import numpy as np
import pinocchio

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
#############################################################################################################################
def main(): pass
if __name__ == '__main__': main()
 


