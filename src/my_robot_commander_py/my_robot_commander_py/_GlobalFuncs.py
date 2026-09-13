import os, sys, pathlib, time, re, glob, math
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
def solve_leg_ik(pinocchio_model, pinocchio_data, pinocchio_joint_inits, pinocchio_leg_ids, target_positions, 
                 damping_factor=1e-6, delta_time=0.1, max_iter=100, eps=1e-4):
    joint_angle = pinocchio_joint_inits.copy()
    for _ in range(max_iter):
        pinocchio.forwardKinematics(pinocchio_model, pinocchio_data, joint_angle)
        pinocchio.updateFramePlacements(pinocchio_model, pinocchio_data)
        oMbase = pinocchio_data.oMf[pinocchio_model.getFrameId('base_link')]        # oM: original frame 
 
        delta_positions, joint_Jacobs = [], []
        for leg_id, target_position in zip(pinocchio_leg_ids, target_positions):
            #curr_position = pinocchio_data.oMf[leg_id].translation                 # world frame
            curr_position = oMbase.actInv(pinocchio_data.oMf[leg_id]).translation   # base frame
            delta_positions.append(curr_position - target_position)
            
            full_Jacob = pinocchio.computeFrameJacobian(pinocchio_model, pinocchio_data, joint_angle, leg_id, 
                                                        pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
            joint_Jacobs.append(full_Jacob[:, 6:])                                  # with only the leg Jacobian

        delta_position_full = np.concatenate(delta_positions)
        if np.linalg.norm(delta_position_full) < eps:
            break

        joint_Jacob_all = np.vstack(joint_Jacobs)
        joint_velocity = np.zeros(pinocchio_model.nv)
        joint_velocity[6:] = -joint_Jacob_all.T @ np.linalg.inv(
            joint_Jacob_all @ joint_Jacob_all.T + damping_factor*np.eye(joint_Jacob_all.shape[0])) @ delta_position_full
    
        joint_angle = pinocchio.integrate(pinocchio_model, joint_angle, joint_velocity*delta_time)
    return joint_angle
#############################################################################################################################




