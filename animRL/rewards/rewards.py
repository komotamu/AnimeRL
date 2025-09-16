from animRL.utils.math import *
import torch

"""
The data dictionary has information about the environment and the current state of the agent.
The data dictionary contains the following:
* 'motion_loader': MotionLoader object that contains the motion data. 
                   You can use this object to call functions from the MotionLoader class (located in animRL/dataloader/motion_loader.py).
* 'num_ee': Number of end effectors.
* 'num_dof': Number of degrees of freedom (same as number of actions).
* 'env_origins': The coordinates of the origin for each environment.
* 'root_states': The root states of the agent. The shape is (num_envs, 13). 
                 Each row contains: root global position (3), root global orientation expressed in quaternion (4),
                 root linear velocity (3), root angular velocity (3).
* 'base_quat': The global quaternion of the base (root) orientation. The shape is (num_envs, 4). This is same as root_states[:, 3:7].
* 'base_lin_vel': The linear velocity of the base (root) in local frame. The shape is (num_envs, 3).
* 'base_ang_vel': The angular velocity of the base (root) in local frame. The shape is (num_envs, 3).
* 'dof_pos': The joint positions of the agent. The shape is (num_envs, num_dof).
* 'dof_vel': The joint velocities of the agent. The shape is (num_envs, num_dof).
* 'ee_global': The global positions of the end effectors. The shape is (num_envs, num_ee, 3).
* 'ee_local': The local positions of the end effectors. The shape is (num_envs, num_ee, 3).
* 'joint_targets_rate': The rate of change of joint targets (from actions), normalized. The shape is (num_envs, 1).
* 'target_frames': The target frame from reference motion based on the current phase. The shape is (num_envs, frame_dim).
* 'reset_frames': The frame from reference motion that the corresponds to the phase at the beginning of the episode. The shape is (num_envs, frame_dim).
"""


class REWARDS:

    @staticmethod
    def reward_track_base_height(data, sigma, tolerance=0.0):
        motion_loader = data['motion_loader']
        target_frames = data['target_frames']
        reward = torch.zeros(0).to(data['root_states'].device)

        root_states = data['root_states']
        target_height = motion_loader.get_root_pos(target_frames)[:, 2]
        height_error = torch.abs(root_states[:, 2] - target_height)
        height_error *= height_error > tolerance
        reward = torch.exp(-torch.square(height_error / sigma))
        return reward

    @staticmethod
    def reward_track_base_orientation(data, sigma, tolerance=0.0):
        motion_loader = data['motion_loader']
        target_frames = data['target_frames']
        reward = torch.zeros(0).to(data['root_states'].device)

        # ----------- TODO 1.1: implement the reward
        root_states = data['root_states']
        current_quat = root_states[:, 3:7]  # Current orientation quaternion
        target_quat = motion_loader.get_root_rot(target_frames)  # Target orientation quaternion
        
        # Calculate quaternion difference and convert to angular error
        quat_difference = quat_diff(current_quat, target_quat)
        orientation_error = quat_to_angle(quat_difference)
        orientation_error *= orientation_error > tolerance
        reward = torch.exp(-torch.square(orientation_error / sigma))
        # ----------- End of implementation
        return reward

    @staticmethod
    def reward_track_joint_pos(data, sigma, tolerance=0.0):
        motion_loader = data['motion_loader']
        target_frames = data['target_frames']
        reward = torch.zeros(0).to(data['root_states'].device)

        # ----------- TODO 1.1: implement the reward
        dof_pos = data['dof_pos']  # Current joint positions
        target_joint_pos = motion_loader.get_joint_pose(target_frames)  # Target joint positions
        
        # Calculate joint position error
        joint_pos_error = torch.norm(dof_pos - target_joint_pos, dim=-1)
        joint_pos_error *= joint_pos_error > tolerance
        reward = torch.exp(-torch.square(joint_pos_error / sigma))
        # ----------- End of implementation
        return reward

    @staticmethod
    def reward_track_base_vel(data, sigma, tolerance=0.0):
        motion_loader = data['motion_loader']
        target_frames = data['target_frames']
        reward = torch.zeros(0).to(data['root_states'].device)

        # ----------- TODO 1.1: implement the reward
        base_lin_vel = data['base_lin_vel']  # Current linear velocity
        base_ang_vel = data['base_ang_vel']  # Current angular velocity
        target_lin_vel = motion_loader.get_linear_vel(target_frames)  # Target linear velocity
        target_ang_vel = motion_loader.get_angular_vel(target_frames)  # Target angular velocity
        
        # Calculate velocity errors
        lin_vel_error = torch.norm(base_lin_vel - target_lin_vel, dim=-1)
        ang_vel_error = torch.norm(base_ang_vel - target_ang_vel, dim=-1)
        total_vel_error = lin_vel_error + ang_vel_error
        
        total_vel_error *= total_vel_error > tolerance
        reward = torch.exp(-torch.square(total_vel_error / sigma))
        # ----------- End of implementation
        return reward

    @staticmethod
    def reward_track_ee_pos(data, sigma, tolerance=0.0):
        motion_loader = data['motion_loader']
        target_frames = data['target_frames']
        reward = torch.zeros(0).to(data['root_states'].device)

        # ----------- TODO 1.1: implement the reward
        ee_global = data['ee_global']  # Current end effector positions (global)
        target_ee_global = motion_loader.get_ee_global_from_frame(target_frames)  # Target end effector positions
        
        # Calculate end effector position error
        ee_pos_error = torch.norm(ee_global - target_ee_global, dim=-1)  # Error per end effector
        ee_pos_error_total = torch.mean(ee_pos_error, dim=-1)  # Average across all end effectors
        
        ee_pos_error_total *= ee_pos_error_total > tolerance
        reward = torch.exp(-torch.square(ee_pos_error_total / sigma))
        # ----------- End of implementation
        return reward

    @staticmethod
    def reward_joint_targets_rate(data, sigma, tolerance=0.0):
        reward = torch.zeros(0).to(data['root_states'].device)

        # ----------- TODO 1.1: implement the reward
        joint_targets_rate = data['joint_targets_rate']  # Rate of change of joint targets
        
        # Encourage smooth actions (low rate of change)
        action_smoothness_error = torch.abs(joint_targets_rate.squeeze())
        action_smoothness_error *= action_smoothness_error > tolerance
        reward = torch.exp(-torch.square(action_smoothness_error / sigma))
        # ----------- End of implementation
        return reward
