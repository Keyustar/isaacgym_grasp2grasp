import numpy as np
import isaacgym
import torch
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil

from datetime import datetime
# from tqdm import tqdm 
import os
from pathlib import Path 
import math
import time

class HandKinematicModelIsaacGym:
    def __init__(self,
                 print_freq=False,
                 hand_urdf='', 
                 base_link='base_link', 
                 n_hand_dof=22,
                 joint_names=''):
        self.print_freq = print_freq
        self.hand_urdf = hand_urdf
        self.base_link = base_link
        self.joint_names = joint_names
        self.n_hand_dof = n_hand_dof
        # initialize gym
        self.gym = gymapi.acquire_gym()
        print(dir(self.gym))
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 30
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False
        print("create sim")
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)



        # asset_root = "./assets"
        # #left_asset_path = "XL_Hand_urdf/urdf/XL_Hand_urdf_New.urdf"
        # right_asset_path = "XL_Hand_urdf/urdf/XL_Hand_urdf_New.urdf"
        asset_root = os.path.dirname(self.hand_urdf)
        asset_file = os.path.basename(self.hand_urdf)
        print("asset_root", asset_root)
        print("asset_file", asset_file)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        #asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        #left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        right_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        #获取仿真关节索引
        num_hand_actuators = self.gym.get_asset_dof_count(right_asset)
        print("num_hand_actuators:",num_hand_actuators)
        actuated_dof_names = self.gym.get_asset_dof_names(right_asset)
        print("actuated_dof_names:",actuated_dof_names)
        actuated_dof_indices = []
        for name in actuated_dof_names:
            dof_index = self.gym.find_asset_dof_index(right_asset, name)
            assert dof_index != -1, f"Could not find dof index for {name}"
            print(f"{name}\t->\t{dof_index}")
            actuated_dof_indices.append(dof_index)
        print("actuated_dof_indices:",actuated_dof_indices)


        self.user_idx_to_sim_idx = [actuated_dof_names.index(x) for x in self.joint_names]
        print("User-to-Sim Joint", self.user_idx_to_sim_idx)
        self.sim_idx_to_user_idx = [self.user_idx_to_sim_idx.index(i) for i in range(len(self.user_idx_to_sim_idx))]
        print("Sim-to-User Joint", self.sim_idx_to_user_idx)



        hand_dof_props = self.gym.get_asset_dof_properties(right_asset)
        print("hand_dof_props_lower:",hand_dof_props["lower"])
        print("hand_dof_props_upper:",hand_dof_props["upper"])
        # load joint range information
        self.hand_dof_lower_limits = np.array([hand_dof_props["lower"][i] for i in range(self.n_hand_dof)])
        self.hand_dof_upper_limits = np.array([hand_dof_props["upper"][i] for i in range(self.n_hand_dof)])
        # set up the env grid
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)


        # left_hand
        # pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        # pose.r = gymapi.Quat(0, 0, 0, 1)
        # self.left_handle = self.gym.create_actor(self.env, left_asset, pose, 'left', 1, 1)
        # self.gym.set_actor_dof_states(self.env, self.left_handle, np.zeros(self.dof, gymapi.DofState.dtype),
        #                               gymapi.STATE_ALL)
        # left_idx = self.gym.get_actor_index(self.env, self.left_handle, gymapi.DOMAIN_SIM)
        print("create right hand")
        # right_hand
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1)
        pose.r = gymapi.Quat(0, 0, 0, 1)

        try:
            self.right_handle = self.gym.create_actor(self.env, right_asset, pose, 'right', 0, 1)
            print("Successfully created right hand actor")
        except Exception as e:
            print(f"Failed to create actor: {e}")
            quit()
        dof_props = self.gym.get_actor_dof_properties(self.env, self.right_handle)
        for i in range(self.n_hand_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = 400.0  # 增加刚度
            dof_props['damping'][i] = 10.0     # 增加阻尼
        self.gym.set_actor_dof_properties(self.env, self.right_handle, dof_props)
        self.gym.set_actor_dof_states(self.env, self.right_handle, np.zeros(self.n_hand_dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        right_idx = self.gym.get_actor_index(self.env, self.right_handle, gymapi.DOMAIN_SIM)

        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)
        self.left_root_states = self.root_states[right_idx]
        self.right_root_states = self.root_states[right_idx]
        # create default viewer
        print("create viewer")
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def set_qpos(self, right_qpos):
        right_qpos = np.asarray(right_qpos, dtype=np.float32).reshape(-1)
        # if right_qpos.shape[0] != 20:
        #     if right_qpos.shape[0] < 20:
        #         right_qpos = np.concatenate([right_qpos, np.zeros(20 - right_qpos.shape[0], dtype=np.float32)], axis=0)
        #     else:
        #         right_qpos = right_qpos[:20]        
        right_qpos = self.convert_user_order_to_sim_order(right_qpos)        
        right_qpos = np.clip(right_qpos, self.hand_dof_lower_limits + 1e-7, self.hand_dof_upper_limits - 1e-7)
        right_states = np.zeros(self.n_hand_dof, dtype=gymapi.DofState.dtype)
        right_states['pos'] = right_qpos
        print(right_states)
        self.gym.set_actor_dof_states(self.env, self.right_handle, right_states, gymapi.STATE_POS)

    # def set_qpos_target(self, qpos):
    #     qpos = self.convert_user_order_to_sim_order(qpos)
    #     qpos = np.clip(qpos, self.hand_dof_lower_limits + 1e-7, self.hand_dof_upper_limits - 1e-7)

    #     self.qpos_target = qpos
    #     self.gym.set_actor_dof_position_targets(self.env, self.right_handle, self.qpos_target.astype(np.float32))
    def set_qpos_target(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        # if qpos.shape[0] != 20:
        #     if qpos.shape[0] < 20:
        #         qpos = np.concatenate([qpos, np.zeros(20 - qpos.shape[0], dtype=np.float32)], axis=0)
        #     else:
        #         qpos = qpos[:20]

        qpos = self.convert_user_order_to_sim_order(qpos)
        #print(qpos)
        #qpos = np.clip(qpos, self.hand_dof_lower_limits + 1e-7, self.hand_dof_upper_limits - 1e-7)

        self.qpos_target = qpos
        self.gym.set_actor_dof_position_targets(self.env, self.right_handle, self.qpos_target.astype(np.float32))
    def convert_user_order_to_sim_order(self, qpos):
        return qpos[self.sim_idx_to_user_idx]

    def step(self):

        if self.print_freq:
            start = time.time()


        # step the physics
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        #self.gym.render_all_camera_sensors(self.sim)
        #self.gym.refresh_actor_root_state_tensor(self.sim)


        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        if self.print_freq:
            end = time.time()
            print('Frequency:', 1 / (end - start))


    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    @staticmethod
    def build_from_config(**kwargs):
        '''
            Build a kinematic model from user config.
        '''
        urdf_path = "/home/cky/faive_gym_oss/assets/open_ai_assets_nowrist/hand/shadow_hand.xml"
        n_hand_dof = 22
        base_link = "robot0:hand mount"
        joint_order = [
        "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
        "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
        "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
        "robot0:LFJ4", "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1", "robot0:LFJ0",
        "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0"
    ]

        model = HandKinematicModelIsaacGym(hand_urdf=urdf_path,n_hand_dof=n_hand_dof, base_link=base_link, joint_names=joint_order)
        return model 

    def get_joint_limit(self):
        '''
            Get the hand joint limit.
        '''
        return self.hand_dof_lower_limits, self.hand_dof_upper_limits


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    # parser.add_argument('--hand', type=str, default='allegro')

    args = parser.parse_args()

    # Load Hand Model
    model = HandKinematicModelIsaacGym.build_from_config()
   
    dof_lower, dof_upper = model.get_joint_limit()

    steps = -1
    while True:
        steps += 1

        if steps % 10 == 0:    
            # targets = np.array([2, 1.0, 1.5, 1.5, 0.0, 0.0, 1.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5])
            # model.set_qpos_target(targets)            
            targets = np.random.uniform(0, 1, model.n_hand_dof) * (np.array(dof_upper) - np.array(dof_lower) - 1e-7) + np.array(dof_lower) + 1e-7
            print("targets:",targets)
        #     #model.set_qpos(targets)
            model.set_qpos_target(targets)
        model.step()
