# robot_hand.py
# Generic class for robot hand RL environments
# Copyright 2023 Soft Robotics Lab, ETH Zurich
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import (
    to_torch,
    quat_mul,
    quat_conjugate,
    tensor_clamp,
    scale,
    unscale,
    quat_apply,
    torch_rand_float,
    quat_from_angle_axis,
    quat_from_euler_xyz
)
import numpy as np
import os
import datetime
import torch
import transforms3d


from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import (
    quat_to_angle_axis,
)

"""
Define the IsaacGym environment for a somewhat generic robot hand
Different versions of the Faive Hand can be implemented by specifying the model in the config yaml file, as well as other similar robot hands
Despite this goal, currently it is somewhat hardcoded specifically to the Faive Hand- if some features are too specific to the Faive Hand, they can be moved to a child class

if the task or the robot deviates too much from what is implemented here, consider creating a child class
"""


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class OurHand(VecTask):
    hand_translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    hand_rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    joint_names = [
    'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
]
    def __init__(
        self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render
    ):
        """
        Args:
            cfg: Configuration object
            sim_params (gymapi.SimParams): Simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            sim_device (str): "cuda" or "cpu"
            headless: True if running headless
        """
        self.cfg = cfg
        self._parse_cfg(self.cfg)
        # overwrite config to have the correct number of observations for actor and critic
        self.cfg["env"]["numObservations"], self.cfg["env"]["numStates"] = self._prepare_observations()

        self.sim_device_id = sim_device

        super().__init__(config=cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)




        self.dt = self.sim_params.dt
        self.control_dt = self.control_freq_inv * self.dt  # dt for policy
        self.max_episode_length = np.ceil(cfg["env"]["episode_length_s"] / self.control_dt)
        
        if not self.headless:
            # set camera
            self.cam_pos = gymapi.Vec3(
                *self.cfg["visualization"]["camera_pos_start"]
            )
            self.cam_target = gymapi.Vec3(
                *self.cfg["visualization"]["camera_target_start"]
            )
            # set camera speed direction
            self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)
            if self.cfg["visualization"]["move_camera"]:
                self.cam_movement_per_step = \
                    gymapi.Vec3(*self.cfg["visualization"]["camera_movement_vector"])
        self._init_buffers()
        self._prepare_reward_function()
        if self.cfg["logging"]["rt_plt"]:
            self._prepare_logged_functions()

    def _init_buffers(self):
        """
        Initialize buffers (torch tensors) that will contain simulation states
        """
        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        # fetch the data from the sim
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # save as appropriately shaped torch tensors
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            -1, 13
        )
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
    
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
            self.num_envs, -1
        )

        self.transition_success_steps = torch.zeros(self.num_envs, device=self.device)  # 成功抓握时间步计数
        # self.current_grasp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 当前手势索引
        self.transition_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        #课程学习
        self.disturbance_force = torch.zeros((self.num_envs, 3), device=self.device)  # 存储当前施加的扰动力
        self.disturbance_duration = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # 扰动持续时间计数器
        self.disturbance_cooldown = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # 扰动冷却时间计数器
        self.disturbance_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # 扰动是否激活
        self.grasp_transition_phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # 课程学习阶段
        self.prev_grasp_transition_phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.phase0_stay_steps = torch.zeros(self.num_envs, device=self.device)
        self.phase1_stay_steps = torch.zeros(self.num_envs, device=self.device) 
        self.phase2_stay_steps = torch.zeros(self.num_envs, device=self.device)               
        # 0: 稳定抓握阶段，1: 过渡阶段，2: 目标姿态稳定阶段

        # next, define new member variables that make it easier to access the state tensors
        # if arrays are not used for indexing, the sliced tensors will be views of the original tensors, and thus their values will be automatically updated
        # Since it uses the first self.num_hand_dofs values of the dof state,
        # this code assumes that the robot hand is the first thing that is loaded into IsaacGym with create_actor().
        dof_state = self.dof_state.view(self.num_envs, -1, 2)
        print("重塑后的形状:", dof_state.shape)
        print(f"总DOF数: {dof_state.shape[1]}")
        #提取所有环境中的手部关节状态
        hand_dof_state = dof_state[:, :self.num_hand_dofs]
        self.hand_dof_pos = hand_dof_state[..., 0]
        self.hand_dof_vel = hand_dof_state[..., 1]
        self.prev_hand_dof_vel = torch.zeros_like(hand_dof_state[..., 1])

        #提取所有环境中目标手势的手部关节状态
        hand_goal_dof_state = dof_state[:, self.num_hand_dofs : 2 * self.num_hand_dofs]
        self.hand_goal_dof_pos = hand_goal_dof_state[..., 0]
        self.hand_goal_dof_vel = hand_goal_dof_state[..., 1]                
        print(f"dof_state shape!!!!!!!!!!!!!!!: {dof_state.shape}")
        print(f"hand_dof_pos shape!!!!!!!!!!!!!!!!: {self.hand_dof_pos.shape}")
        print(f"hand_dof_pos shape!!!!!!!!!!!!!!!!: {self.hand_goal_dof_pos.shape}")
        #提取所有环境中目标物体的自由度状态
        object_goal_dof_state = dof_state[:, 2 * self.num_hand_dofs + self.num_object_dofs:]
        assert object_goal_dof_state.shape[1] == 2 * self.num_object_dofs
        self.object_goal_dof_pos = object_goal_dof_state[..., 0]
        self.object_goal_dof_vel = object_goal_dof_state[..., 1]
        
        self.goal_pos = self.object_goal_states[:, 0:3]
        self.goal_rot = self.object_goal_states[:, 3:7]

        self.pose_sensor_state = self.rigid_body_states[:, self.pose_sensor_handles][
            :, :, 0:13
        ]
        # print("Sensor tensor pointer:", sensor_tensor)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(
            self.num_envs, -1
        )
        assert self.vec_sensor_tensor.shape[1] % 6 == 0  # sanity check

        num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        # current position control targets for each joint (joints with no actuators should be set to 0)
        self.cur_targets = torch.zeros(
            (self.num_envs, num_dofs), dtype=torch.float, device=self.device
        )
        self.prev_targets = torch.zeros(
            (self.num_envs, num_dofs), dtype=torch.float, device=self.device
        )

        self.x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        # add up the number of successes in each env (resets when the env is reset)
        self.successes = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        # 添加分离的成功缓冲区
        self.successes_joint = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.successes_pos = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.successes_rot = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # # 添加前一步的成功状态记录
        # self.successes_joint_pre = torch.zeros(
        #     self.num_envs, dtype=torch.bool, device=self.device
        # )
        # self.successes_pos_pre = torch.zeros(
        #     self.num_envs, dtype=torch.bool, device=self.device
        # )
        # self.successes_rot_pre = torch.zeros(
        #     self.num_envs, dtype=torch.bool, device=self.device
        # )
        # smoothed value keeping track of how many successes before drop / timeout in each env
        self.consecutive_successes = torch.zeros(
            1, dtype=torch.float, device=self.device
        )

        all_observation_names = self.cfg["observations"]["actor_observations"]
        if self.cfg["observations"]["asymmetric_observations"]:
            all_observation_names += self.cfg["observations"]["critic_observations"]
        all_observation_names = list(set(all_observation_names))  # remove duplicates

        # reserve space for previous observation buffer (for object pose and robot dof)
        len_obj_pose_buffer = self.obs_dims["obj_pose_history"]
        assert len_obj_pose_buffer % 7 == 0, \
            "obj_pose_buffer length must be a multiple of 7"
        assert len_obj_pose_buffer >=  7 * 2, \
            "obj_pose_buffer length must be equal to or greater than 14 to save more than one step of history"

        self.obj_pose_buffer = torch.zeros(
            (self.num_envs, len_obj_pose_buffer),
            dtype=torch.float,
            device=self.device,
        )
        # self.contact_steps = torch.zeros(
        #     self.num_envs, 
        #     device=self.device, 
        #     dtype=torch.float32      # 步数类型为浮点数
        # )#接触步数计时器
        print("!!!!!!!!!!!!self.num_actuated_dofs:",self.num_actuated_dofs)
        len_dof_pos_buffer = self.obs_dims["dof_pos_history"]
        assert len_dof_pos_buffer % self.num_actuated_dofs == 0, \
            "dof_pos_buffer length must be a multiple of the " + \
                    f"actuated dofs ({self.num_actuated_dofs})"
        assert len_dof_pos_buffer >= self.num_actuated_dofs * 2, \
            f"dof_pos_buffer length must be equal to or greater than\
            {self.num_actuated_dofs * 2} to save more than one step of history"

        self.dof_pos_buffer = torch.zeros(
            (self.num_envs, len_dof_pos_buffer),
            dtype=torch.float,
            device=self.device,
        )
      


        # joint and sensor readout recording buffers
        self.record_dof_poses = self.cfg["logging"]["record_dofs"]
        self.record_length = self.cfg["logging"]["record_length"]
        self.record_observations = self.cfg["logging"]["record_observations"]
        if self.record_dof_poses:
            self.dof_pose_recording = torch.zeros(
                (self.num_envs, self.record_length, self.num_actuated_dofs),
                dtype=torch.float,
                device=self.device
            )
        if self.record_observations:
            self.observation_recording = torch.zeros(
                (self.num_envs, self.record_length, self.obs_buf.shape[1]),
                dtype=torch.float,
                device=self.device
            )
        self.num_recorded_steps = 0
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        recording_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f'recordings')    
        if not os.path.exists(recording_dir):
            os.makedirs(recording_dir)
        self.recording_save_path = os.path.join(
            recording_dir,
            f'{timestamp_str}')

    def pre_physics_step(self, actions):
        """
        convert actions to commands applicable to the robot and set them
        """
        env_ids_to_reset = self.reset_buf.nonzero(as_tuple=False).flatten()
        goal_ids_to_reset = self.reset_goal_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids=env_ids_to_reset, goal_env_ids=goal_ids_to_reset)

        clip_actions = self.cfg["actions"]["clip_value"]
        actions = torch.clip(actions, -clip_actions, clip_actions)
        self.actions = actions.to(self.device)
        if self.cfg["env"]["use_relative_control"]:
            targets = (
                self.prev_targets[:, self.actuated_dof_indices]
                + self.cfg["env"]["relative_control_speed_scale"] * self.control_dt * self.actions
            )
        else:
            targets = scale(
                self.actions,
                self.actuated_dof_lower_limits,
                self.actuated_dof_upper_limits,
            )#将策略网络输出的标准化动作（通常范围 [-1, 1]）线性映射到​​关节的实际物理范围
        self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
            targets, self.actuated_dof_lower_limits, self.actuated_dof_upper_limits
        )#对映射后的目标值进行二次钳位，确保其严格处于关节的物理限制范围内。
        self.prev_targets[:] = self.cur_targets[:]
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets)
        )

    def post_physics_step(self):
        """
        处理物理仿真步骤后的逻辑
        """
        self.progress_buf += 1
        # 非常频繁的内存清理
        if self.progress_buf[0] % 100 == 0:  # 每50步清理一次
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # # 内存监控
        # if self.progress_buf[0] % 200 == 0:
        #     allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        #     reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        #     print(f"Step {self.progress_buf[0]}: GPU内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
            
        #     # 如果内存使用过高，强制清理
        #     if allocated > 10.0:  # 如果超过10GB
        #         print("GPU内存使用过高，强制清理...")
        #         torch.cuda.empty_cache()
        # 更新状态张量
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)


        # compute tensors for accessing specific parts of the state tensors
        # since it uses an array for indexing, it's not possible (afaik -Yasu) to make them reference the same memory and have them update automatically, like how self.hand_dof_pos is done
        # 更新状态张量
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.hand_pose = self.root_state_tensor[self.hand_indices, :7]
        self.hand_vel = self.root_state_tensor[self.hand_indices, 7:]

        # get pose sensor states
        # 计算各种状态信息
        self.pose_sensor_state[:] = self.rigid_body_states[:, self.pose_sensor_handles][
            :, :, 0:13
        ]

        # update the history buffers
         # 更新历史缓冲区
        self.obj_pose_buffer[:,:-7] = self.obj_pose_buffer[:,7:].clone()
        self.obj_pose_buffer[:,-7:] = self.object_pose.clone()
        self.obj_pose_buffer[:,-7:-4] -= self.object_init_states[:, :3]  # # 标准化位置
        # print("!!!!!!!!!!!!self.num_actuated_dofs",self.num_actuated_dofs)    
        self.dof_pos_buffer[:,:-self.num_actuated_dofs] = self.dof_pos_buffer[:,self.num_actuated_dofs:].clone()
        self.dof_pos_buffer[:,-self.num_actuated_dofs:] = unscale(
            self.hand_dof_pos[:, self.actuated_dof_indices],
            self.actuated_dof_lower_limits,
            self.actuated_dof_upper_limits,
        )

        # compute dof and object velocity numerically
        # this may be more useful in some cases where the velocity reported by isaacgym is not accurate (due to impulses applied for contact?)
        # https://forums.developer.nvidia.com/t/inaccuracy-of-dof-state-readings/197373
        # 计算速度和加速度
        current_hand_dof = self.dof_pos_buffer[:,-self.num_actuated_dofs:]
        previous_hand_dof = self.dof_pos_buffer[:,-self.num_actuated_dofs*2:-self.num_actuated_dofs]
        self.hand_dof_vel_numerical = (current_hand_dof - previous_hand_dof) / self.control_dt
        current_obj_pos = self.obj_pose_buffer[:,-7:-4]
        previous_obj_pos = self.obj_pose_buffer[:,-14:-7][:,-7:-4]
        self.object_linvel_numerical = (current_obj_pos - previous_obj_pos) / self.control_dt
        # compute object rotational velocity
        # 计算加速度
        current_quat = self.obj_pose_buffer[:,-4:]
        previous_quat = self.obj_pose_buffer[:,-14:-7][:,-4:]
        angle, axis = quat_to_angle_axis(quat_mul(current_quat, quat_conjugate(previous_quat)))
        self.object_angvel_numerical = angle.unsqueeze(1) * axis / self.control_dt

        # compute acceleration by taking finite difference of velocity
        self.dof_acceleration = (self.hand_dof_vel - self.prev_hand_dof_vel) / self.control_dt
        self.prev_hand_dof_vel[:] = self.hand_dof_vel.clone()

        # 继续处理
        self.check_termination()
        self.compute_reward()

        # if recording is activated, register dof poses/observations
        # 记录数据
        if self.record_dof_poses or self.record_observations:
            if self.num_recorded_steps <= self.record_length:
                self.record_step()

        # add rewards_dict to extras
        self.extras.update(self.rewards_dict)
        # log additional curriculum info
        self.extras["consecutive_successes"] = self.consecutive_successes.item()
        self.extras["average_rotvel_x"] = self.object_angvel_numerical[:,0].mean().item()
        self.extras["min_rotvel_x"] = self.object_angvel_numerical[:,0].min().item()
        self.extras["max_rotvel_x"] = self.object_angvel_numerical[:,0].max().item()
        self.extras["std_rotvel_x"] = self.object_angvel_numerical[:,0].std().item()
        self.extras["phase0_envs"] = torch.sum((self.grasp_transition_phase == 0)).item()
        self.extras["phase1_envs"] = torch.sum((self.grasp_transition_phase == 1)).item()
        self.extras["phase2_envs"] = torch.sum((self.grasp_transition_phase == 2)).item()
        self.extras["disturbed_envs"] = torch.sum(self.disturbance_active).item()
        self.extras["avg_disturbance"] = torch.mean(torch.norm(self.disturbance_force, dim=1)).item()        
        self.compute_observations()

        # visualize
        if not self.headless and self.cfg["env"]["enable_debug_viz"]:
            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                x, y, z = self.hand_pose[i, :3].cpu().numpy()
                self._draw_sphere(i, x, y, z)
                self._draw_frame_axes(i, self.goal_pos[i], self.goal_rot[i])
                self._draw_frame_axes(i, self.object_pos[i], self.object_rot[i])

        if not self.headless and self.cfg["env"]["enable_contact_viz"]:
            if not self.cfg["env"]["enable_debug_viz"]:
                # clear lines if it has not been cleared already
                self.gym.clear_lines(self.viewer)
            assert not self.cfg["sim"]["use_gpu_pipeline"], "contact visualization can be only done with CPU pipeline"
            assert self.cfg["sim"]["physx"]["contact_collection"] in [1, 2], "contact_collection must be set to 1 or 2"
            assert not self.cfg["task"]["randomize"], "contact visualization is not supported with randomization, since the code for applying DR seems to be hardcoded to use GPU somewhere"
            self.gym.draw_env_rigid_contacts(self.viewer, self.envs[0], gymapi.Vec3(1,0.2,0.2), 1, False)

        # if specified, update the camera position
        if self.cfg["visualization"]["move_camera"]:
            self.cam_pos += self.cam_movement_per_step
            self.cam_target += self.cam_movement_per_step
            self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)

        # update logger
        if self.cfg["logging"]["rt_plt"]:
            self.get_logs()

    def record_step(self):
        '''
        Records the dof and/or observation buffers and saves them to a .npy file.
        '''
        print("Recording!")
        if self.num_recorded_steps < self.record_length:
            if self.record_dof_poses:
                self.dof_pose_recording[:,self.num_recorded_steps, :] = self.dof_pos_buffer[:,-self.num_actuated_dofs:]
            if self.record_observations:
                self.observation_recording[:,self.num_recorded_steps, :] = self.obs_buf
        else:
            if self.record_dof_poses:
                np.save(self.recording_save_path + "_dof_poses.npy", self.dof_pose_recording.numpy(force=True))
                print('dof poses saved')
            if self.record_observations:
                np.save(self.recording_save_path + "_observation.npy", self.observation_recording.numpy(force=True))
            print("all recordings saved, exiting")
            exit()
        self.num_recorded_steps += 1

    def check_termination(self):
        """
        check termination conditions for each env and set the corresponding buffers
        """
        # self.successes_joint_pre = self.successes_joint.clone()
        # self.successes_pos_pre = self.successes_pos.clone()
        # self.successes_rot_pre = self.successes_rot.clone()
        # 计算物体旋转角与目标角度的差距
        quat_diff = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
        rot_diff = 2.0 * torch.asin(
            torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
        )
        # rot_dist =2 * torch.acos(torch.clamp(torch.abs(quat_diff[:, 0]), 0, 1))
        # 计算位置距离
        pos_diff = torch.norm(self.object_pos - self.goal_pos, dim=-1)
        joint_diff = torch.norm(self.hand_dof_pos[:, self.actuated_dof_indices] - self.hand_goal_dof_pos[:, self.actuated_dof_indices], p=2, dim=1)
        # 计算手部关节角与目标关节角的差距
        # joint_diff = (self.hand_dof_pos - self.hand_goal_dof_pos) / self.joint_range 
        # joint_diff_normalized = torch.norm(joint_diff, p=2, dim=-1)
        # current_angles = self.hand_dof_pos  # [batch_size, n_joints]
        # target_angles = self.hand_goal_dof_pos
        # # 计算角度差（考虑周期性，如-π和π实际相同）
        # angle_diff = torch.atan2(
        #     torch.sin(current_angles - target_angles),
        #     torch.cos(current_angles - target_angles)
        # )  # 范围[-π, π]
        # # 归一化到[-1,1]（可选）
        # joint_normalized_diff = torch.abs(angle_diff) / torch.pi
        # joint_mean_diff = torch.norm(joint_normalized_diff, p=2, dim=1)
        # joint_mean_diff = self.calculate_joint_distance_pytorch_batch(self.hand_dof_pos, self.hand_goal_dof_pos)


        rot_success = rot_diff < self.cfg["rewards"]["obj_rot_tolerance"]
        joint_success = joint_diff < self.cfg["rewards"]["hand_success_tolerance"]  
        pos_success = pos_diff < self.cfg["rewards"]["obj_pos_tolerance"]

        # 更新分离的成功状态
        self.successes_joint = self.successes_joint | joint_success
        self.successes_pos = self.successes_pos | pos_success
        self.successes_rot = self.successes_rot | rot_success

        # 完全成功条件：所有三项都成功
        self.success_buf = rot_success & joint_success & pos_success
        #self.success_buf = rot_success    
        # 记录调试信息
        self.extras["joint_diff"] = joint_diff.mean().item()
        self.extras["pos_diff"] = pos_diff.mean().item()
        self.extras["rot_diff"] = rot_diff.mean().item()
        self.extras["joint_success_rate"] = joint_success.float().mean().item()
        self.extras["pos_success_rate"] = pos_success.float().mean().item()
        self.extras["rot_success_rate"] = rot_success.float().mean().item()
        self.extras["complete_success_rate"] = self.success_buf.float().mean().item()
        
        # 记录累积成功状态
        self.extras["joint_ever_success_rate"] = self.successes_joint.float().mean().item()
        self.extras["pos_ever_success_rate"] = self.successes_pos.float().mean().item()
        self.extras["rot_ever_success_rate"] = self.successes_rot.float().mean().item()
    

        # successful_env_indices = torch.nonzero(self.success_buf).flatten()
        # if successful_env_indices.numel() > 0:
        #     print(f"环境成功索引: {successful_env_indices.tolist()}")        
        # for i in torch.nonzero(self.success_buf).flatten():
        #     #更新当前和目标抓取姿态索引
        #     self.current_grasp_idx[i] = self.target_grasp_idx[i].clone()
        #     #选取新的目标抓取姿态
        #     new_goal_index = self.select_goal_by_nn(self.current_grasp_idx[i])            
        #     self.target_grasp_idx[i] = new_goal_index
        #     #加载当前抓取姿态
        #     hand_init_pose = self.load_grasp_poses(self.current_grasp_idx[i])  

        #     #加载目标抓取姿态
        #     hand_goal_pose = self.load_grasp_poses(self.target_grasp_idx[i])
        #     _, _, _, goal_hand_joints_state, _ = hand_goal_pose
 
        self.successes += self.success_buf
        # envs where the cube was dropped
        self.dropped_buf = pos_diff > self.cfg["rewards"]["fall_dist_threshold"]
        # the parent class sets the self.timeout_buf in the step() function so use a local variable here
        # 超时条件
        timeout_buf = self.progress_buf > self.max_episode_length - 1
        # 重置条件
        self.reset_goal_buf = timeout_buf | self.success_buf# | self.dropped_buf
        # for the successful envs, the robot does not need to be reset
        self.reset_buf = timeout_buf | self.dropped_buf# | self.success_buf

        # 计算并平滑更新连续成功率指标
        finished_consecutive_successes = torch.sum(self.reset_buf * self.successes)
        num_successes = torch.sum(self.reset_buf)
        av_factor = 0.1  # smoothing factor
        self.consecutive_successes = torch.where(num_successes > 0, av_factor * finished_consecutive_successes / num_successes + (1.0 - av_factor) * self.consecutive_successes, self.consecutive_successes)
        self.successes[self.reset_buf] = 0
        self.successes_joint[self.reset_buf] = False
        self.successes_pos[self.reset_buf] = False
        self.successes_rot[self.reset_buf] = False
        # self.successes_joint_pre[self.reset_buf] = False
        # self.successes_pos_pre[self.reset_buf] = False
        # self.successes_rot_pre[self.reset_buf] = False        

    def reset_current_grasp_pose(self, env_ids):
        """向量化版本（更高效）"""
        # 调用批量加载函数
        hand_pos, hand_rot, obj_pos, obj_rot, hand_joints = self.load_grasp_poses_batch(
            self.current_grasp_idx[env_ids]
        )
        
        # 批量处理手部关节状态（直接使用返回的张量）
        self.hand_dof_init_pos[env_ids, 2:] = hand_joints  # [B,N] 直接赋值
        
        # 批量处理物体状态（构建13维状态向量）
        object_states_batch = torch.zeros((len(env_ids), 13), device=self.device)
        
        # 填充位置和旋转（四元数）
        object_states_batch[:, :3] = obj_pos  # x,y,z
        object_states_batch[:, 3:7] = obj_rot  # quat (w,x,y,z)
        # 线速度和角速度保持为0（第7-12维）
        
        self.object_init_states[env_ids] = object_states_batch
        
        # 如果需要保留手部位姿信息，可以添加：
        # self.hand_init_pos[env_ids] = hand_pos
        # self.hand_init_rot[env_ids] = hand_rot    

    # 向量化版本（更高效）
    def reset_goal_grasp_pose(self, goal_env_ids):
        """重置目标抓取位姿（向量化版本）"""
        #更新当前和目标抓取姿态索引
        self.current_grasp_idx[goal_env_ids] = self.target_grasp_idx[goal_env_ids].clone()
        #选取新的目标抓取姿态
        new_goal_indices = self.select_goal_by_nn_batch_gpu_parallel(self.current_grasp_idx[goal_env_ids])
        # new_goal_indices = torch.randint(
        #     0, self.grasp_database_size, 
        #     (len(goal_env_ids),), 
        #     device=self.device
        # )                
        # 确保不重复
        # same_mask = new_goal_indices == self.current_grasp_idx[goal_env_ids]
        # if same_mask.any():
        #     num_same = same_mask.sum()
        #     replacement_indices = torch.randint(
        #         0, self.grasp_database_size, 
        #         (num_same,), 
        #         device=self.device
        #     )
        #     new_goal_indices[same_mask] = replacement_indices            
        self.target_grasp_idx[goal_env_ids] = new_goal_indices

        # 调用批量加载函数
        hand_pos, hand_rot, goal_obj_pos, goal_obj_rot, hand_joints = self.load_grasp_poses_batch(
            self.target_grasp_idx[goal_env_ids]
        )
        
        # 批量处理手部关节状态（直接使用返回的张量）
        self.hand_goal_dof_init_pos[goal_env_ids, 2:] = hand_joints  # [B,N] 直接赋值
        
        # 批量处理物体目标状态（构建13维状态向量）
        goal_object_states = torch.zeros((len(goal_env_ids), 13), device=self.device)
        
        # 填充位置和旋转（四元数）
        goal_object_states[:, :3] = goal_obj_pos  # x,y,z
        goal_object_states[:, 3:7] = goal_obj_rot  # quat (w,x,y,z)
        # 线速度和角速度保持为0（第7-12维）
        
        self.object_goal_init_states[goal_env_ids] = goal_object_states
        
        # 如果需要保留手部目标位姿信息，可以添加：
        # self.hand_goal_pos[goal_env_ids] = hand_pos
        # self.hand_goal_rot[goal_env_ids] = hand_rot 




    def reset_goal_states(self, env_ids):
        """
        重置指定环境ID=env_ids的目标状态
        并设置self.resetted_visual_goal_states()用于设置IsaacGym中目标对象的可视化状态.
        可视化目标状态与self.goal_states分开存储,这样可以将可视化目标显示在与实际目标位置不同的地方
        此实现适用于手内重定向任务，如果重置流程不同，请在您的类中重写此方法
        """
        # 1. 设置实际目标状态（固定值）
        self.object_goal_states[env_ids] = self.object_goal_init_states[env_ids].clone()
        self.hand_goal_states[env_ids] = self.hand_goal_init_states[env_ids].clone()
        #关节角度和速度
        self.hand_goal_dof_pos[env_ids] = self.hand_goal_dof_init_pos[env_ids].clone()
        self.hand_goal_dof_vel[env_ids] = torch.zeros_like(self.hand_goal_dof_vel[env_ids])
        self.prev_targets[env_ids, self.num_hand_dofs:2 * self.num_hand_dofs] = self.hand_goal_dof_init_pos[env_ids].clone()
        self.cur_targets[env_ids, self.num_hand_dofs:2 * self.num_hand_dofs] = self.hand_goal_dof_init_pos[env_ids].clone()
        


        # 2. 设置可视化目标状态（与实际目标相同，或固定偏移）
        self.resetted_visual_goal_object_states = self.object_goal_states[env_ids].clone()
        self.resetted_visual_goal_hand_states = self.hand_goal_states[env_ids].clone()
        goal_visual_displacement = [0, 0, 0.5]
        # the goal object within the rendered scene will be displaced by this amount from the actual goal
        self.resetted_visual_goal_object_states[:, 0:3] += torch.tensor(goal_visual_displacement, 
                                                                        device=self.device)
        self.resetted_visual_goal_hand_states[:, 0:3] += torch.tensor(goal_visual_displacement, 
                                                                    device=self.device)        


    
    def reset_object_states(self, env_ids):
        """
        reset the object states to the initial states for env_ids by setting self.resetted_object_states (which has shape (len(env_ids), 13))
        this implementation is for the in-hand reorientation task, but override it in your class if the reset procedure is different
        """
        # rand_floats = torch_rand_float(
        #     -1.0, 1.0, (len(env_ids), 5), self.device
        # )
        # reset object position
        self.resetted_object_states = self.object_init_states[env_ids].clone()
        # self.resetted_object_states[:, 0:3] += (
        #     rand_floats[:, 0:3] * self.cfg["reset_noise"]["object_pos"]
        # )
        # # reset object rotation
        # self.resetted_object_states[:, 3:7] = randomize_rotation(
        #     rand_floats[:, 3],
        #     rand_floats[:, 4],
        #     self.x_unit_tensor[env_ids],
        #     self.y_unit_tensor[env_ids],
        # )

    def custom_reset(self):
        """
        if you want to set up your own specific code to override the pose of objects, do so here
        (e.g. keep the object still in the air for the first few moments to help the hand grab it)
        The function should modify self.root_state_tensor and return a tensor of indices of the objects whose status should be reset.
        Then the reset_idx function will call gym.set_actor_root_state_tensor_indexed() to actually reset the objects in IsaacGym.
        """
        return torch.zeros(0, device=torch.device(self.device))

    def reset_idx(self, env_ids, goal_env_ids):
        """
        Reset the envs (the robot dofs and the object pose) in env_ids and
        the goals in goal_env_ids. The former forcibly resets all the dofs
        of the hand, which isn't ideal for joints constrained by tendons
        #TODO ... fix it if it becomes a problem
        """
        if self.cfg["task"]["randomize"]:
            self.apply_randomizations(self.cfg["task"]["randomization_params"])

        # keep track of which indices of the root state tensor should be reset at the end of this function
        reset_indices = torch.zeros(0, device=torch.device(self.device)).to(torch.int32)
        reset_dof_indices = torch.zeros(0, device=torch.device(self.device)).to(torch.int32)
        prev_targets_indices = torch.zeros(0, device=torch.device(self.device)).to(torch.int32)
        # handle any custom reset procedures and add the indices of those objects to reset_indices
        custom_reset_indices = self.custom_reset()
        reset_indices = torch.cat(
            (reset_indices, custom_reset_indices.to(torch.int32))
        )
 
        #重置目标环境状态
        if len(goal_env_ids) > 0:
            # overwrite self.goal_states in this function
            self.reset_goal_grasp_pose(goal_env_ids)  
            self.reset_goal_states(goal_env_ids) 
            # set the goal states in the sim
            self.root_state_tensor[self.goal_object_indices[goal_env_ids]] = self.resetted_visual_goal_object_states
            self.root_state_tensor[self.goal_hand_indices[goal_env_ids]] = self.resetted_visual_goal_hand_states
            goal_hand_indices = self.goal_hand_indices[goal_env_ids].to(torch.int32)
            goal_object_indices = self.goal_object_indices[goal_env_ids].to(torch.int32)                             
            reset_indices = torch.cat(
                (reset_indices, goal_hand_indices)
            ).to(torch.int32)
            reset_indices = torch.cat(
                (reset_indices, goal_object_indices)
            ).to(torch.int32)              
                 
            #处理目标手势关节角 
            if self.num_object_dofs > 0:                
                reset_dof_indices = torch.cat(
                    (reset_dof_indices, goal_hand_indices)
                ).to(torch.int32)
                reset_dof_indices = torch.cat(
                    (reset_dof_indices, goal_object_indices)
                ).to(torch.int32)                
            else:
                # do not set the object dofs if there are no object dofs (will cause an error otherwise)
                reset_dof_indices = torch.cat(
                    (reset_dof_indices, goal_hand_indices)
                ).to(torch.int32)

            prev_targets_indices =  torch.cat(
                    (prev_targets_indices, goal_hand_indices)
                ).to(torch.int32)

         

        if len(env_ids) > 0:
            self.reset_current_grasp_pose(env_ids) 
            # draw rand floats
            rand_floats = torch_rand_float(
                -1.0, 1.0, (len(env_ids), self.num_hand_dofs * 2), self.device
            )
            # self.reset_cur_states(env_ids)
            # hand_dof_pos = self.hand_cur_states[env_ids]
            #reset hand state
            hand_dof_range = self.hand_dof_upper_limits - self.hand_dof_lower_limits            
            hand_dof_pos = self.hand_dof_init_pos[env_ids]
            hand_dof_vel = self.hand_dof_default_vel
            # hand_dof_vel = (
            #     self.hand_dof_default_vel
            #     + rand_floats[:, self.num_hand_dofs : self.num_hand_dofs * 2]
            #     * self.cfg["reset_noise"]["dof_vel"]
            # )
            self.hand_dof_pos[env_ids] = hand_dof_pos
            self.hand_dof_vel[env_ids] = hand_dof_vel
            self.prev_targets[env_ids, :self.num_hand_dofs] = hand_dof_pos
            self.cur_targets[env_ids, :self.num_hand_dofs] = hand_dof_pos
            #目标手势保持
            # self.hand_goal_dof_pos[env_ids] = self.hand_goal_dof_default_pos.clone()
            # self.hand_goal_dof_vel[env_ids] = self.hand_goal_dof_default_vel.clone()
            # self.prev_targets[env_ids, self.num_hand_dofs:2 * self.num_hand_dofs] = self.hand_goal_dof_default_pos.clone()
            # self.cur_targets[env_ids, self.num_hand_dofs:2 * self.num_hand_dofs] = self.hand_goal_dof_default_pos.clone()            
            # reset object dof state (just set them to zero for now)
            self.object_goal_dof_pos[env_ids] = 0
            self.object_goal_dof_vel[env_ids] = 0

            hand_indices = self.hand_indices[env_ids].to(torch.int32)
            object_indices = self.object_indices[env_ids].to(torch.int32)
            # goal_hand_indices = self.goal_hand_indices[goal_env_ids].to(torch.int32)
            # prev_targets_indices =  torch.cat(
            #         (hand_indices, goal_hand_indices)
            #     ).to(torch.int32)
            # goal_indices = self.goal_object_indices[env_ids].to(torch.int32)
            if self.num_object_dofs > 0:                
                reset_dof_indices = torch.cat(
                    (reset_dof_indices, hand_indices)
                ).to(torch.int32)
                reset_dof_indices = torch.cat(
                    (reset_dof_indices, object_indices)
                ).to(torch.int32)                
            else:
                # do not set the object dofs if there are no object dofs (will cause an error otherwise)
                reset_dof_indices = torch.cat(
                    (reset_dof_indices, hand_indices)
                ).to(torch.int32)           
            # set the dof targets in the sim
            prev_targets_indices =  torch.cat(
                    (prev_targets_indices, hand_indices)
                ).to(torch.int32)

            # if object is fixed to the base, don't change its pose
            if not self.cfg["env"]["object_fix_base"]:
                # set self.resetted_object_states in this function
                self.reset_object_states(env_ids)
                # set the object state in the sim
                self.root_state_tensor[self.object_indices[env_ids]] = self.resetted_object_states

                reset_indices = torch.cat(
                    (reset_indices, self.object_indices[env_ids].to(torch.int32))
                )
            if not self.cfg["env"]["hand_fix_base"]:
                # don't implement reset randomization for now...
                self.root_state_tensor[self.hand_indices[env_ids]] = self.hand_init_states[env_ids].clone()
                reset_indices = torch.cat(
                    (reset_indices, self.hand_indices[env_ids].to(torch.int32))
                )


            # reset buffers
            self.progress_buf[env_ids] = 0 #重置环境计时器
            # self.transition_success_steps[env_ids] = torch.zeros_like(self.transition_success_steps[env_ids])
            # self.grasp_transition_phase[env_ids] = 0  # 重置为阶段0
            # self.disturbance_active[env_ids] = False
            # self.disturbance_force[env_ids] = 0
            # self.disturbance_duration[env_ids] = 0
            # self.disturbance_cooldown[env_ids] = torch.randint(
            #     30, 100, (len(env_ids),), device=self.device)

        if len(reset_indices) > 0:
            # apparently this can only be called once per step?
            # will return False if command fails
            assert self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(reset_indices),
                len(reset_indices),
            )
        #重置关节状态
        if len(prev_targets_indices) > 0:
            assert self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.prev_targets),
                gymtorch.unwrap_tensor(prev_targets_indices),
                len(prev_targets_indices),
            )
            # print("!!!!!!!!reset_dof_indices",reset_dof_indices)    
            # set the dof states in the sim
        if len(reset_dof_indices) > 0:
            assert self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(reset_dof_indices),
                len(reset_dof_indices),
            )            

    def reset(self):
        """Reset all robots and goals"""
        # all_envs = torch.arange(self.num_envs, device=self.device)
        # self.reset_idx(env_ids=all_envs, goal_env_ids=all_envs)
        obs, _, _, _ = self.step(
            torch.zeros(
                self.num_envs, self.num_actions, device=self.device, requires_grad=False
            )
        )
        return obs

    def _prepare_reward_function(self):
        """
        Prepare a list of reward functons, which will be called to compute the total reward
        looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are the nonzero entries in self.cfg["rewards"]["scales"]
        """
        # prepare list of reward functions
        self.reward_functions = []
        self.reward_names = []
        for name, _ in self.reward_scales.items():
            self.reward_names.append(name)
            func_name = "_reward_" + name
            # find member function with name func_name
            try:
                self.reward_functions.append(getattr(self, func_name))
            except AttributeError:
                raise AttributeError(
                    f"Reward function {func_name} not found, remove reward {name} or implement member function {func_name}"
                )
        # init dict for logging each reward value (averaged across all environments) separately
        self.rewards_dict = {}

    def _prepare_observations(self):
        """
        Prepare a list of observation functons, which will be called to compute the full observation
        looks for self._observation_<OBSERVATION_NAME>, where <OBSERVATION_NAME> are the entries defined
        in actor_observations (and critic_observations, if asymmetric_observations is set to True)
        returns the observation dimension for the actor and critic (latter is 0 if asymmetric_observations is False)
        """

        def collect_observation_functions_compute_dim(obs_names):
            """
            create a list of the observation functions for the list obs_names,
            and also compute the total dimension of that set of observations
            """
            obs_functions = []
            obs_dim = 0
            for obs_name in obs_names:
                try:
                    obs_dim += self.obs_dims[obs_name]
                except KeyError:
                    raise KeyError(f"could not find obs_dims for observation {obs_name}, check config file")
                func_name = "_observation_" + obs_name
                # find member function with name func_name
                try:
                    obs_functions.append(getattr(self, func_name))
                except AttributeError:
                    raise AttributeError(
                        f"Observation function {func_name} not found, remove observation {obs_name} or implement member function {func_name}"
                    )
            return obs_functions, obs_dim

        self.actor_obs_functions, actor_obs_dim = collect_observation_functions_compute_dim(
            self.cfg["observations"]["actor_observations"]
        )
        if self.cfg["observations"]["asymmetric_observations"]:
            self.critic_obs_functions, critic_obs_dim = collect_observation_functions_compute_dim(
                self.cfg["observations"]["critic_observations"]
            )
        else:
            critic_obs_dim = 0
        return actor_obs_dim, critic_obs_dim

    def _prepare_logged_functions(self):
        """
        Prepare a list of observation, reward and custom functions, which will be called to
        compute the plots of the visualization. Plot a canvas on which
        """
        # add a logger for online logging and plotting of states
        # TODO: this has been just ported from faive_gym, make it fully work with faive-isaac
        from isaacgymenvs.utils.logger import Logger
        self.logger = Logger(dt = self.control_dt, 
            buf_len_s = self.cfg["logging"]["buf_len_s"],
            rows = self.cfg["logging"]["num_rows"],
            cols = self.cfg["logging"]["num_cols"],)
        
        self.logger.measurement_names = self.cfg["logging"]["measurements"]
        self.logger.measurement_units = self.cfg["logging"]["units"]
        self.logging_functions = []
        self.logging_indices = []
        for name in self.cfg["logging"]["measurements"]:
            try:
                if "dof" in name:
                    func_name = ("_").join(name.split("_")[:-2])
                    dof_name = ("_").join(name.split("_")[-2:])
                    self.logging_indices.append(self.logger.dof_names.index(dof_name))
                    if "pos" in name:
                        self.logger.num_lines_per_subplot.append(2)
                    else:
                        self.logger.num_lines_per_subplot.append(1)
                elif "fingertip" in name:
                    func_name = ("_").join(name.split("_")[:-1])
                    finger = name.split("_")[-1]
                    self.logging_indices.append(self.logger.finger_names.index(finger))
                    obs_name = ("_").join(name.split("_")[2:-1])
                    if obs_name.endswith("stat"):
                        obs_name = obs_name[:-5]
                    self.logger.num_lines_per_subplot.append(
                            getattr(self.cfg["observations"]["obs_dims"], obs_name)//5)
                else:
                    if "reward" in name:
                        self.logger.num_lines_per_subplot.append(1)
                    if "observation" in name:
                        obs_name = ("_").join(name.split("_")[2:])
                        if obs_name.endswith("stat"):
                            obs_name = obs_name[:-5]
                        self.logger.num_lines_per_subplot.append(
                            self.obs_dims[obs_name])
                    self.logging_indices.append(None)
                    func_name = name
                # check for std/mean
                if func_name.endswith("_stat"):
                    func_name = func_name[:-5]
                    self.logger.num_lines_per_subplot[-1] *= 2
                self.logging_functions.append(getattr(self, func_name))
            except AttributeError:
                raise AttributeError(
                    f"Logging function {func_name} not found, remove measurement {name} or implement member function {func_name}"
                )
        # start plot process
        self.logger.plot_process.start()
    
    def _get_observation_to_log(self, obs_function, env_idx, obs_name, num_lines, obs_idx = None):
        """
        Calls the observation function and returns the observation to log
        """
        #print("Getting obs ", obs_name, " for env ", env_idx, " with idx ", obs_idx, " and num_lines ", num_lines)
        obs_tensor = obs_function()
        if "_stat" in obs_name:
                #print(obs_tensor.shape)
                if obs_idx is None:
                    if num_lines == 2:
                        return [torch.mean(obs_tensor).item(), torch.std(obs_tensor).item()]
                    else:
                        values = []
                        for i in range(num_lines//2):
                            values += [torch.mean(obs_tensor, dim=0)[i].item(), 
                                torch.std(obs_tensor, dim=0)[i].item()]
                        return values
                else:
                    if "pos" in obs_name and "dof" in obs_name:
                        pos_mean = torch.mean(obs_tensor, dim=0)[obs_idx].item()
                        target_mean = torch.mean(self.cur_targets, dim=0)[obs_idx].item()
                        pos_std = torch.std(obs_tensor, dim=0)[obs_idx].item()
                        target_std = torch.std(self.cur_targets, dim=0)[obs_idx].item()
                        return [pos_mean, target_mean, pos_std, target_std]
                    elif "fingertip" in obs_name:
                        values = []
                        for i in range(num_lines//2):
                            values += [
                                torch.mean(obs_tensor, dim=0)[obs_idx*5+i].item(),
                                torch.std(obs_tensor, dim=0)[obs_idx*5+i].item()
                            ]
                        return values
                    elif "proxim" in obs_name:
                        values = []
                        for i in range(num_lines//2):
                            values += [
                                torch.mean(obs_tensor, dim=0)[obs_idx*5+i].item(),
                                torch.std(obs_tensor, dim=0)[obs_idx*5+i].item()
                            ]
                        return values
                    else:
                        return [
                            torch.mean(obs_tensor, dim=0)[obs_idx].item(),
                            torch.std(obs_tensor, dim=1)[obs_idx].item()
                        ]
        else:
            if obs_idx is None:
                if num_lines == 1:
                    return obs_tensor[env_idx].item()
                else:
                    return [obs_tensor[env_idx,i].item() for i in range(num_lines)]
            else:
                if "pos" in obs_name and "dof" in obs_name:
                    pos = self.hand_dof_pos[:, self.actuated_dof_indices][env_idx, obs_idx].item()
                    target = self.cur_targets[:, self.actuated_dof_indices][env_idx, obs_idx].item()
                    return [pos, target]
                elif "fingertip" in obs_name:
                    #print("Taking idx ", obs_idx*5, " to ", obs_idx*5+4, " from fingertip obs tensor")
                    return [obs_tensor[env_idx,obs_idx*5+i].item() for i in range(num_lines)]
                elif "proxim" in obs_name:
                    
                    return [obs_tensor[env_idx,obs_idx*5+i].item() for i in range(num_lines)]
                else:
                    return obs_tensor[env_idx, obs_idx].item()

    def compute_reward(self):
        """
        Calls each reward function which has a non-zero scale (processed in self._prepare_reward_function)
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0
        for reward_name, reward_func in zip(self.reward_names, self.reward_functions):
            if self.reward_scales[reward_name] == 0:
                continue  # ignore zero-scaled rewards
            reward = reward_func() * self.reward_scales[reward_name]
            self.rew_buf += reward
            self.rewards_dict[f"rew_{reward_name}"] = reward.mean()

    def _fill_obs(self, obs_tensor, obs_names, obs_functions):
        """
        convenience function fill up the observation tensor
        obs_tensor: tensor that will be filled with the observations
        obs_names: list of observation names
        obs_functions: list of observation functions
        """
        obs_start = 0
        for obs_name, obs_func in zip(obs_names, obs_functions):
            obs_dim = self.obs_dims[obs_name]
            obs_end = obs_start + obs_dim
            obs = obs_func()
            assert (
                obs_dim == obs.shape[1]
            ), f"set correct observation dimension for [{obs_name}] in cfg"
            scale = self.obs_scales.get(obs_name, 1.0)
            assert scale != 0, f"set nonzero observation scale for [{obs_name}] in cfg"
            obs_tensor[:, obs_start:obs_end] = obs * scale
            obs_start = obs_end

    def compute_observations(self):
        """
        updates the observation buffer with the current observations
        """
        self._fill_obs(self.obs_buf, self.cfg["observations"]["actor_observations"], self.actor_obs_functions)
        if self.cfg["observations"]["asymmetric_observations"]:
            self._fill_obs(self.states_buf, self.cfg["observations"]["critic_observations"], self.critic_obs_functions)
        if self.cfg["observations"]["clip"]:
            clip_obs = self.cfg["observations"]["clip_value"]
            self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
            self.states_buf = torch.clip(self.states_buf, -clip_obs, clip_obs)

    def get_logs(self):
        """
        Captures variables from the current environment to the logger,
        which plots the real time on the screen.
        """
        for log_name, log_func, num_lines, obs_idx in zip(
            self.cfg["logging"]["measurements"],
            self.logging_functions,
            self.logger.num_lines_per_subplot,
            self.logging_indices
        ):
            self.logger.log_state(
                key = log_name,
                value=self._get_observation_to_log(log_func,
                    self.logger.logged_env_idx,
                    log_name,
                    num_lines,
                    obs_idx)
            )
        self.logger._state_send()

    def create_sim(self):
        """
        create the simulation environment (called within the super class's __init__)
        """
        self.sim = self.gym.create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()

        if self.cfg["task"]["randomize"]:
            # apply randomization once before first sim step
            self.apply_randomizations(self.cfg["task"]["randomization_params"])

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        env_lower = gymapi.Vec3(
            -self.cfg["env"]["env_spacing"], -self.cfg["env"]["env_spacing"], 0.0
        )
        env_upper = gymapi.Vec3(self.cfg["env"]["env_spacing"], self.cfg["env"]["env_spacing"], 0.0)
        num_per_row = int(np.sqrt(self.num_envs))

        asset_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")
        )
        hand_asset_file = os.path.normpath(self.cfg["asset"]["model_file"])
        asset_files_dict = {
            "bottle":"urdf/sem-Bottle-437678d4bc6be981c8724d5673a063a6/coacd/coacd.urdf",
            "block": "urdf/cube_multicolor.urdf",
            "sphere": "urdf/sphere.urdf",
            "pyramid": "objects_dext_manip/pyramid.xml",
            "prism": "objects_dext_manip/trig_prism.xml",
            "hex_prism": "objects_dext_manip/hex_prism.xml",
            "flat_pyr": "objects_dext_manip/flat_pyr.xml",
            "octahedron": "objects_dext_manip/octahedron.xml",
            "tetrahedron": "objects_dext_manip/tetrahedron.xml",
            "pentaprism": "objects_dext_manip/pentaprism.urdf",
            "dodecahedron": "objects_dext_manip/dodecahedron.xml",
            "stell_dodeca": "objects_dext_manip/stell_dodeca.urdf",
            "stairs": "objects_dext_manip/stairs.urdf",
            "block_pyr": "objects_dext_manip/block_pyr.urdf",
            "simple_book": "mjcf/simple_book.xml"
            
        }
        for i in range(len(self.cfg["env"]["object_type"])):
            try:
                os.path.normpath(
                    asset_files_dict[self.cfg["env"]["object_type"][i]]
                )
            except KeyError:
                raise ValueError(
                    f'Invalid object type: {self.cfg["env"]["object_type"][i]}, must be one of {asset_files_dict.keys()}'
                )
        
        # load Faive Hand asset with these options
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = self.cfg["env"]["hand_fix_base"]
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False  # TODO: check that this doesn't affect performance before PR merge!
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        hand_asset = self.gym.load_asset(
            self.sim, asset_root, hand_asset_file, asset_options
        )

        goal_asset_options = gymapi.AssetOptions()
        goal_asset_options.fix_base_link = True
        goal_asset_options.collapse_fixed_joints = True
        goal_asset_options.disable_gravity = True 
        goal_asset_options.thickness = 0.001
        # goal_asset_options.angular_damping = 1000000000.0  # 极高的角阻尼
        # goal_asset_options.linear_damping = 1000000000.0   # 极高的线性阻尼
        # goal_asset_options.max_angular_velocity = 0.0   # 禁止角速度
        # goal_asset_options.max_linear_velocity = 0.0    # 禁止线性速度
        # goal_asset_options.armature = 10000000.0            # 极高的关节刚度
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        # 设置为运动学模式，而非动力学模式
        goal_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        # 冻结所有自由度
        goal_asset_options.armature = 10.0
        hand_goal_asset = self.gym.load_asset(
        self.sim, asset_root, hand_asset_file, goal_asset_options
        )   
        # set up friction and restitution params
        # sphere rotation is somewhat brittle to these params...
        hand_props = self.gym.get_asset_rigid_shape_properties(hand_asset)
        for p in hand_props:
            p.friction = self.cfg["env"]["hand_friction"]
            p.torsion_friction = self.cfg["env"]["hand_friction"]
            p.restitution = 0.8
        self.gym.set_asset_rigid_shape_properties(hand_asset, hand_props)
        # self.gym.set_asset_rigid_shape_properties(hand_goal_asset, hand_props) 

        # goal_hand_props = self.gym.get_asset_rigid_shape_properties(hand_goal_asset)
        # for p in hand_props:
        #     p.filter = 0
        # self.gym.set_asset_rigid_shape_properties(hand_goal_asset, goal_hand_props)


        # define some variables based on the asset
        self.num_hand_bodies = self.gym.get_asset_rigid_body_count(hand_asset)
        self.num_hand_shapes = self.gym.get_asset_rigid_shape_count(hand_asset)
        self.num_hand_dofs = self.gym.get_asset_dof_count(hand_asset) #24 dim
        self.num_hand_actuators = self.gym.get_asset_actuator_count(hand_asset)
        self.num_hand_tendons = self.gym.get_asset_tendon_count(hand_asset)
        # set up tendons
        # tendons are used for simulating rolling contact joints using two hinge joints.
        # make tendon stiffer than the default stiffness of 1.
        # 30 is the value used in shadow_hand.py
        # however, scale fixed/joint coef of the Shadow Hand model is around 0.007, instead of 1 of the Hand.
        # https://mujoco.readthedocs.io/en/stable/XMLreference.html#tendon-fixed
        # So 30 must be scaled to achieve the same tendon stiffness, i.e. 30 / 0.007 ~ 4000
        # scale the damping as well
        limit_stiffness = 4000
        t_damping = 10
        tendon_props = self.gym.get_asset_tendon_properties(hand_asset)

        # go through all tendons in the robot model and set their properties
        for i in range(self.num_hand_tendons):
            tendon_name = self.gym.get_asset_tendon_name(hand_asset, i)
            tendon_props[i].limit_stiffness = limit_stiffness
            tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(hand_asset, tendon_props)

        # limit_stiffness = 0
        # t_damping = 0
        # tendon_props = self.gym.get_asset_tendon_properties(hand_goal_asset)

        # # go through all tendons in the robot model and set their properties
        # for i in range(self.num_hand_tendons):
        #     tendon_name = self.gym.get_asset_tendon_name(hand_goal_asset, i)
        #     tendon_props[i].limit_stiffness = limit_stiffness
        #     tendon_props[i].damping = t_damping
        # self.gym.set_asset_tendon_properties(hand_goal_asset, tendon_props)

        actuated_dof_names = [
            self.gym.get_asset_actuator_joint_name(hand_asset, i)
            for i in range(self.num_hand_actuators)
        ]
        actuated_dof_indices = []
        for name in actuated_dof_names:
            dof_index = self.gym.find_asset_dof_index(hand_asset, name)
            assert dof_index != -1, f"Could not find dof index for {name}"
            print(f"{name}\t->\t{dof_index}")
            actuated_dof_indices.append(dof_index)



        # get hand dof properties, loaded by Isaac Gym from the MJCF file
        hand_dof_props = self.gym.get_asset_dof_properties(hand_asset)
        print("hand_dof_props_lower:",hand_dof_props["lower"])
        print("hand_dof_props_upper:",hand_dof_props["upper"])
        # load joint range information
        self.hand_dof_lower_limits = []
        self.hand_dof_upper_limits = []
        # self.hand_dof_default_pos = []
        self.hand_dof_default_vel = [] 
        # self.hand_goal_dof_default_pos = []
        self.hand_goal_dof_default_vel = []             
        for i in range(self.num_hand_dofs):
            self.hand_dof_lower_limits.append(hand_dof_props["lower"][i])
            self.hand_dof_upper_limits.append(hand_dof_props["upper"][i])
            # self.hand_dof_default_pos.append(0.0)
            self.hand_dof_default_vel.append(0.0)
            # self.hand_goal_dof_default_pos.append(0.0)
            self.hand_goal_dof_default_vel.append(0.0)            

        # convert to torch tensors so they can be computed in GPU
        self.actuated_dof_indices = to_torch(
            actuated_dof_indices, dtype=torch.long, device=self.device
        )  # indices of dofs with actuators attached to them
        self.hand_dof_lower_limits = to_torch(
            self.hand_dof_lower_limits, device=self.device
        )
        self.hand_dof_upper_limits = to_torch(
            self.hand_dof_upper_limits, device=self.device
        )
        # self.hand_dof_default_pos = to_torch(
        #     self.hand_dof_default_pos, device=self.device
        # )
        self.hand_dof_default_vel = to_torch(
            self.hand_dof_default_vel, device=self.device
        )
        #目标手势的默认状态保存        
        # self.hand_goal_dof_default_pos = to_torch(
        #     self.hand_goal_dof_default_pos, device=self.device
        # )
        self.hand_goal_dof_default_vel = to_torch(
            self.hand_goal_dof_default_vel, device=self.device
        )        

        self.joint_range = (self.hand_dof_upper_limits - self.hand_dof_lower_limits).to(self.device)

        self.num_actuated_dofs = len(self.actuated_dof_indices)
        # if using tendons to simulate rolling contact joints, only some of the dofs have actuators
        # they should be all you need to reconstruct the hand's full state, so use these dofs for observations as well
        self.actuated_dof_lower_limits = self.hand_dof_lower_limits[
            self.actuated_dof_indices
        ]
        self.actuated_dof_upper_limits = self.hand_dof_upper_limits[
            self.actuated_dof_indices
        ]
        # if actuated dof have overridden range, use that instead
        for i in range(self.num_actuated_dofs):
            actuated_dof_lower_limit = self.actuated_dof_lower_limits[i]
            actuated_dof_upper_limit = self.actuated_dof_upper_limits[i]
            actuated_dof_range_override = self.cfg["env"]["actuated_dof_range_override"]
            if actuated_dof_range_override != "None":
                # if dof range override is set, use that instead
                # check that the override is within the range of the hand
                assert actuated_dof_lower_limit - 1e-3 <= actuated_dof_range_override[i][0]
                assert actuated_dof_range_override[i][1] <= actuated_dof_upper_limit + 1e-3
                assert actuated_dof_range_override[i][0] < actuated_dof_range_override[i][1]
                self.actuated_dof_lower_limits[i] = actuated_dof_range_override[i][0]
                self.actuated_dof_upper_limits[i] = actuated_dof_range_override[i][1]

        # create handles to access body parts of interest and force sensors
        sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
        
        pose_sensor_handles = [
            self.gym.find_asset_rigid_body_index(hand_asset, name)
            for name in self.cfg["asset"]["pose_sensor_names"]
        ]
        force_sensor_handles = [
            self.gym.find_asset_rigid_body_index(hand_asset, name)
            for name in self.cfg["asset"]["force_sensor_names"]
        ]
        for fs_handle in force_sensor_handles:
            self.gym.create_asset_force_sensor(hand_asset, fs_handle, sensor_pose)
    
        # load manipulated object and goal assets
        object_asset_list = []
        goal_asset_list = []
        for object_type in self.cfg["env"]["object_type"]:
            object_asset_file = os.path.normpath(
                    asset_files_dict[object_type]
                )
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.fix_base_link = self.cfg["env"]["object_fix_base"]
            object_asset_list.append(self.gym.load_asset(
                self.sim, asset_root, object_asset_file, object_asset_options
            ))
            object_asset_options.disable_gravity = True
            goal_asset_list.append(self.gym.load_asset(
                self.sim, asset_root, object_asset_file, object_asset_options
            ))
            if len(object_asset_list) == 1:
                self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset_list[-1])
                self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset_list[-1])
                self.num_object_dofs = self.gym.get_asset_dof_count(object_asset_list[-1])
            else:
                # check that all object assets are the same
                assert self.num_object_bodies == self.gym.get_asset_rigid_body_count(object_asset_list[-1])
                assert self.num_object_shapes == self.gym.get_asset_rigid_shape_count(object_asset_list[-1])
                assert self.num_object_dofs == self.gym.get_asset_dof_count(object_asset_list[-1])
            # check that goal assets are the same as object assets
            assert self.num_object_bodies == self.gym.get_asset_rigid_body_count(goal_asset_list[-1])
            assert self.num_object_shapes == self.gym.get_asset_rigid_shape_count(goal_asset_list[-1])
            assert self.num_object_dofs == self.gym.get_asset_dof_count(goal_asset_list[-1])
            

        # selected_index, hand_start_pose, object_start_pose, hand_joints_state = self.load_grasp_poses(self.cfg['asset']['grasp_file'])
        #抓取数据集加载
        self.grasp_data_dict = np.load(self.cfg['asset']['grasp_file'], allow_pickle=True)
        self.grasp_database_size = self.grasp_data_dict.shape[0]
        print("self.grasp_database_size:",self.grasp_database_size)
        # 预加载到GPU（在创建环境之前）
        self._init_grasp_data_tensors()  



        # hand_start_pose = gymapi.Transform()
        # hand_start_pose.p = gymapi.Vec3(self.cfg['env']['hand_start_p'][0],
        #                                 self.cfg['env']['hand_start_p'][1],
        #                                 self.cfg['env']['hand_start_p'][2])
        # object_start_pose = gymapi.Transform()
        # object_start_pose.p = gymapi.Vec3()
        # hand_start_pose.r = gymapi.Quat(self.cfg['env']['hand_start_r'][0],
        #                                 self.cfg['env']['hand_start_r'][1],
        #                                 self.cfg['env']['hand_start_r'][2],
        #                                 self.cfg['env']['hand_start_r'][3])
        [pose_dx, pose_dy, pose_dz] = self.cfg["env"]["hand_start_transform"]  #手部姿态与数据集姿态对齐

        # [pose_dx, pose_dy, pose_dz] = self.cfg["env"]["goal_start_offset"]  #目标抓取姿态可视化与初始姿态的相对距离
        # goal_hand_start_pose.p.x +=pose_dx
        # goal_hand_start_pose.p.y +=pose_dy
        # goal_hand_start_pose.p.z +=pose_dz 
        # goal_object_start_pose.p.x +=pose_dx
        # goal_object_start_pose.p.y +=pose_dy
        # goal_object_start_pose.p.z +=pose_dz 

        
        # compute aggregate size
        max_agg_bodies = 2 * self.num_hand_bodies + 2 * self.num_object_bodies
        max_agg_shapes = 2 * self.num_hand_shapes + 2 * self.num_object_shapes

        self.envs = []
        hand_init_states = []
        hand_dof_init_pos = []
        hand_goal_init_states = []
        hand_goal_dof_init_pos = []
        object_init_states = []
        object_goal_init_states = []
        hand_indices = []
        object_indices = []
        goal_hand_indices = []
        goal_object_indices = []
        # one-hot encoding which saves the object type loaded in each environment
        self.object_type = torch.zeros([self.num_envs, len(self.cfg["env"]["object_type"])])
        self.current_grasp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 当前手势索引
        self.target_grasp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)   # 目标手势索引      
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.cfg["env"]["aggregate_mode"]:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # current_index = np.random.randint(0, self.grasp_database_size)
            # self.current_grasp_idx[i] = current_index
            hand_init_pose = self.load_grasp_poses(np.random.randint(0, self.grasp_database_size))
            if hand_init_pose is None or np.any([x is None for x in hand_init_pose]):  # 更安全的检查
                raise ValueError("初始抓取姿态加载失败！")
            selected_index, hand_start_pose, object_start_pose, hand_joints_state, obj_scale= hand_init_pose
            self.current_grasp_idx[i] = selected_index
            # 保存当前状态
            if len(hand_joints_state) != 22:
                raise ValueError(f"hand_joints_state应为22维,但实际为{len(hand_joints_state)}维")

            goal_index = self.select_goal_by_nn(selected_index)
            self.target_grasp_idx[i] = goal_index
            #加载目标手势
            hand_goal_pose = self.load_grasp_poses(goal_index)
            if hand_goal_pose is None or np.any([x is None for x in hand_goal_pose]):  
                raise ValueError("目标抓取姿态加载失败！")
            goal_selected_index, goal_hand_start_pose, goal_object_start_pose, goal_hand_joints_state, goal_obj_scale= hand_goal_pose
            # 保存当前状态
            if len(goal_hand_joints_state) != 22:
                raise ValueError(f"goal_hand_joints_state应为22维,但实际为{len(goal_hand_joints_state)}维")

            hand_start_pose.p.x +=pose_dx
            hand_start_pose.p.y +=pose_dy
            hand_start_pose.p.z +=pose_dz  

            goal_hand_start_pose.p.x +=pose_dx
            goal_hand_start_pose.p.y +=pose_dy
            goal_hand_start_pose.p.z +=pose_dz  

            #手与物体向上平移0.5m
            hand_start_pose.p.z +=  0.5
            object_start_pose.p.z +=  0.5
            goal_hand_start_pose.p.z +=  0.5
            goal_object_start_pose.p.z +=  0.5

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            actor_handle = self.gym.create_actor(
                env_ptr, hand_asset, hand_start_pose, "ourhand", i, 1, -1
            )
            #设置机器人关节物理属性
            self.gym.set_actor_dof_properties(env_ptr, actor_handle, hand_dof_props)

            self.gym.enable_actor_dof_force_sensors(
                env_ptr, actor_handle
            )  # need to be explicitly enabled for torque sensors to work

            # set the first body to be black (base of the hand) to match real robot
            self.gym.set_rigid_body_color(
                env_ptr, actor_handle, 1, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.25, 0.25))

            hand_init_states.append(
                [
                    hand_start_pose.p.x,
                    hand_start_pose.p.y,
                    hand_start_pose.p.z,
                    hand_start_pose.r.x,
                    hand_start_pose.r.y,
                    hand_start_pose.r.z,
                    hand_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ])
            hand_dof_pos = [0.0] * 24 # 初始化24维列表
            hand_dof_pos[2: ] = hand_joints_state
            # print("hand_dof_pos:", hand_dof_pos)
            hand_dof_init_pos.append(hand_dof_pos)

            # add goal hand pose
            goal_actor_handle = self.gym.create_actor(
                env_ptr, hand_goal_asset, goal_hand_start_pose, "goal_hand", i + self.num_envs, 2, 0
            )
            self.gym.set_rigid_body_color(
                env_ptr, goal_actor_handle, 1, gymapi.MESH_VISUAL, gymapi.Vec3(0.25, 1, 0.25))
            # 获取并修改目标手部关节的DOF属性，使其保持静态
            goal_hand_dof_props = self.gym.get_actor_dof_properties(env_ptr, goal_actor_handle)
            for j in range(self.num_hand_dofs):
                goal_hand_dof_props["stiffness"][j] = 1.0e10 # 极高的刚度
                goal_hand_dof_props["damping"][j] = 1.0e10   # 极高的阻尼
                goal_hand_dof_props["driveMode"][j] = gymapi.DOF_MODE_POS # 确保是位置控制模式

            self.gym.set_actor_dof_properties(env_ptr, goal_actor_handle, goal_hand_dof_props)                
            hand_goal_init_states.append(
                [
                    goal_hand_start_pose.p.x,
                    goal_hand_start_pose.p.y,
                    goal_hand_start_pose.p.z,
                    goal_hand_start_pose.r.x,
                    goal_hand_start_pose.r.y,
                    goal_hand_start_pose.r.z,
                    goal_hand_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                    # *goal_hand_joints_state.tolist()  # 关节角度
                ])
            hand_goal_dof_pos = [0.0] * 24 # 初始化24维列表
            hand_goal_dof_pos[2: ] = goal_hand_joints_state
            # print("hand_goal_dof_pos:", hand_goal_dof_pos)
            hand_goal_dof_init_pos.append(hand_goal_dof_pos)

            # add object, seg_id = 1
            # randomly choose which object this environment will have
            # object_index = torch.randint(len(self.cfg["env"]["object_type"]), (1,)).item()
            object_index = 0
            object_handle = self.gym.create_actor(
                env_ptr, object_asset_list[object_index], object_start_pose, "object", i , 0, 1
            )
            self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)
            self.object_type[i][object_index] = 1
            object_init_states.append(
                [
                    object_start_pose.p.x,
                    object_start_pose.p.y,
                    object_start_pose.p.z,
                    object_start_pose.r.x,
                    object_start_pose.r.y,
                    object_start_pose.r.z,
                    object_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ) 

            # add goal object
            # by setting the fifth argument to not coincide with the others, the goal object does not collide with anything else
            goal_obj_handle = self.gym.create_actor(
                env_ptr,
                goal_asset_list[object_index],
                goal_object_start_pose,  # this will be immediately overwritten in reset_goal_states()
                "goal_object",
                i + 2 * self.num_envs,
                3,
                0,
            )
            self.gym.set_actor_scale(env_ptr, goal_obj_handle, obj_scale)
            object_goal_init_states.append(
                [
                    goal_object_start_pose.p.x,
                    goal_object_start_pose.p.y,
                    goal_object_start_pose.p.z,
                    goal_object_start_pose.r.x,
                    goal_object_start_pose.r.y,
                    goal_object_start_pose.r.z,
                    goal_object_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )            
           
            # save the indices of each item in the environment
            hand_idx = self.gym.get_actor_index(
                env_ptr, actor_handle, gymapi.DOMAIN_SIM
            )

            goal_hand_idx = self.gym.get_actor_index(
                env_ptr, goal_actor_handle, gymapi.DOMAIN_SIM
            )
      
            object_idx = self.gym.get_actor_index(
                env_ptr, object_handle, gymapi.DOMAIN_SIM
            )

            goal_object_idx = self.gym.get_actor_index(
                env_ptr, goal_obj_handle, gymapi.DOMAIN_SIM
            )

            hand_indices.append(hand_idx)
            goal_hand_indices.append(goal_hand_idx)            
            object_indices.append(object_idx)
            goal_object_indices.append(goal_object_idx)


            if self.cfg["env"]["aggregate_mode"]:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
        
        # used for resetting the hand and object
        self.hand_init_states = to_torch(
            hand_init_states, dtype=torch.float, device=self.device
        ).view(self.num_envs, 13)
        self.object_init_states = to_torch(
            object_init_states, dtype=torch.float, device=self.device
        ).view(self.num_envs, 13)
        self.hand_goal_init_states = to_torch(
            hand_goal_init_states, dtype=torch.float, device=self.device
        ).view(self.num_envs, 13)        
        self.object_goal_init_states = to_torch(
            object_goal_init_states, dtype=torch.float, device=self.device
        ).view(self.num_envs, 13)


        self.hand_dof_init_pos = to_torch(
            hand_dof_init_pos, dtype=torch.float, device=self.device
        ).view(self.num_envs, 24)
        self.hand_goal_dof_init_pos = to_torch(
            hand_goal_dof_init_pos, dtype=torch.float, device=self.device
        ).view(self.num_envs, 24)

        # this tensor saves the goal state of the object (the actual values should be updated in reset_goal_states())
        self.object_goal_states = self.object_goal_init_states.clone()
        self.hand_goal_states = self.hand_goal_init_states.clone()
       

        self.hand_indices = to_torch(hand_indices, dtype=torch.long, device=self.device)
        self.goal_hand_indices = to_torch(
            goal_hand_indices, dtype=torch.long, device=self.device
        )        
        self.object_indices = to_torch(
            object_indices, dtype=torch.long, device=self.device
        )
    
        self.goal_object_indices = to_torch(
            goal_object_indices, dtype=torch.long, device=self.device
        )
        self.force_sensor_handles = to_torch(
            force_sensor_handles, dtype=torch.long, device=self.device
        )
        self.pose_sensor_handles = to_torch(
            pose_sensor_handles, dtype=torch.long, device=self.device
        )

    def _init_grasp_data_tensors(self):
        """
        在初始化时将所有抓取数据转换为CUDA张量,避免运行时的重复转换和字典访问
        """
        import time
        print("正在将抓取数据转换为CUDA张量...")
        start_time = time.time()
        
        # 获取数据库大小
        num_grasps = self.grasp_database_size
        num_joints = len(self.joint_names)
        
        # 预分配所有张量
        self.grasp_tensors = {
            # 手部位置和旋转
            'hand_pos': torch.zeros((num_grasps, 3), device=self.device, dtype=torch.float32),
            'hand_rot_euler': torch.zeros((num_grasps, 3), device=self.device, dtype=torch.float32),
            'hand_rot_quat': torch.zeros((num_grasps, 4), device=self.device, dtype=torch.float32),
            
            # 手部关节角度
            'hand_joints': torch.zeros((num_grasps, num_joints), device=self.device, dtype=torch.float32),
            
            # 物体位置和旋转
            'obj_pos': torch.zeros((num_grasps, 3), device=self.device, dtype=torch.float32),
            'obj_rot_euler': torch.zeros((num_grasps, 3), device=self.device, dtype=torch.float32),
            'obj_rot_quat': torch.zeros((num_grasps, 4), device=self.device, dtype=torch.float32),
            
            # 物体缩放系数
            'obj_scale': torch.ones(num_grasps, device=self.device, dtype=torch.float32),
            
            # 有效性标记
            'valid_mask': torch.ones(num_grasps, device=self.device, dtype=torch.bool)
        }
        
        print(f"预分配张量完成，开始转换 {num_grasps} 个抓取姿态...")
        
        # 批量转换数据
        batch_size = min(1000, num_grasps)  # 每批处理1000个，避免内存溢出
        
        for start_idx in range(0, num_grasps, batch_size):
            end_idx = min(start_idx + batch_size, num_grasps)
            current_batch_size = end_idx - start_idx
            
            # 临时存储当前批次的数据
            batch_hand_pos = torch.zeros((current_batch_size, 3), dtype=torch.float32)
            batch_hand_rot = torch.zeros((current_batch_size, 3), dtype=torch.float32)
            batch_hand_joints = torch.zeros((current_batch_size, num_joints), dtype=torch.float32)
            batch_obj_pos = torch.zeros((current_batch_size, 3), dtype=torch.float32)
            batch_obj_rot = torch.zeros((current_batch_size, 3), dtype=torch.float32)
            batch_obj_scale = torch.ones(current_batch_size, dtype=torch.float32)
            batch_valid = torch.ones(current_batch_size, dtype=torch.bool)
            
            # 处理当前批次的每个抓取姿态
            for i, grasp_idx in enumerate(range(start_idx, end_idx)):
                try:
                    grasp_data = self.grasp_data_dict[grasp_idx]
                    
                    # 提取基本信息
                    qpos = grasp_data['qpos']
                    obj_scale = grasp_data.get('scale', 1.0)
                    batch_obj_scale[i] = obj_scale
                    
                    # 提取手部数据
                    hand_pos = [qpos[name] for name in self.hand_translation_names]
                    hand_rot = [qpos[name] for name in self.hand_rot_names]
                    hand_joints = [qpos[name] for name in self.joint_names]
                    
                    batch_hand_pos[i] = torch.tensor(hand_pos, dtype=torch.float32)
                    batch_hand_rot[i] = torch.tensor(hand_rot, dtype=torch.float32)
                    batch_hand_joints[i] = torch.tensor(hand_joints, dtype=torch.float32)
                    
                    # 提取物体数据
                    if 'obj' in grasp_data:
                        obj_data = grasp_data['obj']
                        obj_translation_names = ['OBJTx', 'OBJTy', 'OBJTz']
                        obj_rot_names = ['OBJRx', 'OBJRy', 'OBJRz']
                        
                        obj_pos = [obj_data[name] for name in obj_translation_names]
                        obj_rot = [obj_data[name] for name in obj_rot_names]
                        
                        batch_obj_pos[i] = torch.tensor(obj_pos, dtype=torch.float32)
                        batch_obj_rot[i] = torch.tensor(obj_rot, dtype=torch.float32)
                    else:
                        # 如果没有物体数据，使用零值
                        batch_obj_pos[i] = torch.zeros(3, dtype=torch.float32)
                        batch_obj_rot[i] = torch.zeros(3, dtype=torch.float32)
                    
                except Exception as e:
                    print(f"处理抓取姿态 {grasp_idx} 时出错: {e}")
                    batch_valid[i] = False
                    # 使用默认值
                    batch_hand_pos[i] = torch.zeros(3, dtype=torch.float32)
                    batch_hand_rot[i] = torch.zeros(3, dtype=torch.float32)
                    batch_hand_joints[i] = torch.zeros(num_joints, dtype=torch.float32)
                    batch_obj_pos[i] = torch.zeros(3, dtype=torch.float32)
                    batch_obj_rot[i] = torch.zeros(3, dtype=torch.float32)
                    batch_obj_scale[i] = 1.0
            
            # 转换为CUDA张量并存储
            self.grasp_tensors['hand_pos'][start_idx:end_idx] = batch_hand_pos.to(self.device)
            self.grasp_tensors['hand_rot_euler'][start_idx:end_idx] = batch_hand_rot.to(self.device)
            self.grasp_tensors['hand_joints'][start_idx:end_idx] = batch_hand_joints.to(self.device)
            self.grasp_tensors['obj_pos'][start_idx:end_idx] = batch_obj_pos.to(self.device)
            self.grasp_tensors['obj_rot_euler'][start_idx:end_idx] = batch_obj_rot.to(self.device)
            self.grasp_tensors['obj_scale'][start_idx:end_idx] = batch_obj_scale.to(self.device)
            self.grasp_tensors['valid_mask'][start_idx:end_idx] = batch_valid.to(self.device)
            
            if (start_idx // batch_size + 1) % 10 == 0:
                print(f"已处理 {end_idx}/{num_grasps} 个抓取姿态...")
        
        # 预计算四元数
        print("预计算四元数...")
        self._precompute_quaternions()
        
        # 验证数据完整性
        valid_count = self.grasp_tensors['valid_mask'].sum().item()
        print(f"抓取数据转换完成！")
        print(f"总抓取姿态数: {num_grasps}")
        print(f"有效抓取姿态数: {valid_count}")
        print(f"无效抓取姿态数: {num_grasps - valid_count}")
        print(f"预加载用时: {time.time() - start_time:.2f}秒")


    def _precompute_quaternions(self):
        """预计算所有欧拉角对应的四元数"""
        print("正在预计算四元数...")
        
        # 使用批量操作转换欧拉角到四元数
        hand_euler = self.grasp_tensors['hand_rot_euler']
        obj_euler = self.grasp_tensors['obj_rot_euler']
        
        # 批量转换手部四元数
        self.grasp_tensors['hand_rot_quat'] = self._euler_to_quat_batch(hand_euler)
        
        # 批量转换物体四元数
        self.grasp_tensors['obj_rot_quat'] = self._euler_to_quat_batch(obj_euler)
        
        print("四元数预计算完成")

     


    def select_goal_by_nn(self, current_grasp_index, joint_min_dist=0.1, joint_max_dist=1.0, 
                        obj_pos_dist=0.08, obj_rot_dist=1.0, alpha=0.5, sample_size=200):
        """
        使用最近邻搜索从抓取数据库中选择适当距离的目标手势，同时考虑手部关节角度和物体位置的差距
        
        Args:
            current_grasp_index: 当前手势的索引
            joint_min_dist: 关节角度最小距离阈值
            joint_max_dist: 关节角度最大距离阈值
            obj_min_dist: 物体位置最小距离阈值
            obj_max_dist: 物体位置最大距离阈值
            alpha: 关节角度距离的权重系数 (0-1)，物体距离权重为 (1-alpha)
            sample_size: 随机下采样的数量，减少计算量
        
        Returns:
            goal_index: 选择的目标手势索引
        """
        # 获取当前手势数据
        current_grasp = self.grasp_data_dict[current_grasp_index]
        
        # 获取当前手部关节角度
        current_joints = []
        for name in self.joint_names:
            current_joints.append(current_grasp['qpos'][name])
        # current_joints = np.array(current_joints)
        current_joints = to_torch(current_joints, dtype=torch.float, device=self.device)
        # 获取当前物体位置和旋转
        current_obj_pos = None
        current_obj_rot = None
        if 'obj' in current_grasp:
            obj_data = current_grasp['obj']
            current_obj_pos = np.array([
                obj_data['OBJTx'], 
                obj_data['OBJTy'], 
                obj_data['OBJTz']
            ])
            current_obj_rot = torch.tensor([
                obj_data['OBJRx'], 
                obj_data['OBJRy'], 
                obj_data['OBJRz']
            ], dtype=torch.float32, device=self.device)
        
        # 随机下采样以减少计算量
        total_grasps = self.grasp_database_size
        if sample_size >= total_grasps:
            candidate_indices = np.arange(total_grasps)
        else:
            candidate_indices = np.random.choice(total_grasps, size=sample_size, replace=False)
        
        # 计算所有候选手势与当前手势的综合距离
        valid_candidates = []
        
        for idx in candidate_indices:
            if idx == current_grasp_index:
                continue  # 跳过当前手势
                
            candidate_grasp = self.grasp_data_dict[idx]
            
            # 计算关节角度距离
            candidate_joints = []
            for name in self.joint_names:
                candidate_joints.append(candidate_grasp['qpos'][name])
            # candidate_joints = np.array(candidate_joints)
            candidate_joints = to_torch(candidate_joints, dtype=torch.float, device=self.device)            
            #joint_distance = np.linalg.norm(current_joints - candidate_joints)
            joint_distance = self.calculate_joint_distance_pytorch(current_joints, candidate_joints, use_normalized_distance=True)
            
            # 计算物体位置和旋转距离
            obj_pos_distance = float('inf')
            obj_rot_distance = float('inf')
            
            if current_obj_pos is not None and 'obj' in candidate_grasp:
                obj_data = candidate_grasp['obj']
                candidate_obj_pos = np.array([
                    obj_data['OBJTx'], 
                    obj_data['OBJTy'], 
                    obj_data['OBJTz']
                ])
                candidate_obj_rot = torch.tensor([
                    obj_data['OBJRx'], 
                    obj_data['OBJRy'], 
                    obj_data['OBJRz']
                ], dtype=torch.float32, device=self.device)
                
                # 位置距离
                obj_pos_distance = np.linalg.norm(current_obj_pos - candidate_obj_pos)
                
                # 旋转距离（使用欧拉角的范数作为简单近似）
                #obj_rot_distance = np.linalg.norm(current_obj_rot - candidate_obj_rot)
                obj_rot_distance = self.calculate_rotation_distance_pytorch(current_obj_rot, candidate_obj_rot)
            
            # 组合所有度量以计算综合距离
            if current_obj_pos is not None and 'obj' in candidate_grasp:
                # 同时考虑关节角度、物体位置和旋转
                # 位置距离作为主要物体度量
                obj_distance = obj_pos_distance + 0.1 * obj_rot_distance
                
                # 判断是否在合适的范围内
                if (joint_min_dist <= joint_distance <= joint_max_dist and
                    0 <= obj_pos_distance <= obj_pos_dist and 
                    0 <= obj_rot_distance <= obj_rot_dist):
                    # 计算综合距离分数
                    combined_distance = alpha * joint_distance + (1 - alpha) * obj_distance * 10
                    
                    valid_candidates.append({
                        'index': idx,
                        'joint_distance': joint_distance,
                        'obj_pos_distance': obj_pos_distance,
                        'obj_rot_distance': obj_rot_distance,
                        'combined_distance': combined_distance
                    })
            else:
                # 如果没有物体信息，只考虑关节角度
                if joint_min_dist <= joint_distance <= joint_max_dist:
                    valid_candidates.append({
                        'index': idx,
                        'joint_distance': joint_distance,
                        'obj_pos_distance': float('inf'),
                        'obj_rot_distance': float('inf'),
                        'combined_distance': joint_distance
                    })
        
        # 如果没有找到合适范围内的手势，扩大搜索范围
        if not valid_candidates:
            # print("在指定范围内未找到合适的目标手势，扩大搜索范围")
            all_candidates = []
            
            for idx in candidate_indices:
                if idx == current_grasp_index:
                    continue
                    
                candidate_grasp = self.grasp_data_dict[idx]
                
                # 计算关节角度距离
                candidate_joints = []
                for name in self.joint_names:
                    candidate_joints.append(candidate_grasp['qpos'][name])
                # candidate_joints = np.array(candidate_joints)
                candidate_joints = to_torch(candidate_joints, dtype=torch.float, device=self.device)
                #joint_distance = np.linalg.norm(current_joints - candidate_joints)
                joint_distance = self.calculate_joint_distance_pytorch(current_joints, candidate_joints)
                
                # 计算物体位置和旋转距离
                obj_pos_distance = float('inf')
                obj_rot_distance = float('inf')
                
                if current_obj_pos is not None and 'obj' in candidate_grasp:
                    obj_data = candidate_grasp['obj']
                    candidate_obj_pos = np.array([
                        obj_data['OBJTx'], 
                        obj_data['OBJTy'], 
                        obj_data['OBJTz']
                    ])
                    candidate_obj_rot = torch.tensor([
                        obj_data['OBJRx'], 
                        obj_data['OBJRy'], 
                        obj_data['OBJRz']
                    ], dtype=torch.float32, device=self.device)
                    
                    # 位置距离
                    obj_pos_distance = np.linalg.norm(current_obj_pos - candidate_obj_pos)
                    
                    # 旋转距离
                    #obj_rot_distance = np.linalg.norm(current_obj_rot - candidate_obj_rot)
                    obj_rot_distance = self.calculate_rotation_distance_pytorch(current_obj_rot, candidate_obj_rot)
                    
                    obj_distance = obj_pos_distance + 0.1 * obj_rot_distance
                    # 计算综合距离
                    combined_distance = alpha * joint_distance + (1 - alpha) * obj_distance * 10
                else:
                    combined_distance = joint_distance
                
                all_candidates.append({
                    'index': idx,
                    'joint_distance': joint_distance,
                    'obj_pos_distance': obj_pos_distance,
                    'obj_rot_distance': obj_rot_distance,
                    'combined_distance': combined_distance
                })
            
            # 按综合距离排序
            all_candidates.sort(key=lambda x: x['combined_distance'])
            
            # 选择距离适中的候选目标
            if len(all_candidates) > 0:
                # 选择距离中间的候选目标（难度适中）
                middle_idx = len(all_candidates) // 2
                goal_index = all_candidates[middle_idx]['index']
            else:
                # 如果没有其他候选目标，则随机选择（极少发生）
                goal_index = np.random.randint(0, self.grasp_database_size)
                while goal_index == current_grasp_index:
                    goal_index = np.random.randint(0, self.grasp_database_size)
        else:          
            # 当有多个满足条件的候选目标时，随机选择一个候选目标
            selected_idx = np.random.randint(0, len(valid_candidates))
            goal_index = valid_candidates[selected_idx]['index']
        
            # 打印选择的目标信息便于调试
            selected_candidate = valid_candidates[selected_idx]
            # print(f"当前索引：{current_grasp_index},目标索引: {goal_index}, 关节距离: {selected_candidate['joint_distance']:.4f}, "
            #     f"物体位置距离: {selected_candidate['obj_pos_distance']}, 物体旋转距离: {selected_candidate['obj_rot_distance'] :.4f}")
        return goal_index



    def select_goal_by_nn_batch(self, current_grasp_indices, joint_min_dist=0.1, joint_max_dist=1.0, 
                        obj_pos_dist=0.08, obj_rot_dist=1.0, alpha=0.5, sample_size=200):
        """
        使用最近邻搜索从抓取数据库中选择适当距离的目标手势，同时考虑手部关节角度和物体位置的差距（支持批量操作）
        
        Args:
            current_grasp_indices: 当前手势的索引，可以是单个索引或索引张量
            joint_min_dist: 关节角度最小距离阈值
            joint_max_dist: 关节角度最大距离阈值
            obj_pos_dist: 物体位置最小距离阈值
            obj_rot_dist: 物体位置最大距离阈值
            alpha: 关节角度距离的权重系数 (0-1)，物体距离权重为 (1-alpha)
            sample_size: 随机下采样的数量，减少计算量
        
        Returns:
            goal_indices: 选择的目标手势索引，与输入格式相同
        """
        
        # 处理输入格式，统一转换为张量
        if isinstance(current_grasp_indices, (int, np.integer)):
            # 单个索引的情况，转换为张量
            current_grasp_indices = torch.tensor([current_grasp_indices], device=self.device)
        elif isinstance(current_grasp_indices, torch.Tensor):
            # 张量输入
            if current_grasp_indices.dim() == 0:  # 标量张量
                current_grasp_indices = current_grasp_indices.unsqueeze(0)
        else:
            # 其他格式转换为张量
            current_grasp_indices = torch.tensor(current_grasp_indices, device=self.device)
        
        batch_size = current_grasp_indices.shape[0]
        goal_indices = torch.zeros_like(current_grasp_indices)
        
        # 随机下采样候选索引（所有批次共用）
        total_grasps = self.grasp_database_size
        if sample_size >= total_grasps:
            candidate_indices = np.arange(total_grasps)
        else:
            candidate_indices = np.random.choice(total_grasps, size=sample_size, replace=False)
        
        # 对每个环境分别处理
        for batch_idx in range(batch_size):
            current_grasp_index = current_grasp_indices[batch_idx].item()
            
            # 获取当前手势数据
            current_grasp = self.grasp_data_dict[current_grasp_index]
            
            # 获取当前手部关节角度
            current_joints = []
            for name in self.joint_names:
                current_joints.append(current_grasp['qpos'][name])
            current_joints = to_torch(current_joints, dtype=torch.float, device=self.device)
            
            # 获取当前物体位置和旋转
            current_obj_pos = None
            current_obj_rot = None
            if 'obj' in current_grasp:
                obj_data = current_grasp['obj']
                current_obj_pos = np.array([
                    obj_data['OBJTx'], 
                    obj_data['OBJTy'], 
                    obj_data['OBJTz']
                ])
                current_obj_rot = torch.tensor([
                    obj_data['OBJRx'], 
                    obj_data['OBJRy'], 
                    obj_data['OBJRz']
                ], dtype=torch.float32, device=self.device)
            
            # 计算所有候选手势与当前手势的综合距离
            valid_candidates = []
            
            for idx in candidate_indices:
                if idx == current_grasp_index:
                    continue  # 跳过当前手势
                    
                candidate_grasp = self.grasp_data_dict[idx]
                
                # 计算关节角度距离
                candidate_joints = []
                for name in self.joint_names:
                    candidate_joints.append(candidate_grasp['qpos'][name])
                candidate_joints = to_torch(candidate_joints, dtype=torch.float, device=self.device)            
                joint_distance = self.calculate_joint_distance_pytorch(current_joints, candidate_joints, use_normalized_distance=True)
                
                # 计算物体位置和旋转距离
                obj_pos_distance = float('inf')
                obj_rot_distance = float('inf')
                
                if current_obj_pos is not None and 'obj' in candidate_grasp:
                    obj_data = candidate_grasp['obj']
                    candidate_obj_pos = np.array([
                        obj_data['OBJTx'], 
                        obj_data['OBJTy'], 
                        obj_data['OBJTz']
                    ])
                    candidate_obj_rot = torch.tensor([
                        obj_data['OBJRx'], 
                        obj_data['OBJRy'], 
                        obj_data['OBJRz']
                    ], dtype=torch.float32, device=self.device)
                    
                    # 位置距离
                    obj_pos_distance = np.linalg.norm(current_obj_pos - candidate_obj_pos)
                    
                    # 旋转距离（使用欧拉角的范数作为简单近似）
                    obj_rot_distance = self.calculate_rotation_distance_pytorch(current_obj_rot, candidate_obj_rot)
                
                # 组合所有度量以计算综合距离
                if current_obj_pos is not None and 'obj' in candidate_grasp:
                    # 同时考虑关节角度、物体位置和旋转
                    # 位置距离作为主要物体度量
                    obj_distance = obj_pos_distance + 0.1 * obj_rot_distance
                    
                    # 判断是否在合适的范围内
                    if (joint_min_dist <= joint_distance <= joint_max_dist and
                        0 <= obj_pos_distance <= obj_pos_dist and 
                        0 <= obj_rot_distance <= obj_rot_dist):
                        # 计算综合距离分数
                        combined_distance = alpha * joint_distance + (1 - alpha) * obj_distance * 10
                        
                        valid_candidates.append({
                            'index': idx,
                            'joint_distance': joint_distance,
                            'obj_pos_distance': obj_pos_distance,
                            'obj_rot_distance': obj_rot_distance,
                            'combined_distance': combined_distance
                        })
                else:
                    # 如果没有物体信息，只考虑关节角度
                    if joint_min_dist <= joint_distance <= joint_max_dist:
                        valid_candidates.append({
                            'index': idx,
                            'joint_distance': joint_distance,
                            'obj_pos_distance': float('inf'),
                            'obj_rot_distance': float('inf'),
                            'combined_distance': joint_distance
                        })
            
            # 如果没有找到合适范围内的手势，扩大搜索范围
            if not valid_candidates:
                # print(f"批次 {batch_idx}: 在指定范围内未找到合适的目标手势，扩大搜索范围")
                all_candidates = []
                
                for idx in candidate_indices:
                    if idx == current_grasp_index:
                        continue
                        
                    candidate_grasp = self.grasp_data_dict[idx]
                    
                    # 计算关节角度距离
                    candidate_joints = []
                    for name in self.joint_names:
                        candidate_joints.append(candidate_grasp['qpos'][name])
                    candidate_joints = to_torch(candidate_joints, dtype=torch.float, device=self.device)
                    joint_distance = self.calculate_joint_distance_pytorch(current_joints, candidate_joints)
                    
                    # 计算物体位置和旋转距离
                    obj_pos_distance = float('inf')
                    obj_rot_distance = float('inf')
                    
                    if current_obj_pos is not None and 'obj' in candidate_grasp:
                        obj_data = candidate_grasp['obj']
                        candidate_obj_pos = np.array([
                            obj_data['OBJTx'], 
                            obj_data['OBJTy'], 
                            obj_data['OBJTz']
                        ])
                        candidate_obj_rot = torch.tensor([
                            obj_data['OBJRx'], 
                            obj_data['OBJRy'], 
                            obj_data['OBJRz']
                        ], dtype=torch.float32, device=self.device)
                        
                        # 位置距离
                        obj_pos_distance = np.linalg.norm(current_obj_pos - candidate_obj_pos)
                        
                        # 旋转距离
                        obj_rot_distance = self.calculate_rotation_distance_pytorch(current_obj_rot, candidate_obj_rot)
                        
                        obj_distance = obj_pos_distance + 0.1 * obj_rot_distance
                        # 计算综合距离
                        combined_distance = alpha * joint_distance + (1 - alpha) * obj_distance * 10
                    else:
                        combined_distance = joint_distance
                    
                    all_candidates.append({
                        'index': idx,
                        'joint_distance': joint_distance,
                        'obj_pos_distance': obj_pos_distance,
                        'obj_rot_distance': obj_rot_distance,
                        'combined_distance': combined_distance
                    })
                
                # 按综合距离排序
                all_candidates.sort(key=lambda x: x['combined_distance'])
                
                # 选择距离适中的候选目标
                if len(all_candidates) > 0:
                    # 选择距离中间的候选目标（难度适中）
                    middle_idx = len(all_candidates) // 2
                    goal_index = all_candidates[middle_idx]['index']
                else:
                    # 如果没有其他候选目标，则随机选择（极少发生）
                    goal_index = np.random.randint(0, self.grasp_database_size)
                    while goal_index == current_grasp_index:
                        goal_index = np.random.randint(0, self.grasp_database_size)
            else:          
                # 当有多个满足条件的候选目标时，随机选择一个候选目标
                selected_idx = np.random.randint(0, len(valid_candidates))
                goal_index = valid_candidates[selected_idx]['index']
            
                # 打印选择的目标信息便于调试
                selected_candidate = valid_candidates[selected_idx]
                # print(f"批次 {batch_idx}: 当前索引：{current_grasp_index},目标索引: {goal_index}, 关节距离: {selected_candidate['joint_distance']:.4f}, "
                #     f"物体位置距离: {selected_candidate['obj_pos_distance']}, 物体旋转距离: {selected_candidate['obj_rot_distance']:.4f}")
            
            # 存储该批次的目标索引
            goal_indices[batch_idx] = goal_index
        
        # 返回结果张量
        return goal_indices

    def select_goal_by_nn_batch_gpu_parallel(self, current_grasp_indices, joint_min_dist=0.1, joint_max_dist=1.0, 
                                            obj_pos_dist=0.08, obj_rot_dist=1.0, alpha=0.5, sample_size=200):
        """
        使用完全并行的张量运算进行高效的批量最近邻搜索（无for循环）
        
        Args:
            current_grasp_indices: 当前手势的索引张量 [batch_size]
            joint_min_dist: 关节角度最小距离阈值
            joint_max_dist: 关节角度最大距离阈值
            obj_pos_dist: 物体位置最小距离阈值
            obj_rot_dist: 物体旋转最大距离阈值
            alpha: 关节角度距离的权重系数 (0-1)
            sample_size: 随机下采样的数量
        
        Returns:
            goal_indices: 选择的目标手势索引张量 [batch_size]
        """
        # 处理输入格式
        if isinstance(current_grasp_indices, (int, np.integer)):
            current_grasp_indices = torch.tensor([current_grasp_indices], device=self.device)
        elif isinstance(current_grasp_indices, torch.Tensor):
            if current_grasp_indices.dim() == 0:
                current_grasp_indices = current_grasp_indices.unsqueeze(0)
            current_grasp_indices = current_grasp_indices.to(self.device)
        else:
            current_grasp_indices = torch.tensor(current_grasp_indices, device=self.device)
        
        batch_size = current_grasp_indices.shape[0]
        
        # 随机选择候选索引
        total_grasps = self.grasp_database_size
        if sample_size >= total_grasps:
            candidate_indices = torch.arange(total_grasps, device=self.device)
        else:
            candidate_indices = torch.randperm(total_grasps, device=self.device)[:sample_size]
        
        num_candidates = len(candidate_indices)
        
        # 获取当前手势的数据 [batch_size, ...]
        current_joints = self.grasp_tensors['hand_joints'][current_grasp_indices]  # [batch_size, num_joints]
        current_obj_pos = self.grasp_tensors['obj_pos'][current_grasp_indices]     # [batch_size, 3]
        current_obj_rot = self.grasp_tensors['obj_rot_euler'][current_grasp_indices]  # [batch_size, 3]
        current_obj_rot_quat = self.grasp_tensors['obj_rot_quat'][current_grasp_indices]  # [batch_size, 4]
        
        # 获取候选手势的数据 [num_candidates, ...]
        candidate_joints = self.grasp_tensors['hand_joints'][candidate_indices]    # [num_candidates, num_joints]
        candidate_obj_pos = self.grasp_tensors['obj_pos'][candidate_indices]       # [num_candidates, 3]
        candidate_obj_rot = self.grasp_tensors['obj_rot_euler'][candidate_indices] # [num_candidates, 3]
        candidate_obj_rot_quat = self.grasp_tensors['obj_rot_quat'][candidate_indices]  # [num_candidates, 4]
        candidate_valid = self.grasp_tensors['valid_mask'][candidate_indices]      # [num_candidates]
        
        # # 批量计算所有距离 [batch_size, num_candidates]
        # joint_distances = torch.norm(current_joints - candidate_joints, p=2, dim=1)#self._calculate_joint_distance_batch_parallel(current_joints, candidate_joints)
        # pos_distances = torch.norm(current_obj_pos - candidate_obj_pos, p=2, dim=1)#self._calculate_position_distance_batch_parallel(current_obj_pos, candidate_obj_pos)
        # quat_diff = quat_mul(current_obj_rot_quat, quat_conjugate(candidate_obj_rot_quat))
        # rot_distances = 2.0 * torch.asin(
        #     torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
        # )
        # 扩展维度以支持广播
        current_joints_expanded = current_joints.unsqueeze(1)  # [batch_size, 1, num_joints]
        candidate_joints_expanded = candidate_joints.unsqueeze(0)  # [1, num_candidates, num_joints]
        joint_distances = torch.norm(current_joints_expanded - candidate_joints_expanded, p=2, dim=2)  # [batch_size, num_candidates]
        
        current_obj_pos_expanded = current_obj_pos.unsqueeze(1)  # [batch_size, 1, 3]
        candidate_obj_pos_expanded = candidate_obj_pos.unsqueeze(0)  # [1, num_candidates, 3]
        pos_distances = torch.norm(current_obj_pos_expanded - candidate_obj_pos_expanded, p=2, dim=2)  # [batch_size, num_candidates]
        
        # 计算旋转距离 - 使用四元数
        current_obj_rot_quat_expanded = current_obj_rot_quat.unsqueeze(1)  # [batch_size, 1, 4]
        candidate_obj_rot_quat_expanded = candidate_obj_rot_quat.unsqueeze(0)  # [1, num_candidates, 4]
  
        # 计算四元数点积
        dot_product = torch.sum(current_obj_rot_quat_expanded * candidate_obj_rot_quat_expanded, dim=2)  # [batch_size, num_candidates]
        dot_product = torch.clamp(torch.abs(dot_product), 0, 1)
        rot_distances = 2 * torch.acos(dot_product)  # [batch_size, num_candidates]        

        # 创建排除当前索引的掩码 [batch_size, num_candidates]
        current_indices_expanded = current_grasp_indices.unsqueeze(1)  # [batch_size, 1]
        candidate_indices_expanded = candidate_indices.unsqueeze(0)    # [1, num_candidates]
        exclude_current_mask = current_indices_expanded != candidate_indices_expanded  # [batch_size, num_candidates]
        
        # 创建有效性掩码 [batch_size, num_candidates]
        valid_mask = candidate_valid.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_candidates]
        
        # 应用距离约束 [batch_size, num_candidates]
        joint_constraint = (joint_distances >= joint_min_dist) & (joint_distances <= joint_max_dist)
        pos_constraint = pos_distances <= obj_pos_dist
        rot_constraint = rot_distances <= obj_rot_dist
        
        # 综合所有约束
        all_constraints = joint_constraint & pos_constraint & rot_constraint & exclude_current_mask & valid_mask
        # 检查每个batch是否有满足约束的候选
        has_valid_candidates = all_constraints.any(dim=1)  # [batch_size]
        
        # 初始化结果张量
        goal_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # 对于有满足约束候选的batch，执行正常的选择逻辑
        if has_valid_candidates.any():
            # 计算综合距离分数 [batch_size, num_candidates]
            obj_distance = pos_distances + 0.1 * rot_distances
            combined_distance = alpha * joint_distances + (1 - alpha) * obj_distance * 10   
            # 对于每个batch，选择最优候选
            goal_indices = self._select_candidates_random(
                combined_distance, all_constraints, exclude_current_mask, valid_mask, candidate_indices
            )                 
        # 对于没有满足约束候选的batch，随机生成一个非当前索引
        no_valid_candidates = ~has_valid_candidates  # [batch_size]
        if no_valid_candidates.any():
            num_no_valid = no_valid_candidates.sum().item()
            
            # 为每个没有有效候选的batch生成随机索引
            for i, batch_idx in enumerate(torch.where(no_valid_candidates)[0]):
                current_idx = current_grasp_indices[batch_idx].item()
                
                # 生成一个不等于当前索引的随机索引
                while True:
                    random_idx = torch.randint(0, total_grasps, (1,), device=self.device).item()
                    if random_idx != current_idx:
                        goal_indices[batch_idx] = random_idx
                        break
        
        return goal_indices



    # def _select_best_candidates_parallel(self, combined_distance, all_constraints, exclude_current_mask, 
    #                                 valid_mask, candidate_indices):
    #     """
    #     并行选择最优候选
        
    #     Args:
    #         combined_distance: [batch_size, num_candidates] 综合距离分数
    #         all_constraints: [batch_size, num_candidates] 所有约束的掩码
    #         exclude_current_mask: [batch_size, num_candidates] 排除当前索引的掩码
    #         valid_mask: [batch_size, num_candidates] 有效性掩码
    #         candidate_indices: [num_candidates] 候选索引
        
    #     Returns:
    #         goal_indices: [batch_size] 选择的目标索引
    #     """
    #     batch_size, num_candidates = combined_distance.shape
        
    #     # 对于满足所有约束的候选，使用较小的距离分数
    #     constrained_distance = combined_distance.clone()
    #     constrained_distance[~all_constraints] = float('inf')
        
    #     # 检查每个batch是否有满足约束的候选
    #     has_valid_candidates = all_constraints.any(dim=1)  # [batch_size]
        
    #     # 对于没有满足约束的batch，使用所有有效候选
    #     fallback_mask = exclude_current_mask & valid_mask  # [batch_size, num_candidates]
    #     fallback_distance = combined_distance.clone()
    #     fallback_distance[~fallback_mask] = float('inf')
        
    #     # 选择最终的距离矩阵
    #     final_distance = torch.where(
    #         has_valid_candidates.unsqueeze(1),  # [batch_size, 1]
    #         constrained_distance,
    #         fallback_distance
    #     )
        
    #     # 方法1：选择最小距离的候选
    #     # min_indices = torch.argmin(final_distance, dim=1)  # [batch_size]
    #     # goal_indices = candidate_indices[min_indices]
        
    #     # 方法2：选择距离适中的候选（更好的多样性）
    #     # 对距离进行排序，选择中位数位置的候选
    #     sorted_distances, sorted_indices = torch.sort(final_distance, dim=1)  # [batch_size, num_candidates]
        
    #     # 找到每个batch中第一个非无限大的距离位置
    #     valid_positions = (sorted_distances != float('inf')).float()  # [batch_size, num_candidates]
    #     num_valid_per_batch = valid_positions.sum(dim=1)  # [batch_size]
        
    #     # 选择中位数位置（如果有效候选数量>1）
    #     middle_positions = torch.clamp(torch.div(num_valid_per_batch, 2, rounding_mode='floor'), min=0, max=num_candidates-1).long()
        
    #     # 对于只有一个有效候选的情况，选择第一个
    #     middle_positions = torch.where(num_valid_per_batch <= 1, 0, middle_positions)
        
    #     # 获取选择的候选索引
    #     batch_indices = torch.arange(batch_size, device=self.device)
    #     selected_candidate_positions = sorted_indices[batch_indices, middle_positions]  # [batch_size]
    #     goal_indices = candidate_indices[selected_candidate_positions]
        
    #     # 方法3：随机选择（在满足约束的候选中）
    #     # 可以添加随机性来增加多样性
    #     # random_selection = torch.rand(batch_size, num_candidates, device=self.device)
    #     # random_selection[final_distance == float('inf')] = -1  # 排除无效候选
    #     # random_indices = torch.argmax(random_selection, dim=1)  # [batch_size]
    #     # goal_indices = candidate_indices[random_indices]
        
    #     return goal_indices


    def _select_candidates_random(self, combined_distance, all_constraints, exclude_current_mask, 
                                            valid_mask, candidate_indices):
        """
        并行随机选择候选（在满足约束的候选中随机选择）
        
        Args:
            combined_distance: [batch_size, num_candidates] 综合距离分数
            all_constraints: [batch_size, num_candidates] 所有约束的掩码
            exclude_current_mask: [batch_size, num_candidates] 排除当前索引的掩码
            valid_mask: [batch_size, num_candidates] 有效性掩码
            candidate_indices: [num_candidates] 候选索引
        
        Returns:
            goal_indices: [batch_size] 选择的目标索引
        """
        batch_size, num_candidates = combined_distance.shape
        
        # 检查每个batch是否有满足约束的候选
        has_valid_candidates = all_constraints.any(dim=1)  # [batch_size]
        
        # 创建选择掩码
        selection_mask = torch.where(
            has_valid_candidates.unsqueeze(1),  # [batch_size, 1]
            all_constraints,  # 使用满足约束的候选
            exclude_current_mask & valid_mask  # 使用所有有效候选（排除当前索引）
        )
        
        # 检查是否还有任何有效候选（排除当前索引）
        has_any_valid = selection_mask.any(dim=1)  # [batch_size]
        
        # 生成随机权重
        random_weights = torch.rand(batch_size, num_candidates, device=self.device)
        random_weights[~selection_mask] = -1  # 排除无效候选
        
        # 选择随机权重最大的候选
        selected_positions = torch.argmax(random_weights, dim=1)  # [batch_size]
        goal_indices = candidate_indices[selected_positions]
        
        
        return goal_indices

    # def select_goal_by_nn(self, current_grasp_index, joint_min_dist=0.1, joint_max_dist=0.5, 
    #                     obj_min_dist=0.05, obj_max_dist=0.2, alpha=0.7, sample_size=200):
    #     """
    #     使用最近邻搜索从抓取数据库中选择适当距离的目标手势，同时考虑手部关节角度和物体位置的差距
        
    #     Args:
    #         current_grasp_index: 当前手势的索引
    #         joint_min_dist: 关节角度最小距离阈值
    #         joint_max_dist: 关节角度最大距离阈值
    #         obj_min_dist: 物体位置最小距离阈值
    #         obj_max_dist: 物体位置最大距离阈值
    #         alpha: 关节角度距离的权重系数 (0-1)，物体距离权重为 (1-alpha)
    #         sample_size: 随机下采样的数量，减少计算量
        
    #     Returns:
    #         goal_index: 选择的目标手势索引
    #     """
    #     # 获取当前手势数据
    #     current_grasp = self.grasp_data_dict[current_grasp_index]
        
    #     # 获取当前手部关节角度
    #     current_joints = []
    #     for name in self.joint_names:
    #         current_joints.append(current_grasp['qpos'][name])
    #     current_joints = np.array(current_joints)
        
    #     # 获取当前物体位置和旋转
    #     current_obj_pos = None
    #     current_obj_rot = None
    #     if 'obj' in current_grasp:
    #         obj_data = current_grasp['obj']
    #         current_obj_pos = np.array([
    #             obj_data['OBJTx'], 
    #             obj_data['OBJTy'], 
    #             obj_data['OBJTz']
    #         ])
    #         current_obj_rot = np.array([
    #             obj_data['OBJRx'], 
    #             obj_data['OBJRy'], 
    #             obj_data['OBJRz']
    #         ])
        
    #     # 随机下采样以减少计算量
    #     total_grasps = self.grasp_database_size
    #     if sample_size >= total_grasps:
    #         candidate_indices = np.arange(total_grasps)
    #     else:
    #         candidate_indices = np.random.choice(total_grasps, size=sample_size, replace=False)
        
    #     # 计算所有候选手势与当前手势的综合距离
    #     valid_candidates = []
        
    #     for idx in candidate_indices:
    #         if idx == current_grasp_index:
    #             continue  # 跳过当前手势
                
    #         candidate_grasp = self.grasp_data_dict[idx]
            
    #         # 计算关节角度距离
    #         candidate_joints = []
    #         for name in self.joint_names:
    #             candidate_joints.append(candidate_grasp['qpos'][name])
    #         candidate_joints = np.array(candidate_joints)
    #         joint_distance = np.linalg.norm(current_joints - candidate_joints)
            
    #         # 计算物体位置和旋转距离
    #         obj_pos_distance = float('inf')
    #         obj_rot_distance = float('inf')
            
    #         if current_obj_pos is not None and 'obj' in candidate_grasp:
    #             obj_data = candidate_grasp['obj']
    #             candidate_obj_pos = np.array([
    #                 obj_data['OBJTx'], 
    #                 obj_data['OBJTy'], 
    #                 obj_data['OBJTz']
    #             ])
    #             candidate_obj_rot = np.array([
    #                 obj_data['OBJRx'], 
    #                 obj_data['OBJRy'], 
    #                 obj_data['OBJRz']
    #             ])
                
    #             # 位置距离
    #             obj_pos_distance = np.linalg.norm(current_obj_pos - candidate_obj_pos)
                
    #             # 旋转距离（使用欧拉角的范数作为简单近似）
    #             obj_rot_distance = np.linalg.norm(current_obj_rot - candidate_obj_rot)
            
    #         # 组合所有度量以计算综合距离
    #         if current_obj_pos is not None and 'obj' in candidate_grasp:
    #             # 同时考虑关节角度、物体位置和旋转
    #             # 位置距离作为主要物体度量
    #             obj_distance = obj_pos_distance
                
    #             # 判断是否在合适的范围内
    #             if (joint_min_dist <= joint_distance <= joint_max_dist and
    #                 obj_min_dist <= obj_distance <= obj_max_dist):
    #                 # 计算综合距离分数
    #                 combined_distance = alpha * joint_distance + (1 - alpha) * obj_distance
                    
    #                 valid_candidates.append({
    #                     'index': idx,
    #                     'joint_distance': joint_distance,
    #                     'obj_pos_distance': obj_pos_distance,
    #                     'obj_rot_distance': obj_rot_distance,
    #                     'combined_distance': combined_distance
    #                 })
    #         else:
    #             # 如果没有物体信息，只考虑关节角度
    #             if joint_min_dist <= joint_distance <= joint_max_dist:
    #                 valid_candidates.append({
    #                     'index': idx,
    #                     'joint_distance': joint_distance,
    #                     'obj_pos_distance': float('inf'),
    #                     'obj_rot_distance': float('inf'),
    #                     'combined_distance': joint_distance
    #                 })
        
    #     # 如果没有找到合适范围内的手势，扩大搜索范围
    #     if not valid_candidates:
    #         print("在指定范围内未找到合适的目标手势，扩大搜索范围")
    #         all_candidates = []
            
    #         for idx in candidate_indices:
    #             if idx == current_grasp_index:
    #                 continue
                    
    #             candidate_grasp = self.grasp_data_dict[idx]
                
    #             # 计算关节角度距离
    #             candidate_joints = []
    #             for name in self.joint_names:
    #                 candidate_joints.append(candidate_grasp['qpos'][name])
    #             candidate_joints = np.array(candidate_joints)
    #             joint_distance = np.linalg.norm(current_joints - candidate_joints)
                
    #             # 计算物体位置和旋转距离
    #             obj_pos_distance = float('inf')
    #             obj_rot_distance = float('inf')
                
    #             if current_obj_pos is not None and 'obj' in candidate_grasp:
    #                 obj_data = candidate_grasp['obj']
    #                 candidate_obj_pos = np.array([
    #                     obj_data['OBJTx'], 
    #                     obj_data['OBJTy'], 
    #                     obj_data['OBJTz']
    #                 ])
    #                 candidate_obj_rot = np.array([
    #                     obj_data['OBJRx'], 
    #                     obj_data['OBJRy'], 
    #                     obj_data['OBJRz']
    #                 ])
                    
    #                 # 位置距离
    #                 obj_pos_distance = np.linalg.norm(current_obj_pos - candidate_obj_pos)
                    
    #                 # 旋转距离
    #                 obj_rot_distance = np.linalg.norm(current_obj_rot - candidate_obj_rot)
                    
    #                 # 计算综合距离
    #                 combined_distance = alpha * joint_distance + (1 - alpha) * obj_pos_distance
    #             else:
    #                 combined_distance = joint_distance
                
    #             all_candidates.append({
    #                 'index': idx,
    #                 'joint_distance': joint_distance,
    #                 'obj_pos_distance': obj_pos_distance,
    #                 'obj_rot_distance': obj_rot_distance,
    #                 'combined_distance': combined_distance
    #             })
            
    #         # 按综合距离排序
    #         all_candidates.sort(key=lambda x: x['combined_distance'])
            
    #         # 选择距离适中的候选目标
    #         if len(all_candidates) > 0:
    #             # 选择距离中间的候选目标（难度适中）
    #             middle_idx = len(all_candidates) // 2
    #             goal_index = all_candidates[middle_idx]['index']
    #         else:
    #             # 如果没有其他候选目标，则随机选择（极少发生）
    #             goal_index = np.random.randint(0, self.grasp_database_size)
    #             while goal_index == current_grasp_index:
    #                 goal_index = np.random.randint(0, self.grasp_database_size)
    #     else:
    #         # 当有多个满足条件的候选目标时，随机选择一个候选目标
    #         selected_idx = np.random.randint(0, len(valid_candidates))
    #         goal_index = valid_candidates[selected_idx]['index']
            
    #         # 打印选择的目标信息便于调试
    #         selected_candidate = valid_candidates[selected_idx]
    #         print(f"选择目标，索引: {goal_index}, 关节距离: {selected_candidate['joint_distance']:.4f}, "
    #             f"物体位置距离: {selected_candidate['obj_pos_distance']:.4f}")
        
    #     return goal_index
    def load_grasp_poses(self, grasp_index=None):
        """
        从抓取数据文件中加载手势并返回手部位姿、物体位姿和关节角度
        
        Args:
            grasp_file_path: 抓取数据文件的路径(.npy文件)
            grasp_index: 要加载的手势索引,如果为None则随机选择一个手势
        
        Returns:
            selected_index: 选择的手势索引
            hand_pose: 手部姿态 (gymapi.Transform)
            object_pose: 物体姿态 (gymapi.Transform)
            hand_joints_state: 手指关节角度 (numpy.ndarray)
        """
        # 加载抓取数据
        # data_dict = np.load(grasp_file_path, allow_pickle=True)
        # batch_size = data_dict.shape[0]
        if isinstance(grasp_index, torch.Tensor):
            grasp_index = grasp_index.item()  # 转换为Python整数
        # 选择手势索引
        if grasp_index is None:
            # 随机选择一个手势
            selected_index = np.random.randint(0, self.grasp_database_size)
        else:
            # 使用指定索引
            selected_index = grasp_index
            # 检查索引是否有效
            if selected_index < 0 or selected_index >= self.grasp_database_size:
                print(f"无效的抓取索引: {selected_index}，总抓取数量: {self.grasp_database_size}")
                return None, None, None, None
        
        selected_grasp = self.grasp_data_dict[selected_index]

        # 提取物体缩放系数
        obj_scale = selected_grasp.get('scale', 1.0)
        # 提取手部位姿数据
        qpos = selected_grasp['qpos']
        
        # 获取手部旋转数据(欧拉角)并转换为四元数
        hand_rot = [qpos[name] for name in self.hand_rot_names]
        hand_quat = transforms3d.euler.euler2quat(*hand_rot)
        
        # 获取手部位置数据
        hand_pos = [qpos[name] for name in self.hand_translation_names]
        
        # 获取关节角度数据
        hand_joints_state = np.array([qpos[name] for name in self.joint_names])
        
        # 创建手部位姿变换
        hand_pose = gymapi.Transform()
        hand_pose.p = gymapi.Vec3(hand_pos[0], hand_pos[1], hand_pos[2])
        hand_pose.r = gymapi.Quat(hand_quat[1], hand_quat[2], hand_quat[3], hand_quat[0])  # 注意四元数顺序 (x,y,z,w)
        
        # 创建物体位姿变换
        object_pose = gymapi.Transform()
        
        # 如果数据中包含物体位姿
        if 'obj' in selected_grasp:
            obj_data = selected_grasp['obj']
            obj_translation_names = ['OBJTx', 'OBJTy', 'OBJTz']
            obj_rot_names = ['OBJRx', 'OBJRy', 'OBJRz']
            
            # 获取物体位置和旋转
            obj_pos = [obj_data[name] for name in obj_translation_names]
            obj_rot = [obj_data[name] for name in obj_rot_names]
            obj_quat = transforms3d.euler.euler2quat(*obj_rot)
            
            # 设置物体位姿
            object_pose.p = gymapi.Vec3(obj_pos[0], obj_pos[1], obj_pos[2])
            object_pose.r = gymapi.Quat(obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0])  # 注意四元数顺序 (x,y,z,w)
        
        return selected_index, hand_pose, object_pose, hand_joints_state, obj_scale


    def load_grasp_poses_batch_gpu(self, grasp_indices):
        """
        GPU优化版本：从预加载的GPU张量中直接索引抓取数据
        
        Args:
            grasp_indices: 要加载的手势索引张量 [B] (torch.Tensor)
            
        Returns:
            hand_pos: 手部位置 [B,3] (torch.Tensor)
            hand_rot: 手部旋转（四元数）[B,4] (torch.Tensor)
            obj_pos: 物体位置 [B,3] (torch.Tensor)
            obj_rot: 物体旋转（四元数）[B,4] (torch.Tensor)
            hand_joints: 手指关节角度 [B,N] (torch.Tensor)
        """
        # 处理输入索引（确保为张量）
        if isinstance(grasp_indices, (int, np.integer)):
            grasp_indices = torch.tensor([grasp_indices], device=self.device)
        elif isinstance(grasp_indices, (list, tuple, np.ndarray)):
            grasp_indices = torch.tensor(grasp_indices, device=self.device)
        elif not isinstance(grasp_indices, torch.Tensor):
            grasp_indices = torch.tensor(grasp_indices, device=self.device)
        else:
            grasp_indices = grasp_indices.to(self.device)
        
        # 索引有效性检查
        valid_mask = (grasp_indices >= 0) & (grasp_indices < self.grasp_database_size)
        if not valid_mask.all():
            print(f"警告: 发现无效索引，将使用随机索引替代")
            invalid_indices = ~valid_mask
            random_indices = torch.randint(0, self.grasp_database_size, 
                                        (invalid_indices.sum(),), device=self.device)
            grasp_indices[invalid_indices] = random_indices
        
        # 检查是否已初始化GPU张量
        if not hasattr(self, 'grasp_tensors'):
            print("警告: GPU张量未初始化，回退到CPU版本")
            return self.load_grasp_poses_batch(grasp_indices)
        
        # 直接从GPU张量中索引数据（超快速！）
        hand_pos = self.grasp_tensors['hand_pos'][grasp_indices].clone()
        hand_rot_quat = self.grasp_tensors['hand_rot_quat'][grasp_indices].clone()
        hand_joints = self.grasp_tensors['hand_joints'][grasp_indices].clone()
        obj_pos = self.grasp_tensors['obj_pos'][grasp_indices].clone()
        obj_rot_quat = self.grasp_tensors['obj_rot_quat'][grasp_indices].clone()
        
        # 应用z轴偏移（与CPU版本保持一致）
        offset = torch.tensor([0., 0., 0.5], device=self.device)
        hand_pos = hand_pos + offset
        obj_pos = obj_pos + offset
        
        return hand_pos, hand_rot_quat, obj_pos, obj_rot_quat, hand_joints

    def load_grasp_poses_batch(self, grasp_index=None):
        """
        兼容性包装器:调用GPU优化版本
        """
        if grasp_index is None:
            grasp_index = torch.randint(0, self.grasp_database_size, (1,), device=self.device)
        
        return self.load_grasp_poses_batch_gpu(grasp_index)

    # def load_grasp_poses_batch(self, grasp_index=None):
    #     """
    #     从抓取数据文件中加载手势并返回手部位姿、物体位姿和关节角度
        
    #     Args:
    #         grasp_index: 要加载的手势索引,可以是单个值、torch.Tensor 或 None
            
    #     Returns:
    #         hand_pos: 手部位置 [B,3] (torch.Tensor)
    #         hand_rot: 手部旋转（四元数）[B,4] (torch.Tensor)
    #         obj_pos: 物体位置 [B,3] (torch.Tensor)
    #         obj_rot: 物体旋转（四元数）[B,4] (torch.Tensor)
    #         hand_joints: 手指关节角度 [B,N] (torch.Tensor)
    #     """
    #     # 处理输入索引（转换为张量）
    #     if grasp_index is None:
    #         selected_index = torch.randint(0, self.grasp_database_size, (1,), device=self.device)
    #     else:
    #         if not isinstance(grasp_index, torch.Tensor):
    #             grasp_index = torch.tensor([grasp_index], device=self.device)
    #         selected_index = grasp_index.to(self.device)
        
    #     # 检查索引是否有效
    #     invalid_mask = (selected_index < 0) | (selected_index >= self.grasp_database_size)
    #     if invalid_mask.any():
    #         print(f"发现无效索引: {selected_index[invalid_mask]}, 使用随机索引替代")
    #         selected_index[invalid_mask] = torch.randint(
    #             0, self.grasp_database_size, 
    #             (invalid_mask.sum(),), 
    #             device=self.device
    #         )
        
    #     batch_size = len(selected_index)
        
    #     # 预分配张量
    #     hand_pos = torch.zeros((batch_size, 3), device=self.device)
    #     hand_rot_euler = torch.zeros((batch_size, 3), device=self.device)
    #     hand_joints = torch.zeros((batch_size, len(self.joint_names)), device=self.device)
    #     obj_pos = torch.zeros((batch_size, 3), device=self.device)
    #     obj_rot_euler = torch.zeros((batch_size, 3), device=self.device)
        
    #     # 填充数据
    #     for i, idx in enumerate(selected_index.cpu().numpy()):
    #         data = self.grasp_data_dict[int(idx)]
    #         qpos = data['qpos']
            
    #         # 手部数据（z轴+0.5）
    #         hand_pos[i] = torch.tensor([
    #             qpos[name] for name in self.hand_translation_names
    #         ], device=self.device) + torch.tensor([0., 0., 0.5], device=self.device)
            
    #         hand_rot_euler[i] = torch.tensor([qpos[name] for name in self.hand_rot_names])
    #         hand_joints[i] = torch.tensor([qpos[name] for name in self.joint_names])
            
    #         # 物体数据（z轴+0.5）
    #         if 'obj' in data:
    #             obj_data = data['obj']
    #             obj_pos[i] = torch.tensor([
    #                 obj_data[name] for name in ['OBJTx', 'OBJTy', 'OBJTz']
    #             ], device=self.device) + torch.tensor([0., 0., 0.5], device=self.device)
    #             obj_rot_euler[i] = torch.tensor([obj_data[name] for name in ['OBJRx', 'OBJRy', 'OBJRz']])
        
    #     # 欧拉角 -> 四元数（批量转换）
    #     def euler_to_quat(euler):
    #         roll, pitch, yaw = euler.unbind(-1)
    #         cy = torch.cos(yaw * 0.5)
    #         sy = torch.sin(yaw * 0.5)
    #         cp = torch.cos(pitch * 0.5)
    #         sp = torch.sin(pitch * 0.5)
    #         cr = torch.cos(roll * 0.5)
    #         sr = torch.sin(roll * 0.5)
            
    #         w = cr * cp * cy + sr * sp * sy
    #         x = sr * cp * cy - cr * sp * sy
    #         y = cr * sp * cy + sr * cp * sy
    #         z = cr * cp * sy - sr * sp * cy
            
    #         return torch.stack([x, y, z, w], dim=-1)
        
    #     hand_rot = euler_to_quat(hand_rot_euler)  # [B,4]
    #     obj_rot = euler_to_quat(obj_rot_euler)    # [B,4]
        
    #     return hand_pos, hand_rot, obj_pos, obj_rot, hand_joints


    def calculate_joint_distance(self, joints1, joints2, use_normalized_distance=True):
        """
        计算ShadowHand关节角度之间的距离,使用实际的关节上下限
        
        Args:
            joints1: 第一组关节角度
            joints2: 第二组关节角度
            use_normalized_distance: 是否使用标准化距离计算（基于关节范围）
        
        Returns:
            distance: 关节角度距离
        """
        if not use_normalized_distance:
            # 简单的欧氏距离
            return np.linalg.norm(joints1 - joints2)
    
        # 计算角度差异
        angle_diffs = torch.abs(joints1 - joints2)
        
        # 标准化到[0,1]范围，基于关节的实际运动范围
        # 使用torch.where处理除零情况
        normalized_diffs = angle_diffs / self.joint_range[:22]
        
        # 计算标准化欧氏距离
        return torch.norm(normalized_diffs, p=2)
        # 使用实际的关节上下限进行标准化
        # joint_diffs = []
        # for i in range(len(joints1)):
        #     angle1 = joints1[i]
        #     angle2 = joints2[i]
            
        #     # 获取当前关节的上下限和范围
        #     joint_lower = self.hand_dof_lower_limits[i]
        #     joint_upper = self.hand_dof_upper_limits[i]
        #     joint_range = self.joint_range[i]
            
        #     # 计算角度差异
        #     angle_diff = abs(angle1 - angle2)
            
        #     # 标准化到[0,1]范围，基于关节的实际运动范围
        #     if joint_range > 1e-6:  # 避免除零
        #         normalized_diff = angle_diff / joint_range
        #     else:
        #         normalized_diff = angle_diff
                
        #     joint_diffs.append(normalized_diff)
        
        # # 计算标准化欧氏距离
        # return np.linalg.norm(joint_diffs)
    def calculate_joint_distance_pytorch_batch(self, joints1: torch.Tensor, 
                                    joints2: torch.Tensor, 
                                    use_normalized_distance: bool = True) -> torch.Tensor:
        """
        计算考虑周期性的关节角度距离（支持批量处理）
        
        Args:
            joints1: 当前关节角度 [batch_size, n_joints] 或 [n_joints]
            joints2: 目标关节角度
            use_normalized_distance: 是否使用关节范围归一化
            
        Returns:
            distance: 各样本的距离值 [batch_size,] 或 标量
        """
        # 确保输入为张量且设备一致
        joints1 = torch.as_tensor(joints1, device=self.device)
        joints2 = torch.as_tensor(joints2, device=self.device)
        
        # 计算周期性角度差（自动处理批量）
        angle_diff = torch.atan2(
            torch.sin(joints1 - joints2),
            torch.cos(joints1 - joints2)
        )  # 结果范围[-π, π]
        
        if not use_normalized_distance:
            # 原始角度差范数
            return torch.norm(torch.abs(angle_diff), dim=-1)
        # 修复归一化问题
        joint_range = self.joint_range[:joints1.shape[-1]]  # 确保维度匹配
        if joints1.dim() == 2:  # 批处理情况
            joint_range = joint_range.unsqueeze(0).expand(joints1.shape[0], -1)
        
        normalized_diff = torch.abs(angle_diff) / (joint_range + 1e-8)  # 避免除零
        
        return torch.norm(normalized_diff, p=2, dim=-1)        
        # # 使用关节范围归一化（假设self.joint_range已定义）
        # normalized_diff = torch.abs(angle_diff) / self.joint_range.unsqueeze(0)  # 广播机制
        
        # # 计算加权欧氏距离（按关节重要性可在此处添加权重）
        # return torch.norm(normalized_diff, p=2, dim=-1)

    def calculate_joint_distance_pytorch(self, joints1, joints2, use_normalized_distance=True):
        """
        计算ShadowHand关节角度之间的距离,使用实际的关节上下限
        
        Args:
            joints1: 第一组关节角度
            joints2: 第二组关节角度
            use_normalized_distance: 是否使用标准化距离计算（基于关节范围）
        
        Returns:
            distance: 关节角度距离
        """
        # joints1 = torch.as_tensor(joints1, device=self.device)
        # joints2 = torch.as_tensor(joints2, device=self.device)   
             
        if not use_normalized_distance:
            # 处理周期性后计算简单欧氏距离
            periodic_diff = (joints1 - joints2 + torch.pi) % (2 * torch.pi) - torch.pi
            return torch.norm(torch.abs(periodic_diff), dim=-1)
        
        # 计算考虑周期性的角度差值
        periodic_diff = (joints1 - joints2 + torch.pi) % (2 * torch.pi) - torch.pi
        corrected_diff = torch.abs(periodic_diff)
        normalized_diff = corrected_diff / self.joint_range[:22]
        
        # 计算加权欧氏距离
        return torch.norm(normalized_diff, dim=-1)

    def calculate_rotation_distance_pytorch(self, rot1, rot2):
        """
        使用PyTorch和您的四元数方法计算旋转距离
        
        Args:
            rot1: 第一个旋转的欧拉角 [rx, ry, rz]
            rot2: 第二个旋转的欧拉角 [rx, ry, rz]
        
        Returns:
            distance: 旋转距离（弧度）
        """
        # 将欧拉角转换为四元数
        quat1 = quat_from_euler_xyz(rot1[0], rot1[1], rot1[2])  # 您需要实现这个函数
        quat2 = quat_from_euler_xyz(rot2[0], rot2[1], rot2[2])
        
        # 使用您的方法计算旋转距离
        quat_diff = quat_mul(quat1, quat_conjugate(quat2))
        rot_dist = 2.0 * torch.asin(
            torch.clamp(torch.norm(quat_diff[0:3], p=2, dim=-1), max=1.0)
        )        
        return rot_dist





    # def load_grasp_poses(self, grasp_file_path, grasp_index=None):
    #     """
    #     从抓取数据文件中加载手势并返回手部位姿、物体位姿和关节角度
        
    #     Args:
    #         grasp_file_path: 抓取数据文件的路径(.npy文件)
    #         grasp_index: 要加载的手势索引,如果为None则随机选择一个手势
        
    #     Returns:
    #         selected_index: 选择的手势索引
    #         hand_pose: 手部姿态 (gymapi.Transform)
    #         object_pose: 物体姿态 (gymapi.Transform)
    #         hand_joints_state: 手指关节角度 (numpy.ndarray)
    #         obj_scale: 物体缩放系数
    #     """
    #     # 加载抓取数据
    #     data_dict = np.load(grasp_file_path, allow_pickle=True)
    #     batch_size = data_dict.shape[0]
        
    #     # 选择手势索引
    #     if grasp_index is None:
    #         # 随机选择一个手势
    #         selected_index = np.random.randint(0, batch_size)
    #     else:
    #         # 使用指定索引
    #         selected_index = grasp_index
    #         # 检查索引是否有效
    #         if selected_index < 0 or selected_index >= batch_size:
    #             print(f"无效的抓取索引: {selected_index}，总抓取数量: {batch_size}")
    #             return None, None, None, None, None
        
    #     selected_grasp = data_dict[selected_index]
        
    #     # 定义关节名称和坐标轴名称
    #     hand_translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    #     hand_rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    #     joint_names = [
    #         'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
    #         'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
    #         'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
    #         'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
    #         'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
    #     ]
    #     # 提取物体缩放系数，保持原始类型
    #     obj_scale = selected_grasp.get('scale', 1.0)
    #     # 提取手部位姿数据
    #     qpos = selected_grasp['qpos']
        
    #     # 获取手部旋转数据(欧拉角)并转换为四元数，保持原始数据类型
    #     hand_rot = [qpos[name] for name in hand_rot_names]
    #     hand_quat = transforms3d.euler.euler2quat(*hand_rot)
        
    #     # 获取手部位置数据，保持原始数据类型
    #     hand_pos = [qpos[name] for name in hand_translation_names]
        
    #     # 获取关节角度数据，保持原始数据类型
    #     hand_joints_state = np.array([qpos[name] for name in joint_names])
        
    #     # 创建手部位姿变换
    #     hand_pose = gymapi.Transform()
    #     hand_pose.p = gymapi.Vec3(hand_pos[0], hand_pos[1], hand_pos[2])
    #     hand_pose.r = gymapi.Quat(hand_quat[1], hand_quat[2], hand_quat[3], hand_quat[0])  # 注意四元数顺序 (x,y,z,w)
        
    #     # 创建物体位姿变换
    #     object_pose = gymapi.Transform()
        
    #     # 如果数据中包含物体位姿
    #     if 'obj' in selected_grasp:
    #         obj_data = selected_grasp['obj']
    #         obj_translation_names = ['OBJTx', 'OBJTy', 'OBJTz']
    #         obj_rot_names = ['OBJRx', 'OBJRy', 'OBJRz']
            
    #         # 获取物体位置和旋转，保持原始数据类型
    #         obj_pos = [obj_data[name] for name in obj_translation_names]
    #         obj_rot = [obj_data[name] for name in obj_rot_names]
    #         obj_quat = transforms3d.euler.euler2quat(*obj_rot)
            
    #         # 设置物体位姿
    #         object_pose.p = gymapi.Vec3(obj_pos[0], obj_pos[1], obj_pos[2])
    #         object_pose.r = gymapi.Quat(obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0])  # 注意四元数顺序 (x,y,z,w)
        
    #     return selected_index, hand_pose, object_pose, hand_joints_state, obj_scale



    def _parse_cfg(self, cfg):
        """
        Parse the configuration object and save relevant parameters
        """
        self.reward_scales = class_to_dict(cfg["rewards"]["scales"])
        self.obs_dims = class_to_dict(cfg["observations"]["obs_dims"])
        self.obs_scales = class_to_dict(cfg["observations"]["obs_scales"])

    # ------------- debug visualizer functions -----------------
    def _draw_sphere(self, env_idx, x, y, z, radius=0.05, color=(1, 1, 0)):
        """
        draw a sphere designated location for the environment with index env_idx
        """
        sphere_geom = gymutil.WireframeSphereGeometry(radius, 26, 26, None, color=color)
        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        gymutil.draw_lines(
            sphere_geom, self.gym, self.viewer, self.envs[env_idx], sphere_pose
        )

    def _draw_frame_axes(self, env_idx, pos, rot, ax_len=0.2):
        """
        draw xyz axes for the given frame
        """
        x_tip = pos + quat_apply(rot, to_torch([ax_len, 0, 0], device=self.device))
        y_tip = pos + quat_apply(rot, to_torch([0, ax_len, 0], device=self.device))
        z_tip = pos + quat_apply(rot, to_torch([0, 0, ax_len], device=self.device))
        x_tip = x_tip.cpu().numpy()
        y_tip = y_tip.cpu().numpy()
        z_tip = z_tip.cpu().numpy()
        pos = pos.cpu().numpy()
        self.gym.add_lines(
            self.viewer,
            self.envs[env_idx],
            1,
            [pos[0], pos[1], pos[2], x_tip[0], x_tip[1], x_tip[2]],
            [0.85, 0.1, 0.1],
        )
        self.gym.add_lines(
            self.viewer,
            self.envs[env_idx],
            1,
            [pos[0], pos[1], pos[2], y_tip[0], y_tip[1], y_tip[2]],
            [0.1, 0.85, 0.1],
        )
        self.gym.add_lines(
            self.viewer,
            self.envs[env_idx],
            1,
            [pos[0], pos[1], pos[2], z_tip[0], z_tip[1], z_tip[2]],
            [0.1, 0.1, 0.85],
        )

    # ------------- reward functions -----------------
    # the reward functions below may not be called depending on the reward
    # configuration, so they must not contain computation that is used in
    # other functions i.e. they should only compute the reward term and
    # nothing else.
    # for readability, rewards specific to a task should be named [task_name]task_[reward_name]

    # first define generic reward functions that can be used for any task

    def _reward_dof_acc_penalty(self):
        """
        Penalize joint acceleration, could remove shaking
        """
        return torch.norm(self.dof_acceleration, p=2, dim=-1)

    def _reward_dof_vel_penalty(self):
        """
        Penalize speed of the joints, smooth out movement
        """
        return torch.norm(self.hand_dof_vel, p=2, dim=-1)

    def _reward_action_penalty(self):
        """
        Penalize the magnitude of the action
        """
        return torch.norm(self.actions, p=2, dim=-1)

    def _reward_dof_trq_penalty(self):
        """
        Penalize the magnitude of the joint torque
        """
        return torch.norm(self.dof_force_tensor, p=2, dim=-1)

    def _reward_success(self):
        """
        Reward the agent for success (success_buf is computed in check_termination(), its definition is different for each task)
        """
        return self.success_buf
    # def _reward_joint_first_success(self):
    #     """关节第一次成功的奖励"""
    #     return (self.successes_joint & (~self.successes_joint_pre)).float()

    # def _reward_pos_first_success(self):
    #     """位置第一次成功的奖励"""
    #     return (self.successes_pos & (~self.successes_pos_pre)).float()

    # def _reward_rot_first_success(self):
    #     """姿态第一次成功的奖励"""
    #     return (self.successes_rot & (~self.successes_rot_pre)).float()
    def _reward_drop_penalty(self):
        """
        Penalize the agent for falling over
        """
        return self.dropped_buf

    def _reward_simple_hand_flat(self):
        """
        simple reward function that rewards the joint pos being close to zero
        useful for debugging policies, since it is such an easy task
        """
        dist_from_zero = torch.norm(self.hand_dof_pos, p=2, dim=-1)
        return 1.0 / (dist_from_zero + 0.1)

    # ---------------------------------------------------------------------
    # 定义手内重定向任务专用的奖励函数
    # def _reward_reorienttask_obj_dist(self):
    #     """
    #     Reward the agent based on the distance between the object and the goal
    #     """
    #     return torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)

    # def _reward_reorienttask_obj_rot(self):
    #     """
    #     Orientation alignment for the cube in hand and goal cube
    #     """
    #     quat_diff = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
    #     rot_dist = 2.0 * torch.asin(
    #         torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
    #     )
    #     rot_eps = 0.0001
    #     return 1.0 / (torch.abs(rot_dist) + rot_eps)

    def _reward_obj_pos_rot_combined(self):
        """
        按照论文公式的组合位置和姿态奖励
        """
        # 位置误差
        pos_diff = torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)**2
        
        # 姿态误差
        quat_diff = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
        # rot_diff =2 * torch.acos(torch.clamp(torch.abs(quat_diff[:, 0]), 0, 1))
        rot_diff = 2.0 * torch.asin(
            torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
        )
        # 组合在exp内
        m = 1.0
        n = 0.5
        combined = m * pos_diff + n * rot_diff
        reward = torch.exp(-combined)
        
        self.extras["pos_diff"] = pos_diff.mean().item()
        self.extras["rot_diff"] = rot_diff.mean().item()
        self.extras["combined_reward"] = reward.mean().item()
        
        return reward        
    def _reward_object_vel_penalty(self):
        """
        Penalize object linear and angular velocity to reduce object shaking/oscillation
        """
        # 线速度惩罚
        linear_vel_penalty = torch.norm(self.object_linvel, p=2, dim=-1)
        
        # 角速度惩罚
        angular_vel_penalty = torch.norm(self.object_angvel, p=2, dim=-1)
        
        # 组合惩罚（可以调整权重）
        return linear_vel_penalty + 0.1 * angular_vel_penalty

    def _reward_joint_pose(self):
        """
        Anygrasp手部关节角度奖励: -a_hand ||q - q^target||²
        手部关节角度越接近目标，奖励越高（惩罚越小）
        """
        # 计算关节角度差异
        joint_diff = torch.norm(self.hand_dof_pos[:, self.actuated_dof_indices] - self.hand_goal_dof_pos[:, self.actuated_dof_indices], p=2, dim=1)
        
        # 使用负的平方差作为奖励

        reward = joint_diff**2
        
        self.extras["joint_diff"] = joint_diff.mean().item()
        
        return reward     

    def _reward_work_penalty(self):
        """功的惩罚项：-a_work||q^T||τ||"""
        joint_velocities = self.hand_dof_vel[:, self.actuated_dof_indices]
        joint_torques = self.dof_force_tensor[:, self.actuated_dof_indices]
        
        power_per_joint = joint_torques * joint_velocities

        work_term = torch.norm(power_per_joint, p=2, dim=-1)
        

        return work_term
    # def _reward_reorienttask_obj_dist(self):
    #     """
    #         基于物体与目标位置之间的距离给予奖励
    #         (距离越小奖励越高)
    #     """
    #     return torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)

    # def _reward_reorienttask_obj_rot(self):
    #     """
    #         计算手中立方体与目标立方体的朝向对齐度
    #         (使用四元数差异衡量旋转偏差)
    #     """
    #     quat_diff = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
    #     # rot_dist = 2.0 * torch.asin(
    #     #     torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
    #     # )
    #     rot_dist = 2 * torch.acos(torch.clamp(torch.abs(quat_diff[:, 0]), 0, 1))
                
    #     rot_eps = 0.0001
    #     return 1.0 / (torch.abs(rot_dist) + rot_eps)


    # # 手部关节角与目标关节角差距奖励
    # def _reward_joint_angle_alignment(self):
    #     """
    #     计算当前关节角与目标关节角的对齐度（基于角度差）
    #     """
    #     # # 获取当前与目标关节角（假设单位是弧度）
    #     # current_angles = self.hand_dof_pos  # [batch_size, n_joints]
    #     # target_angles = self.hand_goal_dof_pos
        
    #     # # 计算角度差（考虑周期性，如-π和π实际相同）
    #     # angle_diff = torch.atan2(
    #     #     torch.sin(current_angles - target_angles),
    #     #     torch.cos(current_angles - target_angles)
    #     # )  # 范围[-π, π]
        
    #     # # # 归一化到[-1,1]（可选）
    #     # # normalized_diff = angle_diff / (self.hand_dof_upper_limits - self.hand_dof_lower_limits + 1e-6)
        
    #     # # # 奖励设计：差异越小奖励越高
    #     # # angle_eps = 0.0001  # 避免除零
    #     # # return 1.0 / (torch.norm(joint_normalized_diff, p=2, dim=1) + angle_eps)  # 对所有关节取平均    
    #     # # 方案1：简单归一化到[0,1]范围，π表示最大差异
    #     # normalized_diff = torch.abs(angle_diff) / torch.pi
    #     joint_diff = self.calculate_joint_distance_pytorch_batch(self.hand_dof_pos, self.hand_goal_dof_pos)
    #     # 采用指数衰减奖励，避免过度激励
    #     # 定义一个小的epsilon值，避免除以零，并使奖励在接近目标时非常大
    #     # joint_eps = 0.0001 # 可以根据实际效果调整此值
        
    #     # # 奖励形式：1.0 / (距离 + epsilon)，距离越小奖励越大
    #     # reward = 1.0 / (joint_diff + joint_eps)
    #     #reward = torch.exp(-5.0 * torch.mean(normalized_diff, dim=1))
    #     # reward = torch.exp(-1.0 * joint_diff)
    #     reward = 1.0 / (1.0 + joint_diff * 5.0)  # 更平滑的反比例函数
    #     self.extras["joint_diff_penalty"] = joint_diff.mean().item()
    #     return reward        



    # ---------------------------------------------------------------------
    # 定义球体旋转任务专用的奖励函数

    def _reward_rottask_obj_xrotvel(self):
        """
        reward the rotational velocity in the X axis of the object
        numerically computed velocity is used to avoid instability from isaacgym
        
        note: if the reward tapers down to 0 (instead of keep going negative like in this implementation), the agent tends to oscillate the ball instead of rotating
        (why? because it rotates in the desired direction at desired speed, then rotates back with a quick motion in a short amount of time)

        This reward was used to generate the motions used in TEDx demo, but it may be more stable to set a goal object with that slowly rotates, like in "Circus ANYmal" paper
        """
        rotvel = self.object_angvel_numerical
        # give max reward when rotvel is between -1 and -2
        # optionally flip this sign back to positive for the
        # ablation study
        direction = - self.cfg["env"]["x_rotation_dir"]
        a = direction * rotvel[:, 0] + 1
        b = torch.ones_like(a) * 2
        c = rotvel[:, 0] + 4
        # return the smallest of the three
        return torch.min(torch.min(a, b), c)

    # ------------- observation functions -----------------
    # the observation functions below may not be called depending on the
    # reward configuration, so they must not contain computation that is
    # used in other functions i.e. they should only compute the observation
    # and nothing else.



    def _observation_dof_position(self):
        """
        Returns the position of actuated DoFs in the hand, scaled to [-1, 1]
        """
        return unscale(
            self.hand_dof_pos[:, self.actuated_dof_indices],
            self.actuated_dof_lower_limits,
            self.actuated_dof_upper_limits,
        )
    
    def _observation_goal_hand_pose(self):
        """
        返回目标手势的关节角度
        """
        return unscale(
            self.hand_goal_dof_pos[:, self.actuated_dof_indices],
            self.actuated_dof_lower_limits,
            self.actuated_dof_upper_limits,
        )
    def _observation_goal_hand_pose_diff(self):
        """
        返回当前手势与目标手势之间的关节角度差异
        """
        # return torch.norm(self.hand_dof_pos[:, self.actuated_dof_indices] - self.hand_goal_dof_pos[:, self.actuated_dof_indices], p=2, dim=1)       
        return unscale(
            self.hand_dof_pos[:, self.actuated_dof_indices] - self.hand_goal_dof_pos[:, self.actuated_dof_indices],
            self.actuated_dof_lower_limits,
            self.actuated_dof_upper_limits,
        )    
    def _observation_dof_pos_target(self):
        """
        position control target of the joints, scaled to [-1, 1]
        recommended when using relative control (as the policy won't know the joint control target otherwise)
        """
        dof_pos_target = self.cur_targets[:, self.actuated_dof_indices]
        dof_pos_target_normalized = unscale(
            dof_pos_target,
            self.actuated_dof_lower_limits,
            self.actuated_dof_upper_limits,
        )
        return dof_pos_target_normalized
    
    def _observation_obj_type(self):
        """
        Returns object type
        """
        return self.object_type
        
    def _observation_obj_pose_history(self):
        """
        Returns the history of object poses (7 DoF) (first element is the
        current pose, so using this with obj pos/quat is redundant)
        """
        return self.obj_pose_buffer
        
    def _observation_dof_pos_history(self):
        """
        Returns the history of joint positions, scaled to [-1, 1]
        (rightmost element is the current position, so using this with dof_pos is redundant)
        """
        return self.dof_pos_buffer
    
    def _observation_dof_speed(self):
        """
        Returns the speed of the actuated DoFs in the hand
        """
        return unscale(
            self.hand_dof_vel[:, self.actuated_dof_indices],
            self.actuated_dof_lower_limits,
            self.actuated_dof_upper_limits,
        )

    def _observation_dof_speed_numerical(self):
        """
        speed of the actuated DoFs in the hand computed numerically
        """
        return unscale(self.hand_dof_vel_numerical,
                       self.actuated_dof_lower_limits,
                       self.actuated_dof_upper_limits)

    def _observation_dof_force(self):
        """
        Returns the forces/torques measured in each DoF
        """
        return self.dof_force_tensor[:, self.actuated_dof_indices]

    def _observation_obj_pos(self):
        """
        Returns the observed object pos in the env's
        coordinate system (TODO wrt hand base)
        """
        obj_pos = self.object_pos.clone()
        obj_pos -= self.object_init_states[:, :3]
        return obj_pos

    def _observation_obj_quat(self):
        """
        Returns the observed object orientation in the env's
        coordinate system (TODO wrt hand base) represented by
        a quaternion
        """
        obj_quat = self.object_rot.clone()
        return obj_quat

    def _observation_obj_linvel(self):
        """
        Returns the linear velocity of the manipulated
        object
        """
        return self.object_linvel

    def _observation_obj_angvel(self):
        """
        Returns the angular velocity of the manipulated
        object
        """
        return self.object_angvel

    def _observation_obj_linvel_numerical(self):
        """
        the linear velocity computed numerically with finite differences
        """
        return self.object_linvel_numerical

    def _observation_obj_angvel_numerical(self):
        """
        the angular velocity computed numerically with finite differences
        """
        return self.object_angvel_numerical

    def _observation_goal_pos(self):
        """
        Returns the goal object position
        """
        goal_pos = self.goal_pos.clone()
        goal_pos -= self.object_init_states[:, :3]
        return goal_pos

    def _observation_goal_quat(self):
        """
        Returns the goal object orientation, represented by 
        a quaternion
        """
        goal_quat = self.goal_rot.clone()
        return goal_quat

    def _observation_goal_quat_diff(self):
        """
        Returns the difference in rotation between the
        current and the goal quaternion
        """
        return quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
    
      
    
    def _observation_pose_sensor_pos(self):
        """
        Returns the pose_sensor position for each pose_sensor
        """
        pose_sensor_pos = self.pose_sensor_state.clone()[:, :, :3]
        pose_sensor_pos -= self.object_init_states[:, :3].unsqueeze(1)
        return pose_sensor_pos.reshape(self.num_envs, -1)

    def _observation_pose_sensor_quat(self):
        """
        Returns the orientation for each pose_sensor, represented
        by a quaternion
        """
        pose_sensor_quats = self.pose_sensor_state.clone()[:, :, 3:7]
        return pose_sensor_quats.reshape(self.num_envs, -1)

    def _observation_pose_sensor_linvel(self):
        """
        Returns the linear velocity of each pose_sensor
        """
        pose_sensor_linvel = self.pose_sensor_state.clone()[:, :, 7:10]
        return pose_sensor_linvel.reshape(self.num_envs, -1)

    def _observation_pose_sensor_angvel(self):
        """
        Returns the 13-DoF full state (pos, quat, linvel, angvel)
        of each pose_sensor
        """
        pose_sensor_angvel = self.pose_sensor_state.clone()[:, :, 10:13]
        return pose_sensor_angvel.reshape(self.num_envs, -1)

    def _observation_force_sensor_force(self):
        """
        Returns 6 DoF force + torque measurements from
        the proxims
        """
        
        return self.vec_sensor_tensor

    def _observation_actions(self):
        """
        Returns the latest actions from the policy
        """
        
        return self.actions
    

    @staticmethod
    def _euler_to_quat_batch(euler):
        """批量欧拉角转四元数"""
        roll, pitch, yaw = euler.unbind(-1)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        # 组合为四元数张量 [N, 4] (w,x,y,z)
        quat = torch.stack([x, y, z, w], dim=1)
        
        # 归一化
        quat = quat / torch.norm(quat, dim=1, keepdim=True)
        
        return quat
@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor),
    )
