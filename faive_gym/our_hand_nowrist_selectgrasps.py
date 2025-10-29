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

import os
import atexit
import datetime
import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch

# 注意:基础类名为 nowrist(小写)
from faive_gym.our_hand_nowrist import nowrist


class nowristselect(nowrist):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        # 轨迹收集：每个环境一个列表S
        self._traj_steps = [[] for _ in range(self.num_envs)]
        # 数据集缓冲区：收集完成转换的轨迹X
        self._dataset_buffer = []
        self._shard_size = 50000
        self._shard_count = 0
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_dataset_shards")
        os.makedirs(self._save_dir, exist_ok=True)
        self._run_id = ts
        # 退出时保存残留缓冲
        atexit.register(self._flush_dataset_buffer)

    def post_physics_step(self):
        # 先运行父类的物理与奖励、终止判定
        super().post_physics_step()
        # 每步记录当前抓取手势数据（所有手关节角 + 物体位置）
        self._append_step_records()
        # 在完成姿态转换成功的环境中归档当前轨迹并分片保存
        self._archive_success_trajectories()

    def reset_idx(self, env_ids, goal_env_ids):
        # 先调用父类重置
        super().reset_idx(env_ids, goal_env_ids)
        # 清空被重置环境的S
        if env_ids is not None and len(env_ids) > 0:
            if isinstance(env_ids, torch.Tensor):
                ids = env_ids.tolist()
            else:
                ids = list(env_ids)
            for idx in ids:
                if 0 <= idx < self.num_envs:
                    self._traj_steps[idx] = []

    def _append_step_records(self):
        # hand_dof_pos: [num_envs, num_hand_dofs]
        # object_pos: [num_envs, 3]
        # 将当前步的每个环境数据压入对应S
        # 为减少CPU/GPU同步开销，尽量一次性取出，再逐环境to(cpu)
        fingertip_pos_step = self.pose_sensor_state[:, :, 0:3].detach()
        hand_dof_pos_step = self.hand_dof_pos[:, self.actuated_dof_indices].detach()
        hand_dof_pos_target_step = self.cur_targets[:, self.actuated_dof_indices].detach()
        # object_pos_step = self.object_pos.detach()
        # object_rot_step = self.object_rot.detach()
        control_error = (self.cur_targets[:, self.actuated_dof_indices] - 
                 self.hand_dof_pos[:, self.actuated_dof_indices]).detach()
        
        
        # 若为GPU Tensor，延迟转为CPU在循环中逐个env转换，避免一次性大张量拷回
        for env_idx in range(self.num_envs):
            fingertip_list = fingertip_pos_step[env_idx].to("cpu", non_blocking=True).tolist()
            hand_list = hand_dof_pos_step[env_idx].to("cpu", non_blocking=True).tolist()
            hand_target_list = hand_dof_pos_target_step[env_idx].to("cpu", non_blocking=True).tolist()
            # obj_pos_list = object_pos_step[env_idx].to("cpu", non_blocking=True).tolist()
            # obj_rot_list = object_rot_step[env_idx].to("cpu", non_blocking=True).tolist()
            control_error_list = control_error[env_idx].to("cpu", non_blocking=True).tolist()
            self._traj_steps[env_idx].append({
                "fingertip_pos": fingertip_list,
                "hand_dof_pos": hand_list,
                "hand_dof_pos_target" : hand_target_list,
                # "object_pos": obj_pos_list,
                # "object_rot": obj_rot_list,
                "control_error": control_error_list
            })
    def _stack_history_data(self, history_data):
        """堆叠4步历史数据"""
        stacked = {
            "fingertip_pos": [],  # [4, K, 3]
            "joint_pos": [],      # [4, num_dofs]
            "joint_targets": [],  # [4, num_dofs]
            "control_error": []   # [4, num_dofs]
        }
        
        for step in history_data:
            stacked["fingertip_pos"].append(step["fingertip_pos"])
            stacked["joint_pos"].append(step["hand_dof_pos"])
            stacked["joint_targets"].append(step["hand_dof_pos_target"])
            stacked["control_error"].append(step["control_error"])
        
        return stacked

    def _extract_future_actions(self, future_actions):
        """提取未来2步动作"""
        actions = {            
            "fingertip_pos": [],  # [4, K, 3]
            "joint_pos": []
        }
        for step in future_actions:
            # 这里需要根据你的动作空间定义来提取
            # 假设动作是关节目标位置的变化
            actions["fingertip_pos"].append(step["fingertip_pos"])
            actions["joint_pos"].append(step["hand_dof_pos"])

        return actions      

    def create_diffusion_training_data(self, trajectory_data, history_steps=4, future_steps=2):
        """
        将轨迹数据重组为扩散模型训练格式
        trajectory_data: 单条轨迹的步骤列表
        """
        training_samples = []
        
        # 滑动窗口生成训练样本
        for i in range(len(trajectory_data) - history_steps - future_steps + 1):
            # 历史4步数据
            history_data = trajectory_data[i:i+history_steps]
            
            # 未来2步动作目标
            future_actions = trajectory_data[i+history_steps:i+history_steps+future_steps]
            
            # 重组为扩散模型输入格式
            sample = {
                "history": self._stack_history_data(history_data),
                "future_actions": self._extract_future_actions(future_actions),
                #"current_state": trajectory_data[i+history_steps-1]  # 当前状态
            }
            training_samples.append(sample)
        
        return training_samples


    def _archive_success_trajectories(self):
        if self.success_buf is None:
            return
        success_envs = torch.nonzero(self.success_buf, as_tuple=False).flatten()
        if success_envs.numel() == 0:
            return
        
        for env_idx_t in success_envs:
            env_idx = int(env_idx_t.item())
            traj = self._traj_steps[env_idx]
            if len(traj) > 0:
                # 重组为扩散模型训练格式
                diffusion_samples = self.create_diffusion_training_data(traj)
                if len(diffusion_samples) > 0:
                    self._dataset_buffer.extend(diffusion_samples)
                self._traj_steps[env_idx] = []
        
        if len(self._dataset_buffer) >= self._shard_size:
            self._save_shard()



    # def _archive_success_trajectories(self):
    #     # self.success_buf: [num_envs]，True表示该步完成姿态转换
    #     if self.success_buf is None:
    #         return
    #     success_envs = torch.nonzero(self.success_buf, as_tuple=False).flatten()
    #     if success_envs.numel() == 0:
    #         return
    #     for env_idx_t in success_envs:
    #         env_idx = int(env_idx_t.item())
    #         traj = self._traj_steps[env_idx]
    #         if len(traj) > 0:
    #             # 放入数据集缓冲区，并清空该环境S
    #             self._dataset_buffer.append(traj)
    #             self._traj_steps[env_idx] = []
    #     # 分片保存
    #     if len(self._dataset_buffer) >= self._shard_size:
    #         self._save_shard()

    def _save_shard(self):
        # 保存一个分片，文件名包含run_id与递增计数
        shard_name = f"grasp_traj_{self._run_id}_shard_{self._shard_count:05d}.npy"
        shard_path = os.path.join(self._save_dir, shard_name)
        # 使用np.save，允许pickle
        count = len(self._dataset_buffer)
        np.save(shard_path, np.array(self._dataset_buffer, dtype=object), allow_pickle=True)
        self._shard_count += 1
        self._dataset_buffer = []
        print(f"数据已保存至: {shard_path}(轨迹数: {count})")

    def _flush_dataset_buffer(self):
        # 退出时把剩余未满分片的数据也保存
        if len(self._dataset_buffer) > 0:
            self._save_shard()