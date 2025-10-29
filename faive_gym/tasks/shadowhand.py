from faive_gym.robot_hand import RobotHand
from isaacgymenvs.utils.torch_jit_utils import (
    quat_rotate_inverse,
    quat_conjugate,
    quat_mul,
    quat_to_angle_axis,
)
import torch

"""
not a serious task, just intended as an example of
- how to make a custom task that is derived from the RobotHand class
- how to create environments where the robot hand doesn't have a fixed base

The robot hand learns to crawl on the floor like a horror film - except it doesn't even learn to crawl right now, see if you can make it work...
"""

class ShadowHand_mine(RobotHand):
    
    def _reward_obj_height_penalty(self):
        """
        改进的高度惩罚函数，设定合理高度范围
        """
        # 计算物体相对初始位置的高度变化
        initial_height = 0.02
        current_height = self.object_pos[:, 2]
        height_diff = current_height - initial_height
        
        # 定义合理高度范围
        min_height = -0.02  # 允许略微下降
        max_height = 0.02  # 允许略微上升
        
        # 计算惩罚：超出范围的距离的平方
        too_low = torch.clamp(min_height - height_diff, min=0) ** 2
        too_high = torch.clamp(height_diff - max_height, min=0) ** 2
        
        # 高度越偏离，惩罚越大
        return too_low + too_high
    
    def _reward_contact_force_reward(self):
        # 1. 传感器数据处理
        sensor_forces = self.vec_sensor_tensor.view(self.num_envs, -1, 6)  # [envs, sensors, 6]
        
        # 2. 接触检测
        force_threshold = 0.1
        active_contacts = (torch.norm(sensor_forces[:, :, :3], dim=-1) > force_threshold).sum(dim=-1)
        valid_contact = active_contacts >= 2

        # 3. 动态时间窗口
        max_history = 600
        
        # 4. 更新接触历史（带衰减）
        self.contact_steps = torch.where(
            valid_contact,
            torch.clamp(self.contact_steps + 1, max=max_history),
            torch.clamp(self.contact_steps - 0.8, min=0)
        )
        
        # 5. 奖励计算
        contact_quality = (active_contacts.float() / 6.0) * 0.7  # 质量权重70%
        contact_stability = (self.contact_steps.float() / max_history) * 0.3  # 稳定性权重30%
        
        return contact_quality + contact_stability
    
    # def _reward_object_stability_penalty(self):
    #     """
    #     奖励物体姿态稳定
    #     """
    #     # 惩罚过高的线速度和角速度
    #     lin_vel_penalty = torch.norm(self.object_linvel, dim=1) ** 2
    #     ang_vel_penalty = torch.norm(self.object_angvel, dim=1) ** 2
        
    #     # 速度越低，奖励越高
    #     return torch.exp(2.0 * (lin_vel_penalty + 0.5 * ang_vel_penalty))
    
