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

class ShadowHand_mine_origin(RobotHand):    
    pass
