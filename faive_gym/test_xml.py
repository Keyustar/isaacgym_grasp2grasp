from isaacgym import gymapi, gymutil

# 初始化基础配置
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z  # 设置Z轴向上
sim_params.gravity = gymapi.Vec3(0, 0, 0)  # 设置重力

# 创建仿真实例
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 检查仿真是否创建成功
if sim is None:
    raise Exception("Failed to create simulation")

# 创建环境边界
env_lower = gymapi.Vec3(-2, -2, 0)
env_upper = gymapi.Vec3(2, 2, 2)
num_per_row = 1

# 创建环境
env = gym.create_env(sim, env_lower, env_upper, num_per_row)

# 加载资产
   #臂
# asset_root = "/home/cky/rm_models/RM65/urdf/rm_65_b_description/urdf"
# asset_file = "rm_65_b_description.xml"
   #手
# asset_root = "/home/cky/faive_gym_oss/assets/open_ai_assets/hand"
# asset_file = "shadow_hand.xml"
   #臂+手
asset_root = "/home/cky/XL_Hand_urdf/urdf"
asset_file = "XL_Hand.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True  
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# 创建actor
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0.2)  # 初始位置
actor = gym.create_actor(env, asset, pose, "robot", 0, 0)

# 定义机械手位置为观察目标
hand_position = gymapi.Vec3(0, 0, 1.0)  # 与上面创建actor时的位置相同

# 创建观察者摄像机
cam_props = gymapi.CameraProperties()
cam_props.horizontal_fov = 100.0
cam_props.width = 1280
cam_props.height = 720

# 首先创建viewer
viewer = gym.create_viewer(sim, cam_props)

# 检查观察者是否创建成功
if viewer is None:
    raise Exception("Failed to create viewer")

# 设置摄像机位置，确保能看到机械手
cam_pos = gymapi.Vec3(0, 1.0, 2.0)  # 摄像机位置
gym.viewer_camera_look_at(viewer, env, hand_position, cam_pos)

# 主仿真循环
while not gym.query_viewer_has_closed(viewer):
    # 仿真步进
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # 更新图形
    gym.step_graphics(sim)
    
    # 如果机械手位置会变动，可以在每帧更新摄像机位置
    # 获取机械手的当前位置
    # actor_pose = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    # new_hand_position = gymapi.Vec3(actor_pose[0].pose.p.x, actor_pose[0].pose.p.y, actor_pose[0].pose.p.z)
    # gym.viewer_camera_look_at(viewer, env, new_hand_position, cam_pos)
    
    # 绘制场景
    gym.draw_viewer(viewer, sim, True)
    
    # 同步帧率
    gym.sync_frame_time(sim)

# 清理资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)