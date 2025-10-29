# diffusion_model/robot_action_diffusion.py
import os
import glob
import math
import argparse
import numpy as np
from typing import List, Tuple, Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# 数据集：读取 our_hand_nowrist_selectgrasps.py 保存的分片
# ----------------------------


class GraspShardDataset(Dataset):
    """
    每个样本结构:
    {
      "history": {
         "fingertip_pos": [4, K, 3],
         "joint_pos":      [4, D],
         "joint_targets":  [4, D],
         "control_error":  [4, D],
      },
      "future_actions": {
         "fingertip_pos": [2, K, 3],
         "joint_pos":     [2, D],
      }
    }
    """
    def __init__(self, shards_dir: str, mode_dim: int = 0):
        self.samples: List[Dict] = []
        shard_paths = sorted(glob.glob(os.path.join(shards_dir, "*.npy")))
        if len(shard_paths) == 0:
            raise FileNotFoundError(f"未找到分片: {shards_dir}/*.npy")

        bad, ok = 0, 0
        for p in shard_paths:
            try:
                # 跳过空/过小文件（常见于中断写入）
                if not os.path.isfile(p) or os.path.getsize(p) < 128:
                    bad += 1
                    print(f"[warn] 跳过可疑分片(过小): {p}")
                    continue
                arr = np.load(p, allow_pickle=True)
                # 兼容单对象保存成 0 维数组的情况
                if getattr(arr, "ndim", None) == 0:
                    try:
                        arr = [arr.item()]
                    except Exception:
                        print(f"[warn] 分片结构异常(0维且不可item): {p}")
                        bad += 1
                        continue
                # 转 list 追加
                self.samples.extend(list(arr))
                ok += 1
            except (EOFError, ValueError, OSError) as e:
                bad += 1
                print(f"[warn] 读取分片失败，已跳过: {p} ({type(e).__name__}: {e})")
        
        if len(self.samples) == 0:
            raise RuntimeError(f"有效样本为 0 (有效分片 {ok}，坏分片 {bad}）。请检查分片目录 {shards_dir}")

        # 推断维度
        s0 = self.samples[0]
        hist = s0["history"]
        fut = s0["future_actions"]
        self.K = int(np.array(hist["fingertip_pos"]).shape[1])
        self.D = int(np.array(hist["joint_pos"]).shape[-1])
        self.T_hist = 4
        self.T_future = 2
        self.mode_dim = mode_dim

        self.state_dim = (
            self.T_hist * (self.K * 3) +
            self.T_hist * self.D +
            self.T_hist * self.D +
            self.T_hist * self.D
        )
        self.future_dim = (
            self.T_future * self.D
        )
        
        print(f"[info] 分片统计: 有效 {ok}，损坏/跳过 {bad}，样本数 {len(self.samples)}")
        
        # 计算数据统计信息
        self._compute_data_stats()

    def _extract_state(self, sample):
        """从样本中提取状态向量"""
        hist = sample["history"]
        
        # 条件状态展平
        h_tip = np.array(hist["fingertip_pos"], dtype=np.float32).reshape(-1)
        h_joint = np.array(hist["joint_pos"], dtype=np.float32).reshape(-1)
        h_target = np.array(hist["joint_targets"], dtype=np.float32).reshape(-1)
        h_err = np.array(hist["control_error"], dtype=np.float32).reshape(-1)
        state = np.concatenate([h_tip, h_joint, h_target, h_err], axis=0)
        
        return state

    def _extract_future(self, sample):
        """从样本中提取未来动作向量"""
        fut = sample["future_actions"]
        
        # 未来动作(监督的 x0)
        f_joint = np.array(fut["joint_pos"], dtype=np.float32).reshape(-1)
        x0 = np.concatenate([f_joint], axis=0)
        
        return x0

    def _compute_data_stats(self):
        print("[info] 计算数据统计信息...")
        total_samples = len(self.samples)
        if total_samples <= 10000:
            # 小数据集：使用全部数据
            sample_size = total_samples
            indices = np.arange(total_samples)
            print(f"[info] 使用全部 {sample_size} 个样本")
        else:
            # 大数据集：分层采样 + 固定种子
            sample_size = min(100000, total_samples // 2)  # 增加样本量
            
            # 🚨 关键：使用固定种子确保可重复性
            old_state = np.random.get_state()
            np.random.seed(42)  # 固定种子
            indices = np.random.choice(total_samples, sample_size, replace=False)
            np.random.set_state(old_state)  # 恢复原来的随机状态
            
            print(f"[info] 从 {total_samples} 个样本中采样 {sample_size} 个 (固定种子)")
        
        # 批量处理避免内存问题
        all_states = []
        all_futures = []
        batch_size = 1000
        failed_samples = 0
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_states = []
            batch_futures = []
            
            for idx in batch_indices:
                try:
                    sample = self.samples[idx]
                    state = self._extract_state(sample)
                    future = self._extract_future(sample)
                    batch_states.append(state)
                    batch_futures.append(future)
                except Exception as e:
                    failed_samples += 1
                    if failed_samples < 10:  # 只打印前几个错误
                        print(f"[warn] 跳过样本 {idx}: {e}")
                    continue
            
            if batch_states:
                all_states.extend(batch_states)
                all_futures.extend(batch_futures)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"[info] 已处理 {i + len(batch_indices)}/{len(indices)} 个样本")
        
        if failed_samples > 10:
            print(f"[warn] 总共跳过了 {failed_samples} 个问题样本")
        
        if len(all_states) == 0:
            raise RuntimeError("无法计算数据统计信息，所有样本都有问题")
        
        # 转为numpy数组
        all_states = np.stack(all_states)
        all_futures = np.stack(all_futures)
        
        print(f"[info] 有效样本数: {len(all_states)}")
        print(f"[info] 原始数据范围:")
        print(f"  状态: [{all_states.min():.3f}, {all_states.max():.3f}]")
        print(f"  未来: [{all_futures.min():.3f}, {all_futures.max():.3f}]")
        
        # 🚨 改进4：更鲁棒的统计计算
        # 计算均值
        self.state_mean = np.mean(all_states, axis=0).astype(np.float32)
        self.future_mean = np.mean(all_futures, axis=0).astype(np.float32)
        self.state_std = (np.std(all_states, axis=0) + 1e-8).astype(np.float32)
        self.future_std = (np.std(all_futures, axis=0) + 1e-8).astype(np.float32)    
        print(f"[info] 数据统计完成，使用 {len(all_states)} 个样本")
        print(f"[info] 状态范围: [{all_states.min():.3f}, {all_states.max():.3f}]")
        print(f"[info] 未来动作范围: [{all_futures.min():.3f}, {all_futures.max():.3f}]")    
    # def _compute_data_stats(self):
    #     """计算数据统计信息用于归一化"""
    #     print("[info] 计算数据统计信息...")
    #     all_states = []
    #     all_futures = []
        
    #     # 采样部分数据计算统计，避免内存爆炸
    #     sample_size = min(5000, len(self.samples))
    #     indices = np.random.choice(len(self.samples), sample_size, replace=False)
        
    #     for idx in indices:
    #         sample = self.samples[idx]
    #         try:
    #             state = self._extract_state(sample)
    #             future = self._extract_future(sample)
    #             all_states.append(state)
    #             all_futures.append(future)
    #         except Exception as e:
    #             print(f"[warn] 跳过样本 {idx}: {e}")
    #             continue
        
    #     if len(all_states) == 0:
    #         raise RuntimeError("无法计算数据统计信息，所有样本都有问题")
        
    #     all_states = np.stack(all_states)
    #     all_futures = np.stack(all_futures)
        
    #     self.state_mean = np.mean(all_states, axis=0).astype(np.float32)
    #     self.state_std = (np.std(all_states, axis=0) + 1e-8).astype(np.float32)
    #     self.future_mean = np.mean(all_futures, axis=0).astype(np.float32)
    #     self.future_std = (np.std(all_futures, axis=0) + 1e-8).astype(np.float32)
        
    #     print(f"[info] 数据统计完成，使用 {len(all_states)} 个样本")
    #     print(f"[info] 状态范围: [{all_states.min():.3f}, {all_states.max():.3f}]")
    #     print(f"[info] 未来动作范围: [{all_futures.min():.3f}, {all_futures.max():.3f}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 使用提取方法
        state = self._extract_state(sample)
        x0 = self._extract_future(sample)

        # 归一化
        state = (state - self.state_mean) / self.state_std
        x0 = (x0 - self.future_mean) / self.future_std

        # 可选: 模式标签（如果没有，返回全 0）
        if "mode" in sample:
            mode = int(sample["mode"])
        else:
            mode = 0

        return {
            "state": torch.from_numpy(state),
            "x0": torch.from_numpy(x0),
            "mode": torch.tensor(mode, dtype=torch.long)
        }
    def denormalize_future(self, normalized_future):
        """将归一化的未来动作还原"""
        if isinstance(normalized_future, torch.Tensor):
            normalized_future = normalized_future.cpu().numpy()
        return normalized_future * self.future_std + self.future_mean

    def normalize_state(self, state):
        """归一化状态（用于推理时）"""
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        return (state - self.state_mean) / self.state_std
# ----------------------------
# 扩散过程（含 DDIM 采样）
# ----------------------------

class Diffusion:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.device = device
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()
        self.sqrt_inv_alphas = (1.0 / self.alphas).sqrt()
        self.noise_coef = self.betas / self.sqrt_one_minus_alphas_cumprod
        self.variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x0, noise, t):  # x_t
        s1 = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        return s1 * x0 + s2 * noise
    def motion_distance(self, delta_x, delta_x_input):
        """
        计算预测运动和输入参考运动之间的距离
        delta_x: [B, future_dim] - 预测的未来动作
        delta_x_input: [B, future_dim] - 输入的参考运动命令
        返回: [B] - 每个样本的距离
        """
        # 论文中的公式 (3): Dist(Δx, Δx_input) = Σ ||Δx_i - Δx_input_i||²
        diff = delta_x - delta_x_input  # [B, future_dim]
        
        # 如果是关节位置序列，需要按时间步计算
        # 假设 future_dim = T_future * D，需要重塑为 [B, T_future, D]
        B, total_dim = diff.shape
        T_future = 2  # 根据你的代码，未来有2个时间步
        D = total_dim // T_future  # 每个时间步的关节维度
        
        diff_reshaped = diff.view(B, T_future, D)  # [B, T_future, D]
        
        # 按时间步计算L2距离
        distances = torch.sum(torch.norm(diff_reshaped, p=2, dim=-1), dim=-1)  # [B]
        return distances

    def ddim_sample_with_guidance(self, model, cond, future_dim, steps: int, 
                                 eta: float = 0.0, mode_id: int = 0, device="cpu",
                                 reference_motion=None, guidance_strength=1.0):
        """
        带运动条件引导的DDIM采样
        reference_motion: [B, future_dim] - 参考运动命令 Δx_input
        guidance_strength: float - 引导强度参数 α
        """
        assert steps >= 1
        step_indices = torch.linspace(self.num_timesteps - 1, 0, steps, dtype=torch.long, device=device)
        x = torch.randn(cond.shape[0], future_dim, device=device)
        
        for i, t in enumerate(step_indices):
            t_batch = torch.full((x.size(0),), t.item(), dtype=torch.long, device=device)
            
            # 原始噪声预测
            eps = model(x, t_batch, cond, mode=torch.full_like(t_batch, mode_id))
            
            # 如果提供了参考运动，应用引导
            if reference_motion is not None:
                # 计算当前预测的 x0
                alpha_t = self.alphas_cumprod[t]
                sqrt_alpha_t = alpha_t.sqrt()
                sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()
                x0_pred = (x - sqrt_one_minus_alpha_t * eps) / (sqrt_alpha_t + 1e-8)
                
                # 计算引导梯度
                x0_pred_detached = x0_pred.detach().requires_grad_(True)
                distance = self.motion_distance(x0_pred_detached, reference_motion)
                
                # 计算梯度 ∇_{x0} Dist(x0, reference_motion)
                guidance_grad = torch.autograd.grad(
                    outputs=distance.sum(), 
                    inputs=x0_pred_detached, 
                    create_graph=False
                )[0]
                
                # 将 x0 梯度转换为噪声梯度
                # 由于 x0 = (x - √(1-α_t) * ε) / √α_t
                # 所以 ∇_ε x0 = -√(1-α_t) / √α_t
                eps_guidance_factor = -sqrt_one_minus_alpha_t / (sqrt_alpha_t + 1e-8)
                eps_guidance = guidance_grad * eps_guidance_factor
                
                # 应用引导：调整噪声预测
                eps = eps - guidance_strength * eps_guidance

            # 继续标准DDIM采样流程
            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = alpha_t.sqrt()
            sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()
            x0_pred = (x - sqrt_one_minus_alpha_t * eps) / (sqrt_alpha_t + 1e-8)

            if i == steps - 1:
                x = x0_pred
            else:
                t_next = step_indices[i + 1]
                alpha_next = self.alphas_cumprod[t_next]
                sigma = eta * ((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next)).sqrt()
                dir_xt = (alpha_next.sqrt()) * x0_pred
                noise = sigma * torch.randn_like(x)
                x = dir_xt + (1 - alpha_next - sigma**2).sqrt() * eps + noise
        
        return x


    # DDPM 单步
    def ddpm_p_sample(self, model_output, t_scalar, sample):
        s1 = self.sqrt_inv_alphas[t_scalar].reshape(-1, 1)
        s2 = self.noise_coef[t_scalar].reshape(-1, 1)
        s3 = self.variance[t_scalar].reshape(-1, 1).sqrt()
        noise = torch.randn_like(model_output)
        return s1 * (sample - s2 * model_output) + s3 * noise

    # DDIM 采样（eta=0 为确定性）
    def ddim_sample(self, model, cond, future_dim, steps: int, eta: float = 0.0, mode_id: int = 0, device="cpu"):
        assert steps >= 1
        step_indices = torch.linspace(self.num_timesteps - 1, 0, steps, dtype=torch.long, device=device)
        x = torch.randn(cond.shape[0], future_dim, device=device)
        for i, t in enumerate(step_indices):
            t_batch = torch.full((x.size(0),), t.item(), dtype=torch.long, device=device)
            eps = model(x, t_batch, cond, mode=torch.full_like(t_batch, mode_id))
            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = alpha_t.sqrt()
            sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()
            # 预测 x0
            x0_pred = (x - sqrt_one_minus_alpha_t * eps) / (sqrt_alpha_t + 1e-8)

            if i == steps - 1:
                x = x0_pred
            else:
                t_next = step_indices[i + 1]
                alpha_next = self.alphas_cumprod[t_next]
                sigma = eta * ((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next)).sqrt()
                dir_xt = (alpha_next.sqrt()) * x0_pred
                noise = sigma * torch.randn_like(x)
                x = dir_xt + (1 - alpha_next - sigma**2).sqrt() * eps + noise
        return x

# ----------------------------
# 模型模块
# ----------------------------

class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale
        half = size // 2
        emb = torch.log(torch.tensor([10000.0])) / (half - 1)
        emb = torch.exp(-emb * torch.arange(half))
        self.register_buffer("emb", emb, persistent=False)

    def forward(self, x: torch.Tensor):
        x = x.float() * self.scale
        if x.dim() == 2 and x.size(1) == 1:
            x = x.squeeze(1)                 # -> [N]
        emb = x.unsqueeze(1) * self.emb.unsqueeze(0)  # [N, half]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class LinearGN(nn.Module):
    """
    线性层 + GroupNorm(group size=8) + GELU
    GroupNorm 期望 (N, C, *)，这里用 (N, C, 1)
    """
    def __init__(self, in_dim, out_dim, groupsize=8, act=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        groups = max(1, out_dim // groupsize)
        self.gn = nn.GroupNorm(num_groups=groups, num_channels=out_dim, eps=1e-6, affine=True)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        y = self.lin(x)
        y = self.gn(y.unsqueeze(-1)).squeeze(-1)
        return self.act(y)

class StateEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 1024, layers: int = 6):
        super().__init__()
        blocks = []
        dim = in_dim
        for i in range(layers - 1):
            blocks.append(LinearGN(dim, hidden))
            dim = hidden
        blocks.append(LinearGN(dim, hidden, act=True))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

class FiLM(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden

    def forward(self, x, gamma, beta):
        # 确保gamma和beta的维度与x匹配
        assert gamma.size(-1) == x.size(-1), f"gamma dim {gamma.size(-1)} != x dim {x.size(-1)}"
        assert beta.size(-1) == x.size(-1), f"beta dim {beta.size(-1)} != x dim {x.size(-1)}"
        
        # FiLM变换：x * (1 + gamma) + beta
        return x * (1 + gamma) + beta


class UNetFC(nn.Module):
    """
    在每一层都使用FiLM的UNet - 编码器、中间层、解码器都有条件调制
    """
    def __init__(self, in_dim: int, hidden: int = 768, out_dim: int = None, cond_dim: int = 768, mode_vocab: int = 1):
        super().__init__()
        self.hidden = hidden
        self.num_layers = 7  # e1, e2, e3, mid, d1, d2, d3
        out_dim = out_dim or in_dim

        # 时间步编码
        self.t_embed = SinusoidalEmbedding(hidden)
        self.t_proj = nn.Linear(hidden, hidden)

        # 模式/任务编码
        self.mode_embed = nn.Embedding(num_embeddings=max(1, mode_vocab), embedding_dim=hidden)

        # 条件投影
        self.cond_proj = nn.Linear(cond_dim, hidden)

        # 网络结构
        self.e1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.e2 = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.e3 = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.mid = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.d1 = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.d2 = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.d3 = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, out_dim)

        # 为每一层创建FiLM模块
        self.film_layers = nn.ModuleList([FiLM(hidden) for _ in range(self.num_layers)])
        
        # FiLM参数生成器 - 为每一层生成gamma和beta
        self.film_gen = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 4),  # 先扩展维度
            nn.GELU(),
            nn.Linear(hidden * 4, hidden * 2 * self.num_layers)  # 输出所有层的gamma+beta
        )

    def forward(self, x, t, cond, mode=None):
        # 条件嵌入
        t_emb = self.t_proj(self.t_embed(t))  # [B, hidden]
        
        if mode is None:
            mode = torch.zeros_like(t)
        m_emb = self.mode_embed(mode)  # [B, hidden]
        
        c_emb = self.cond_proj(cond)  # [B, hidden]

        # 生成所有层的FiLM参数
        film_input = torch.cat([t_emb, m_emb, c_emb], dim=-1)  # [B, hidden*3]
        film_params = self.film_gen(film_input)  # [B, hidden*2*num_layers]
        
        # 重塑为 [B, num_layers, hidden*2] 便于处理
        film_params = film_params.view(-1, self.num_layers, self.hidden * 2)
        
        # 分离每层的gamma和beta
        gammas, betas = [], []
        for i in range(self.num_layers):
            gamma, beta = torch.chunk(film_params[:, i], 2, dim=-1)  # 各自 [B, hidden]
            gammas.append(gamma)
            betas.append(beta)

        # Encoder with FiLM
        s1 = self.e1(x)
        s1 = self.film_layers[0](s1, gammas[0], betas[0])
        
        s2 = self.e2(s1)
        s2 = self.film_layers[1](s2, gammas[1], betas[1])
        
        s3 = self.e3(s2)
        s3 = self.film_layers[2](s3, gammas[2], betas[2])

        # Middle with FiLM
        mid = self.mid(s3)
        mid = self.film_layers[3](mid, gammas[3], betas[3])

        # Decoder with skips and FiLM
        d1 = self.d1(mid) + s3  # 残差连接
        d1 = self.film_layers[4](d1, gammas[4], betas[4])
        
        d2 = self.d2(d1) + s2
        d2 = self.film_layers[5](d2, gammas[5], betas[5])
        
        d3 = self.d3(d2) + s1
        d3 = self.film_layers[6](d3, gammas[6], betas[6])

        return self.out(d3)

class DexGenDiffusionModel(nn.Module):
    """
    总体：StateEncoder + Projection + UNetFC
    """
    def __init__(self, state_dim: int, future_dim: int, mode_vocab: int = 1):
        super().__init__()
        self.state_encoder = StateEncoder(state_dim, hidden=1024, layers=6)
        self.proj = nn.Linear(1024, 768)
        self.unet = UNetFC(in_dim=future_dim, hidden=768, out_dim=future_dim, cond_dim=768, mode_vocab=mode_vocab)

    def forward(self, x, t, state, mode=None):
        s = self.state_encoder(state)
        s = self.proj(s)
        return self.unet(x, t, s, mode=mode)

# ----------------------------
# 训练/评估
# ----------------------------

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dataset = GraspShardDataset(args.shards_dir, mode_dim=args.mode_vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    model = DexGenDiffusionModel(dataset.state_dim, dataset.future_dim, mode_vocab=max(1, args.mode_vocab)).to(device)
    diffusion = Diffusion(num_timesteps=args.num_timesteps, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))

    # 🚨 新增：从检查点恢复训练状态
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    
    if args.resume_ckpt:
        print(f"从检查点恢复训练: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        
        # 加载模型权重
        model.load_state_dict(ckpt["model"])
        print("✓ 模型权重已加载")
        
        # 加载优化器状态
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
            print("✓ 优化器状态已加载")
        
        # 加载学习率调度器状态
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            print("✓ 学习率调度器状态已加载")
        
        # 加载训练进度
        if "training_state" in ckpt:
            training_state = ckpt["training_state"]
            start_epoch = training_state.get("epoch", 0)
            global_step = training_state.get("global_step", 0)
            best_loss = training_state.get("best_loss", float('inf'))
            print(f"✓ 训练状态已加载: epoch={start_epoch}, step={global_step}, best_loss={best_loss:.6f}")
        
        # 检查数据集兼容性
        if "normalization" in ckpt:
            norm_ckpt = ckpt["normalization"]
            norm_current = {
                "state_mean": dataset.state_mean,
                "state_std": dataset.state_std,
                "future_mean": dataset.future_mean,
                "future_std": dataset.future_std,
            }
            
            # 比较归一化参数是否匹配
            for key in norm_ckpt:
                if not np.allclose(norm_ckpt[key], norm_current[key], rtol=1e-4):
                    print(f"⚠️  警告: {key} 不匹配，可能数据集发生了变化")
                    if args.strict_resume:
                        raise ValueError(f"严格模式下 {key} 不匹配，请检查数据集")
            print("✓ 数据归一化参数兼容")

    model.train()
    
    # 🚨 修改：从 start_epoch 开始训练
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        for batch in loader:
            state = batch["state"].to(device)
            x0 = batch["x0"].to(device)
            mode = batch["mode"].to(device)

            noise = torch.randn_like(x0)
            t = torch.randint(0, diffusion.num_timesteps, (x0.size(0),), dtype=torch.long, device=device)
            xt = diffusion.add_noise(x0, noise, t)

            pred_noise = model(xt, t, state, mode=mode)
            loss = F.mse_loss(pred_noise, noise)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            scheduler.step()

            epoch_loss += loss.item()

            if global_step % args.log_interval == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.6f} lr {scheduler.get_last_lr()[0]:.6e}")
                
                if global_step % (args.log_interval * 5) == 0:
                    with torch.no_grad():
                        # pred_x0 = (xt - diffusion.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1) * pred_noise) / \
                        #          (diffusion.sqrt_alphas_cumprod[t].reshape(-1, 1) + 1e-8)
                        alpha_t = diffusion.alphas_cumprod[t].reshape(-1, 1)
                        sqrt_alpha_t = alpha_t.sqrt()
                        sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()
                        
                        # 避免除零，使用更大的epsilon
                        pred_x0 = (xt - sqrt_one_minus_alpha_t * pred_noise) / (sqrt_alpha_t.clamp(min=1e-6))
                        reconstruction_mse = F.mse_loss(pred_x0, x0).item()
                        print(f"  -> 重构MSE: {reconstruction_mse:.6f}")
                        
            global_step += 1
        
        avg_epoch_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} 完成, 平均loss: {avg_epoch_loss:.6f}")
        
        # 判断是否为最佳模型
        save_best_model = avg_epoch_loss < best_loss
        if save_best_model:
            best_loss = avg_epoch_loss

        # 🚨 新增：定期保存检查点 (每 args.save_interval 个epoch)
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                model=model, 
                optimizer=opt, 
                scheduler=scheduler,
                epoch=epoch + 1,  # 下次从这个epoch开始
                global_step=global_step,
                best_loss=best_loss,
                dataset=dataset,
                args=args,
                checkpoint_type="interval"
            )
            print(f"✓ 检查点已保存 (epoch {epoch})")

    # 最终保存包含完整训练状态
    os.makedirs(args.outdir, exist_ok=True)
    save_checkpoint(
        model=model,
        optimizer=opt,
        scheduler=scheduler, 
        epoch=args.epochs,
        global_step=global_step,
        best_loss=best_loss,
        dataset=dataset,
        args=args,
        checkpoint_type="final"
    )
    print(f"✓最终模型已保存,最佳loss: {best_loss:.6f}")


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_loss, dataset, args, checkpoint_type="interval"):
    """保存训练检查点"""
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "meta": {
            "state_dim": dataset.state_dim,
            "future_dim": dataset.future_dim,
            "K": dataset.K,
            "D": dataset.D,
            "mode_vocab": max(1, args.mode_vocab),
            "num_timesteps": args.num_timesteps
        },
        "normalization": {
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
            "future_mean": dataset.future_mean,
            "future_std": dataset.future_std,
        },
        "training_state": {
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
        },
        "hyperparams": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "num_timesteps": args.num_timesteps,
        }
    }
    
    # 根据保存类型决定文件名
    if checkpoint_type == "final":
        if not args.save_name.endswith('.pth'):
            filename = args.save_name + '.pth'
        else:
            filename = args.save_name
    else:  # interval checkpoint
        base_name = args.save_name.replace('.pth', '') if args.save_name.endswith('.pth') else args.save_name
        filename = f"{base_name}_epoch{epoch}.pth"
    
    save_path = os.path.join(args.outdir, filename)
    torch.save(save_dict, save_path)
    return save_path

def infer(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    meta = ckpt["meta"]
    
    model = DexGenDiffusionModel(meta["state_dim"], meta["future_dim"], mode_vocab=meta["mode_vocab"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = Diffusion(num_timesteps=meta["num_timesteps"], device=device)

    # 加载原始状态数据
    state_raw = np.load(args.state_npy).astype(np.float32)
    print(f"原始状态形状: {state_raw.shape}")
    
    # 归一化状态
    if 'normalization' in ckpt:
        norm = ckpt['normalization']
        state_normalized = (state_raw - norm['state_mean']) / norm['state_std']
        print("使用训练时的归一化参数")
    else:
        print("⚠️  警告：检查点中没有归一化参数，使用原始数据")
        state_normalized = state_raw
    
    state = torch.from_numpy(state_normalized).to(device)
    mode_id = int(args.mode_id)

    # 新增：处理参考运动（如果提供）
    reference_motion = None
    if hasattr(args, 'reference_motion_npy') and args.reference_motion_npy:
        ref_raw = np.load(args.reference_motion_npy).astype(np.float32)
        print(f"参考运动形状: {ref_raw.shape}")
        
        if 'normalization' in ckpt:
            ref_normalized = (ref_raw - norm['future_mean']) / norm['future_std']
        else:
            ref_normalized = ref_raw
        reference_motion = torch.from_numpy(ref_normalized).to(device)

    with torch.no_grad():
        # 选择使用引导采样还是标准采样
        if reference_motion is not None and hasattr(args, 'guidance_strength'):
            print(f"使用引导采样，引导强度: {args.guidance_strength}")
            samples = diffusion.ddim_sample_with_guidance(
                model=model,
                cond=state,
                future_dim=meta["future_dim"],
                steps=args.ddim_steps,
                eta=0.0,
                mode_id=mode_id,
                device=device,
                reference_motion=reference_motion,
                guidance_strength=args.guidance_strength
            )
        else:
            print("使用标准采样")
            samples = diffusion.ddim_sample(
                model=model,
                cond=state,
                future_dim=meta["future_dim"],
                steps=args.ddim_steps,
                eta=0.0,
                mode_id=mode_id,
                device=device
            )
    
    # 反归一化预测结果
    samples_np = samples.cpu().numpy()
    if 'normalization' in ckpt:
        norm = ckpt['normalization'] 
        samples_denormalized = samples_np * norm['future_std'] + norm['future_mean']
        print("预测结果已反归一化")
    else:
        samples_denormalized = samples_np
        print("预测结果未反归一化")
    
    np.save(args.out_path, samples_denormalized)
    print(f"生成未来动作已保存: {args.out_path}")
    print(f"预测结果形状: {samples_denormalized.shape}")
    print(f"预测范围: [{samples_denormalized.min():.3f}, {samples_denormalized.max():.3f}]")


def build_argparser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # 训练
    pt = sub.add_parser("train")
    pt.add_argument("--shards_dir", type=str, required=True, help="our_hand_nowrist_selectgrasps.py 保存的分片目录")
    pt.add_argument("--save_name", type=str, default="lr1e-3+timesteps1000.pth", help="模型保存文件名")
    pt.add_argument("--outdir", type=str, default="exps/dm_output")
    pt.add_argument("--device", type=str, default="cuda:0")
    pt.add_argument("--batch_size", type=int, default=64)
    pt.add_argument("--epochs", type=int, default=50)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--num_timesteps", type=int, default=1000)
    pt.add_argument("--log_interval", type=int, default=50)
    pt.add_argument("--mode_vocab", type=int, default=1, help="模式/任务类别数；若无则设 1")
    
    # 新增断点续训相关参数
    pt.add_argument("--resume_ckpt", type=str, default=None, help="要恢复的检查点路径")
    pt.add_argument("--save_interval", type=int, default=1000, help="每隔多少个epoch保存一次检查点")
    pt.add_argument("--strict_resume", action="store_true", help="严格模式：数据归一化参数必须匹配")

    # 推理 (保持不变)
    pi = sub.add_parser("infer")
    pi.add_argument("--ckpt", type=str, required=True)
    pi.add_argument("--state_npy", type=str, required=True)
    pi.add_argument("--out_path", type=str, default="pred_future.npy")
    pi.add_argument("--device", type=str, default="cpu")
    pi.add_argument("--ddim_steps", type=int, choices=[8, 12, 16], default=8)
    pi.add_argument("--mode_id", type=int, default=0)
    pi.add_argument("--guidance_strength", type=float, default=1.0,
                   help="运动引导强度参数")

    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.cmd == "train":
        train(args)
    else:
        infer(args)