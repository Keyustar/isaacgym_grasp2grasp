# ddim_guidance_evaluation.py
import os
import argparse
import numpy as np
from typing import Dict, List, Tuple
import json
import time
from envs.hand_isaacgym import HandKinematicModelIsaacGym
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

# 导入你的模型类
from dm_joint import (
    GraspShardDataset, 
    Diffusion, 
    DexGenDiffusionModel
)

class DDIMGuidanceEvaluator:
    def __init__(self, ckpt_path: str, device: str = "cuda:0"):
        """
        DDIM梯度引导评估器
        
        Args:
            ckpt_path: 训练好的模型检查点路径
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载模型检查点
        print(f"加载模型检查点: {ckpt_path}")
        self.ckpt = torch.load(ckpt_path, map_location=self.device)
        self.meta = self.ckpt["meta"]
        
        # 初始化模型
        self.model = DexGenDiffusionModel(
            state_dim=self.meta["state_dim"],
            future_dim=self.meta["future_dim"],
            mode_vocab=self.meta["mode_vocab"]
        ).to(self.device)
        
        self.model.load_state_dict(self.ckpt["model"])
        self.model.eval()
        
        # 初始化扩散过程
        self.diffusion = Diffusion(
            num_timesteps=self.meta["num_timesteps"],
            device=self.device
        )
        
        # 获取归一化参数
        self.normalization = self.ckpt.get("normalization", None)
        if self.normalization is None:
            raise ValueError("检查点中没有归一化参数，无法正确处理数据")
        
        # 解析维度信息
        self.T_future = 2  # 根据代码中的设定
        self.D = self.meta["future_dim"] // self.T_future  # 每个时间步的关节维度
        
        print(f"✓ 模型加载完成")
        print(f"  - 状态维度: {self.meta['state_dim']}")
        print(f"  - 未来动作维度: {self.meta['future_dim']}")
        print(f"  - 未来时间步数: {self.T_future}")
        print(f"  - 关节维度: {self.D}")

    def load_test_dataset(self, shards_dir: str, max_samples: int = None):
        """
        加载测试数据集
        
        Args:
            shards_dir: 数据分片目录
            max_samples: 最大样本数（用于快速测试）
        """
        print(f"加载测试数据集: {shards_dir}")
        self.dataset = GraspShardDataset(shards_dir)
        
        if max_samples is not None and len(self.dataset) > max_samples:
            # 随机采样部分数据用于测试
            indices = np.random.choice(len(self.dataset), max_samples, replace=False)
            self.dataset.samples = [self.dataset.samples[i] for i in indices]
            print(f"随机采样 {max_samples} 个样本用于测试")
        
        print(f"测试集大小: {len(self.dataset)}")
        return self.dataset

    def extract_reference_motion(self, sample, perturbation_type: str = "none", noise_level: float = 0.1):
        """
        从样本中提取参考运动（未来关节角）
        
        Args:
            sample: 数据集样本
            perturbation_type: 扰动类型 ("none", "gaussian", "scaled")
            noise_level: 噪声水平
            
        Returns:
            参考运动张量 [1, future_dim] (已归一化)
        """
        # 获取原始的未来关节角（已归一化）
        reference_motion = sample["x0"].clone()  # [future_dim]
        
        if perturbation_type == "gaussian":
            # 添加高斯噪声
            noise = torch.randn_like(reference_motion) * noise_level
            reference_motion = reference_motion + noise
            
        elif perturbation_type == "scaled":
            # 缩放扰动
            scale_factor = 1.0 + np.random.uniform(-noise_level, noise_level)
            reference_motion = reference_motion * scale_factor
            
        elif perturbation_type == "partial":
            # 只扰动部分维度
            mask = torch.rand_like(reference_motion) < 0.5  # 随机选择50%的维度
            noise = torch.randn_like(reference_motion) * noise_level
            reference_motion = reference_motion + mask.float() * noise
            
        elif perturbation_type == "time_shifted":
            # 时间维度的扰动 - 交换两个时间步
            ref_reshaped = reference_motion.view(self.T_future, self.D)  # [2, D]
            ref_reshaped = torch.flip(ref_reshaped, dims=[0])  # 翻转时间维度
            reference_motion = ref_reshaped.view(-1)  # [future_dim]
            
        return reference_motion.unsqueeze(0).to(self.device)  # [1, future_dim]

    def _denorm_future(self, x: torch.Tensor) -> np.ndarray:
        """反归一化未来动作 [B, future_dim] -> numpy"""
        x_np = x.detach().cpu().numpy()
        mean = self.normalization["future_mean"]
        std = self.normalization["future_std"]
        return x_np * std + mean

    def _metrics(self, pred: torch.Tensor, gt: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
        """
        计算指标（均在归一化空间）
        - mse_to_gt: 与GT x0的MSE
        - l2_to_gt: 与GT的L2范数
        - l2_to_ref: 与参考运动的L2范数
        - motion_dist_to_ref: 使用 diffusion.motion_distance 的时间步L2和
        """
        mse_to_gt = F.mse_loss(pred, gt).item()
        l2_to_gt = torch.norm(pred - gt, p=2).item()
        l2_to_ref = torch.norm(pred - ref, p=2).item()
        motion_dist_to_ref = self.diffusion.motion_distance(pred, ref).mean().item()
        return {
            "mse_to_gt": mse_to_gt,
            "l2_to_gt": l2_to_gt,
            "l2_to_ref": l2_to_ref,
            "motion_dist_to_ref": motion_dist_to_ref
        }

    @torch.no_grad()
    def _sample_without_guidance(self, state: torch.Tensor, mode_id: int, steps: int) -> torch.Tensor:
        return self.diffusion.ddim_sample(
            model=self.model,
            cond=state,
            future_dim=self.meta["future_dim"],
            steps=steps,
            eta=0.0,
            mode_id=mode_id,
            device=self.device
        )

    def _sample_with_guidance(self, state: torch.Tensor, mode_id: int, steps: int,
                              reference_motion: torch.Tensor, guidance_strength: float) -> torch.Tensor:
        # 注意：需要梯度计算引导，所以不能用 no_grad 包裹该函数内部
        samples = self.diffusion.ddim_sample_with_guidance(
            model=self.model,
            cond=state,
            future_dim=self.meta["future_dim"],
            steps=steps,
            eta=0.0,
            mode_id=mode_id,
            device=self.device,
            reference_motion=reference_motion,
            guidance_strength=guidance_strength
        )
        return samples

    def _isaacgym_play(self, ig_model, denorm_traj_np: np.ndarray, refine_per_step: int, steps_per_target: int):
        def expand_18_to_22(q18):
            q22 = np.zeros(22, dtype=np.float32)
            # FF
            q22[1] = q18[0]   # FFJ2
            q22[2] = q18[1]   # FFJ1
            q22[3] = q18[2]   # FFJ0
            # MF
            q22[5] = q18[3]   # MFJ2
            q22[6] = q18[4]   # MFJ1
            q22[7] = q18[5]   # MFJ0
            # RF
            q22[9]  = q18[6]  # RFJ2
            q22[10] = q18[7]  # RFJ1
            q22[11] = q18[8]  # RFJ0
            # LF（注意：跳过 LFJ4，保留 LFJ3）
            q22[13] = q18[9]   # LFJ3
            q22[14] = q18[10]  # LFJ2
            q22[15] = q18[11]  # LFJ1
            q22[16] = q18[12]  # LFJ0
            # TH
            q22[17] = q18[13]  # THJ4
            q22[18] = q18[14]  # THJ3
            q22[19] = q18[15]  # THJ2
            q22[20] = q18[16]  # THJ1
            q22[21] = q18[17]  # THJ0
            return q22        
        
        # 形状统一为 [T, D]
        if denorm_traj_np.ndim == 1:
            denorm_traj_np = denorm_traj_np[None, :]
        T, D = denorm_traj_np.shape

        traj = denorm_traj_np

        # # 你的未来时间步是 2，如果是两步，则线性细分为更多帧以平滑回放
        # if T == 2 and refine_per_step > 1:
        #     a, b = denorm_traj_np[0], denorm_traj_np[1]
        #     alphas = np.linspace(0.0, 1.0, refine_per_step, endpoint=True)
        #     traj = np.stack([(1 - u) * a + u * b for u in alphas], axis=0)
        # else:
        #     traj = denorm_traj_np

        # 回放：使用位置目标控制
        for q in traj:
            ig_model.set_qpos_target(expand_18_to_22(q))
            for _ in range(steps_per_target):
                ig_model.step()

    def evaluate(self,
                 output_dir: str,
                 num_samples: int,
                 ddim_steps: int,
                 guidance_strengths: List[float],
                 perturbation_types: List[str],
                 noise_level: float,
                 single_sample: int = None):
        """
        主评估流程
        """
        os.makedirs(output_dir, exist_ok=True)

        # 选择评估索引
        if single_sample is not None:
            indices = [single_sample]
            num_samples = 1
        else:
            indices = list(range(min(num_samples, len(self.dataset))))

        all_results = {
            "meta": {
                "ddim_steps": ddim_steps,
                "guidance_strengths": guidance_strengths,
                "perturbation_types": perturbation_types,
                "noise_level": noise_level,
                "future_dim": self.meta["future_dim"],
                "state_dim": self.meta["state_dim"]
            },
            "per_sample": [],
            "aggregates": {}
        }

        # 若开启仿真验证，初始化一次手模
        ig_model = None
        if hasattr(self, "args_for_isaacgym") and self.args_for_isaacgym.isaacgym_validate:
            ig_model = HandKinematicModelIsaacGym.build_from_config()

        # 逐样本评估
        for idx in indices:
            raw = self.dataset[idx]  # {'state','x0','mode'} 均为已归一化
            state = raw["state"].unsqueeze(0).to(self.device).float()  # [1, state_dim]
            gt_x0 = raw["x0"].unsqueeze(0).to(self.device).float()     # [1, future_dim]
            mode_id = int(raw["mode"].item()) if "mode" in raw else 0

            sample_dir = os.path.join(output_dir, f"sample_{idx:06d}")
            os.makedirs(sample_dir, exist_ok=True)

            # baseline: 无引导
            with torch.no_grad():
                pred_baseline = self._sample_without_guidance(state, mode_id, steps=ddim_steps)
            metrics_baseline = self._metrics(pred_baseline, gt_x0, gt_x0)  # 参考=GT仅用于数值占位
            # np.save(os.path.join(sample_dir, "pred_baseline_denorm.npy"), self._denorm_future(pred_baseline))
            # np.save(os.path.join(sample_dir, "gt_denorm.npy"), self._denorm_future(gt_x0))

            # 仿真回放（可选）
            if ig_model is not None:
                self._isaacgym_play(
                    ig_model,
                    denorm_traj_np=self._denorm_future(pred_baseline).reshape(-1, self.D),
                    refine_per_step=self.args_for_isaacgym.ig_refine_per_step,
                    steps_per_target=self.args_for_isaacgym.ig_steps_per_target,
                )

            per_sample_entry = {
                "index": idx,
                "baseline": metrics_baseline,
                "guidance": {}
            }

            # 各扰动+引导强度
            for ptype in perturbation_types:
                per_sample_entry["guidance"].setdefault(ptype, {})
                ref = self.extract_reference_motion(raw, perturbation_type=ptype, noise_level=noise_level)  # [1, future_dim]

                for gs in guidance_strengths:
                    # 当gs为0时，效果等同无引导
                    if gs == 0.0:
                        pred = pred_baseline
                    else:
                        pred = self._sample_with_guidance(state, mode_id, steps=ddim_steps,
                                                          reference_motion=ref, guidance_strength=gs)

                    m = self._metrics(pred, gt_x0, ref)
                    per_sample_entry["guidance"][ptype][str(gs)] = m

                    # 保存预测（反归一化），便于后处理
                    # out_name = f"pred_{ptype}_g{gs:.3f}_denorm.npy".replace(".", "p")
                    # np.save(os.path.join(sample_dir, out_name), self._denorm_future(pred))

                    # 仿真回放（可选）
                    if ig_model is not None:
                        self._isaacgym_play(
                            ig_model,
                            denorm_traj_np=self._denorm_future(pred).reshape(-1, self.D),
                            refine_per_step=self.args_for_isaacgym.ig_refine_per_step,
                            steps_per_target=self.args_for_isaacgym.ig_steps_per_target,
                        )

            all_results["per_sample"].append(per_sample_entry)

        # 聚合统计
        def agg(key: str, ptype: str, gs: float) -> float:
            vals = []
            for s in all_results["per_sample"]:
                if ptype in s["guidance"] and str(gs) in s["guidance"][ptype]:
                    vals.append(s["guidance"][ptype][str(gs)][key])
            return float(np.mean(vals)) if len(vals) > 0 else float("nan")

        aggregates = {"baseline": {}}
        base_keys = ["mse_to_gt", "l2_to_gt", "l2_to_ref", "motion_dist_to_ref"]
        for k in base_keys:
            aggregates["baseline"][k] = float(np.mean([s["baseline"][k] for s in all_results["per_sample"]]))

        aggregates["guidance"] = {}
        for ptype in perturbation_types:
            aggregates["guidance"][ptype] = {}
            for gs in guidance_strengths:
                aggregates["guidance"][ptype][str(gs)] = {
                    "mse_to_gt": agg("mse_to_gt", ptype, gs),
                    "l2_to_gt": agg("l2_to_gt", ptype, gs),
                    "l2_to_ref": agg("l2_to_ref", ptype, gs),
                    "motion_dist_to_ref": agg("motion_dist_to_ref", ptype, gs),
                }

        all_results["aggregates"] = aggregates

        # 保存结果
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print("✓ 评估完成，结果已保存到:", output_dir)

def main():
    parser = argparse.ArgumentParser(description="DDIM梯度引导评估")
    parser.add_argument("--ckpt", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据分片目录")
    parser.add_argument("--output_dir", type=str, default="guidance_eval_results", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备")
    
    # 评估参数
    parser.add_argument("--num_samples", type=int, default=50, help="测试样本数量")
    parser.add_argument("--ddim_steps", type=int, default=8, help="DDIM采样步数")
    parser.add_argument("--guidance_strengths", nargs='+', type=float, 
                       default=[0.0, 0.5, 1.0, 2.0, 5.0], help="引导强度列表")
    parser.add_argument("--perturbation_types", nargs='+', type=str,
                       default=["none", "gaussian", "scaled"], help="扰动类型列表")
    parser.add_argument("--noise_level", type=float, default=0.1, help="扰动噪声水平")
    
    # 运行模式
    parser.add_argument("--single_sample", type=int, default=None, help="只评估单个样本的索引")
    parser.add_argument("--isaacgym_validate", action="store_true", help="将预测送入Isaac Gym进行回放验证")
    parser.add_argument("--ig_refine_per_step", type=int, default=30, help="将每个未来时间步细分为多少帧进行回放")
    parser.add_argument("--ig_steps_per_target", type=int, default=2, help="每个目标关节位姿在仿真中推进多少步")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化评估器
    evaluator = DDIMGuidanceEvaluator(args.ckpt, device=args.device)
    
    # 加载测试数据集
    evaluator.load_test_dataset(args.test_data, max_samples=args.num_samples*2)
    
    # 把 isaacgym 相关参数传入评估器
    evaluator.args_for_isaacgym = args
    
    # 运行评估
    evaluator.evaluate(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        ddim_steps=args.ddim_steps,
        guidance_strengths=args.guidance_strengths,
        perturbation_types=args.perturbation_types,
        noise_level=args.noise_level,
        single_sample=args.single_sample
    )

if __name__ == "__main__":
    main()# ddim_guidance_evaluation.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from envs.hand_isaacgym import HandKinematicModelIsaacGym
# 导入你的模型类
from dm_joint import (
    GraspShardDataset, 
    Diffusion, 
    DexGenDiffusionModel
)

class DDIMGuidanceEvaluator:
    def __init__(self, ckpt_path: str, device: str = "cuda:0"):
        """
        DDIM梯度引导评估器
        
        Args:
            ckpt_path: 训练好的模型检查点路径
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载模型检查点
        print(f"加载模型检查点: {ckpt_path}")
        self.ckpt = torch.load(ckpt_path, map_location=self.device)
        self.meta = self.ckpt["meta"]
        
        # 初始化模型
        self.model = DexGenDiffusionModel(
            state_dim=self.meta["state_dim"],
            future_dim=self.meta["future_dim"],
            mode_vocab=self.meta["mode_vocab"]
        ).to(self.device)
        
        self.model.load_state_dict(self.ckpt["model"])
        self.model.eval()
        
        # 初始化扩散过程
        self.diffusion = Diffusion(
            num_timesteps=self.meta["num_timesteps"],
            device=self.device
        )
        
        # 获取归一化参数
        self.normalization = self.ckpt.get("normalization", None)
        if self.normalization is None:
            raise ValueError("检查点中没有归一化参数，无法正确处理数据")
        
        # 解析维度信息
        self.T_future = 2  # 根据代码中的设定
        self.D = self.meta["future_dim"] // self.T_future  # 每个时间步的关节维度
        
        print(f"✓ 模型加载完成")
        print(f"  - 状态维度: {self.meta['state_dim']}")
        print(f"  - 未来动作维度: {self.meta['future_dim']}")
        print(f"  - 未来时间步数: {self.T_future}")
        print(f"  - 关节维度: {self.D}")

    def load_test_dataset(self, shards_dir: str, max_samples: int = None):
        """
        加载测试数据集
        
        Args:
            shards_dir: 数据分片目录
            max_samples: 最大样本数（用于快速测试）
        """
        print(f"加载测试数据集: {shards_dir}")
        self.dataset = GraspShardDataset(shards_dir)
        
        if max_samples is not None and len(self.dataset) > max_samples:
            # 随机采样部分数据用于测试
            indices = np.random.choice(len(self.dataset), max_samples, replace=False)
            self.dataset.samples = [self.dataset.samples[i] for i in indices]
            print(f"随机采样 {max_samples} 个样本用于测试")
        
        print(f"测试集大小: {len(self.dataset)}")
        return self.dataset

    def extract_reference_motion(self, sample, perturbation_type: str = "none", noise_level: float = 0.1):
        """
        从样本中提取参考运动（未来关节角）
        
        Args:
            sample: 数据集样本
            perturbation_type: 扰动类型 ("none", "gaussian", "scaled")
            noise_level: 噪声水平
            
        Returns:
            参考运动张量 [1, future_dim] (已归一化)
        """
        # 获取原始的未来关节角（已归一化）
        reference_motion = sample["x0"].clone()  # [future_dim]
        
        if perturbation_type == "gaussian":
            # 添加高斯噪声
            noise = torch.randn_like(reference_motion) * noise_level
            reference_motion = reference_motion + noise
            
        elif perturbation_type == "scaled":
            # 缩放扰动
            scale_factor = 1.0 + np.random.uniform(-noise_level, noise_level)
            reference_motion = reference_motion * scale_factor
            
        elif perturbation_type == "partial":
            # 只扰动部分维度
            mask = torch.rand_like(reference_motion) < 0.5  # 随机选择50%的维度
            noise = torch.randn_like(reference_motion) * noise_level
            reference_motion = reference_motion + mask.float() * noise
            
        elif perturbation_type == "time_shifted":
            # 时间维度的扰动 - 交换两个时间步
            ref_reshaped = reference_motion.view(self.T_future, self.D)  # [2, D]
            ref_reshaped = torch.flip(ref_reshaped, dims=[0])  # 翻转时间维度
            reference_motion = ref_reshaped.view(-1)  # [future_dim]
            
        return reference_motion.unsqueeze(0).to(self.device)  # [1, future_dim]

    def _denorm_future(self, x: torch.Tensor) -> np.ndarray:
        """反归一化未来动作 [B, future_dim] -> numpy"""
        x_np = x.detach().cpu().numpy()
        mean = self.normalization["future_mean"]
        std = self.normalization["future_std"]
        return x_np * std + mean

    def _metrics(self, pred: torch.Tensor, gt: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
        """
        计算指标（均在归一化空间）
        - mse_to_gt: 与GT x0的MSE
        - l2_to_gt: 与GT的L2范数
        - l2_to_ref: 与参考运动的L2范数
        - motion_dist_to_ref: 使用 diffusion.motion_distance 的时间步L2和
        """
        mse_to_gt = F.mse_loss(pred, gt).item()
        l2_to_gt = torch.norm(pred - gt, p=2).item()
        l2_to_ref = torch.norm(pred - ref, p=2).item()
        motion_dist_to_ref = self.diffusion.motion_distance(pred, ref).mean().item()
        return {
            "mse_to_gt": mse_to_gt,
            "l2_to_gt": l2_to_gt,
            "l2_to_ref": l2_to_ref,
            "motion_dist_to_ref": motion_dist_to_ref
        }

    @torch.no_grad()
    def _sample_without_guidance(self, state: torch.Tensor, mode_id: int, steps: int) -> torch.Tensor:
        return self.diffusion.ddim_sample(
            model=self.model,
            cond=state,
            future_dim=self.meta["future_dim"],
            steps=steps,
            eta=0.0,
            mode_id=mode_id,
            device=self.device
        )

    def _sample_with_guidance(self, state: torch.Tensor, mode_id: int, steps: int,
                              reference_motion: torch.Tensor, guidance_strength: float) -> torch.Tensor:
        # 注意：需要梯度计算引导，所以不能用 no_grad 包裹该函数内部
        samples = self.diffusion.ddim_sample_with_guidance(
            model=self.model,
            cond=state,
            future_dim=self.meta["future_dim"],
            steps=steps,
            eta=0.0,
            mode_id=mode_id,
            device=self.device,
            reference_motion=reference_motion,
            guidance_strength=guidance_strength
        )
        return samples

    def _isaacgym_play(self, ig_model, denorm_traj_np: np.ndarray, refine_per_step: int, steps_per_target: int):
        # 形状统一为 [T, D]
        if denorm_traj_np.ndim == 1:
            denorm_traj_np = denorm_traj_np[None, :]
        T, D = denorm_traj_np.shape

        # 你的未来时间步是 2，如果是两步，则线性细分为更多帧以平滑回放
        if T == 2 and refine_per_step > 1:
            a, b = denorm_traj_np[0], denorm_traj_np[1]
            alphas = np.linspace(0.0, 1.0, refine_per_step, endpoint=True)
            traj = np.stack([(1 - u) * a + u * b for u in alphas], axis=0)
        else:
            traj = denorm_traj_np

        # 回放：使用位置目标控制
        for q in traj:
            ig_model.set_qpos_target(q)
            for _ in range(steps_per_target):
                ig_model.step()

    def evaluate(self,
                 output_dir: str,
                 num_samples: int,
                 ddim_steps: int,
                 guidance_strengths: List[float],
                 perturbation_types: List[str],
                 noise_level: float,
                 single_sample: int = None):
        """
        主评估流程
        """
        os.makedirs(output_dir, exist_ok=True)

        # 选择评估索引
        if single_sample is not None:
            indices = [single_sample]
            num_samples = 1
        else:
            indices = list(range(min(num_samples, len(self.dataset))))

        all_results = {
            "meta": {
                "ddim_steps": ddim_steps,
                "guidance_strengths": guidance_strengths,
                "perturbation_types": perturbation_types,
                "noise_level": noise_level,
                "future_dim": self.meta["future_dim"],
                "state_dim": self.meta["state_dim"]
            },
            "per_sample": [],
            "aggregates": {}
        }

        # 若开启仿真验证，初始化一次手模
        ig_model = None
        if hasattr(self, "args_for_isaacgym") and self.args_for_isaacgym.isaacgym_validate:
            ig_model = HandKinematicModelIsaacGym.build_from_config()

        # 逐样本评估
        for idx in indices:
            raw = self.dataset[idx]  # {'state','x0','mode'} 均为已归一化
            state = raw["state"].unsqueeze(0).to(self.device).float()  # [1, state_dim]
            gt_x0 = raw["x0"].unsqueeze(0).to(self.device).float()     # [1, future_dim]
            mode_id = int(raw["mode"].item()) if "mode" in raw else 0

            sample_dir = os.path.join(output_dir, f"sample_{idx:06d}")
            os.makedirs(sample_dir, exist_ok=True)

            # baseline: 无引导
            with torch.no_grad():
                pred_baseline = self._sample_without_guidance(state, mode_id, steps=ddim_steps)
            metrics_baseline = self._metrics(pred_baseline, gt_x0, gt_x0)  # 参考=GT仅用于数值占位
            # np.save(os.path.join(sample_dir, "pred_baseline_denorm.npy"), self._denorm_future(pred_baseline))
            # np.save(os.path.join(sample_dir, "gt_denorm.npy"), self._denorm_future(gt_x0))

            # 仿真回放（可选）
            if ig_model is not None:
                self._isaacgym_play(
                    ig_model,
                    denorm_traj_np=self._denorm_future(pred_baseline).reshape(-1, self.D),
                    refine_per_step=self.args_for_isaacgym.ig_refine_per_step,
                    steps_per_target=self.args_for_isaacgym.ig_steps_per_target,
                )

            per_sample_entry = {
                "index": idx,
                "baseline": metrics_baseline,
                "guidance": {}
            }

            # 各扰动+引导强度
            for ptype in perturbation_types:
                per_sample_entry["guidance"].setdefault(ptype, {})
                ref = self.extract_reference_motion(raw, perturbation_type=ptype, noise_level=noise_level)  # [1, future_dim]

                for gs in guidance_strengths:
                    # 当gs为0时，效果等同无引导
                    if gs == 0.0:
                        pred = pred_baseline
                    else:
                        pred = self._sample_with_guidance(state, mode_id, steps=ddim_steps,
                                                          reference_motion=ref, guidance_strength=gs)

                    m = self._metrics(pred, gt_x0, ref)
                    per_sample_entry["guidance"][ptype][str(gs)] = m

                    # # 保存预测（反归一化），便于后处理
                    # out_name = f"pred_{ptype}_g{gs:.3f}_denorm.npy".replace(".", "p")
                    # np.save(os.path.join(sample_dir, out_name), self._denorm_future(pred))

                    # 仿真回放（可选）
                    if ig_model is not None:
                        self._isaacgym_play(
                            ig_model,
                            denorm_traj_np=self._denorm_future(pred).reshape(-1, self.D),
                            refine_per_step=self.args_for_isaacgym.ig_refine_per_step,
                            steps_per_target=self.args_for_isaacgym.ig_steps_per_target,
                        )

            all_results["per_sample"].append(per_sample_entry)

        # 聚合统计
        def agg(key: str, ptype: str, gs: float) -> float:
            vals = []
            for s in all_results["per_sample"]:
                if ptype in s["guidance"] and str(gs) in s["guidance"][ptype]:
                    vals.append(s["guidance"][ptype][str(gs)][key])
            return float(np.mean(vals)) if len(vals) > 0 else float("nan")

        aggregates = {"baseline": {}}
        base_keys = ["mse_to_gt", "l2_to_gt", "l2_to_ref", "motion_dist_to_ref"]
        for k in base_keys:
            aggregates["baseline"][k] = float(np.mean([s["baseline"][k] for s in all_results["per_sample"]]))

        aggregates["guidance"] = {}
        for ptype in perturbation_types:
            aggregates["guidance"][ptype] = {}
            for gs in guidance_strengths:
                aggregates["guidance"][ptype][str(gs)] = {
                    "mse_to_gt": agg("mse_to_gt", ptype, gs),
                    "l2_to_gt": agg("l2_to_gt", ptype, gs),
                    "l2_to_ref": agg("l2_to_ref", ptype, gs),
                    "motion_dist_to_ref": agg("motion_dist_to_ref", ptype, gs),
                }

        all_results["aggregates"] = aggregates

        # 保存结果
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print("✓ 评估完成，结果已保存到:", output_dir)

def main():
    parser = argparse.ArgumentParser(description="DDIM梯度引导评估")
    parser.add_argument("--ckpt", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据分片目录")
    parser.add_argument("--output_dir", type=str, default="guidance_eval_results", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备")
    
    # 评估参数
    parser.add_argument("--num_samples", type=int, default=50, help="测试样本数量")
    parser.add_argument("--ddim_steps", type=int, default=8, help="DDIM采样步数")
    parser.add_argument("--guidance_strengths", nargs='+', type=float, 
                       default=[0.0, 0.5, 1.0, 2.0, 5.0], help="引导强度列表")
    parser.add_argument("--perturbation_types", nargs='+', type=str,
                       default=["none", "gaussian", "scaled"], help="扰动类型列表")
    parser.add_argument("--noise_level", type=float, default=0.1, help="扰动噪声水平")
    
    # 运行模式
    parser.add_argument("--single_sample", type=int, default=None, help="只评估单个样本的索引")
    parser.add_argument("--isaacgym_validate", action="store_true", help="将预测送入Isaac Gym进行回放验证")
    parser.add_argument("--ig_refine_per_step", type=int, default=30, help="将每个未来时间步细分为多少帧进行回放")
    parser.add_argument("--ig_steps_per_target", type=int, default=2, help="每个目标关节位姿在仿真中推进多少步")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化评估器
    evaluator = DDIMGuidanceEvaluator(args.ckpt, device=args.device)
    
    # 加载测试数据集
    evaluator.load_test_dataset(args.test_data, max_samples=args.num_samples*2)
    
    # 把 isaacgym 相关参数传入评估器
    evaluator.args_for_isaacgym = args
    
    # 运行评估
    evaluator.evaluate(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        ddim_steps=args.ddim_steps,
        guidance_strengths=args.guidance_strengths,
        perturbation_types=args.perturbation_types,
        noise_level=args.noise_level,
        single_sample=args.single_sample
    )

if __name__ == "__main__":
    main()