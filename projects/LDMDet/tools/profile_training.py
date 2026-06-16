"""LDMDet 训练性能 Profile 脚本

使用 PyTorch Profiler 对训练流程进行细粒度性能分析，
定位训练速度瓶颈。

用法:
    python LDMDet/tools/profile_training.py \
        LDMDet/configs/ldmdet_rf_heun_shifted_bs8.py \
        --num-warmup 3 --num-profile 5

输出:
    - 终端打印: 各阶段耗时统计 + GPU Kernel 分布
    - TensorBoard 日志: ./profiler_logs/ (可用 tensorboard --logdir 查看)
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from mmengine.config import Config
from mmengine.runner import Runner


# ============================================================
# 计时工具
# ============================================================

class TimingStats:
    """累积计时统计"""

    def __init__(self):
        self._timings = defaultdict(list)

    def record(self, name: str, elapsed: float):
        self._timings[name].append(elapsed)

    def summary(self) -> str:
        lines = []
        lines.append(f"{'Stage':<40} {'Mean(ms)':>10} {'Std(ms)':>10} {'Total(ms)':>12} {'Count':>6} {'Pct':>6}")
        lines.append("-" * 90)

        total_time = sum(
            sum(v) for v in self._timings.values()
        )

        # 按总耗时排序
        sorted_items = sorted(
            self._timings.items(), key=lambda x: sum(x[1]), reverse=True
        )

        for name, times in sorted_items:
            import numpy as np
            arr = np.array(times) * 1000  # to ms
            pct = sum(times) / max(total_time, 1e-10) * 100
            lines.append(
                f"{name:<40} {arr.mean():>10.2f} {arr.std():>10.2f} "
                f"{arr.sum():>12.2f} {len(arr):>6d} {pct:>5.1f}%"
            )

        lines.append("-" * 90)
        lines.append(f"{'TOTAL':<40} {'':>10} {'':>10} {total_time*1000:>12.2f}")
        return "\n".join(lines)


@contextmanager
def timer(stats: TimingStats, name: str, cuda: bool = True):
    """计时上下文管理器"""
    if cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    stats.record(name, elapsed)


# ============================================================
# 模型级细粒度 Profile
# ============================================================

def profile_model_forward(model, batch_inputs, batch_data_samples, stats, device):
    """对模型前向传播进行细粒度计时"""
    from projects.LDMDet.mods.structures import ImageMeta

    # 1. Backbone + Neck
    with timer(stats, "1. extract_feat (backbone+neck)"):
        features = model.extract_feat(batch_inputs)

    # 2. 准备数据
    with timer(stats, "2. data preparation (img_metas, gt)"):
        img_metas = []
        gt_bboxes = []
        gt_labels = []
        for data_sample in batch_data_samples:
            meta = ImageMeta(
                img_shape=data_sample.metainfo['img_shape'],
                pad_shape=data_sample.metainfo.get('pad_shape'),
                ori_shape=data_sample.metainfo.get('ori_shape'),
                scale_factor=data_sample.metainfo.get('scale_factor'),
            )
            img_metas.append(meta)
            gt_bboxes.append(data_sample.gt_instances.bboxes)
            gt_labels.append(data_sample.gt_instances.labels)

    bbox_head = model.bbox_head

    # 3. Target normalization
    with timer(stats, "3. _normalize_targets"):
        targets = bbox_head._normalize_targets(gt_bboxes, gt_labels, img_metas, len(img_metas))

    # 4. Time step sampling
    with timer(stats, "4. _sample_t"):
        t = bbox_head._sample_t(len(img_metas), device)

    # 5. Build training targets (coupling + forward diffusion)
    with timer(stats, "5. _build_training_targets (coupling+diffusion)"):
        x_boxes, x_starts, x_noises, matched_gt_indices = bbox_head._build_training_targets(
            len(img_metas), device, t, targets, gt_bboxes, img_metas
        )

    # 细分 OT coupling
    if bbox_head.ot_coupling:
        with timer(stats, "5a. OT coupling (per-image)"):
            for i in range(len(img_metas)):
                num_gt = gt_bboxes[i].shape[0]
                if num_gt == 0:
                    continue
                from projects.LDMDet.mods.utils import bbox_xyxy_to_cxcywh
                norm_gt_cxcywh = bbox_xyxy_to_cxcywh(targets[i].bboxes)
                gt_diffusion = (norm_gt_cxcywh * 2 - 1) * bbox_head.snr_scale
                noise = torch.randn(bbox_head.num_proposals, 4, device=device)
                with timer(stats, "5a1. OT sinkhorn"):
                    bbox_head.ot_module.couple(noise, gt_diffusion, targets[i].labels, device)

    x_noisy_batch = torch.stack(x_boxes)
    with timer(stats, "5b. raw_to_xyxy"):
        curr_bboxes = bbox_head._sampler.raw_to_xyxy(x_noisy_batch, img_metas)

    # 6. Forward pass through head series
    t_input = t if bbox_head.diffusion_type == 'ddpm' else t * bbox_head.timesteps

    with timer(stats, "6. bbox_head.forward (all heads)"):
        time_emb = bbox_head.time_mlp(t_input)

        inter_cls_logits = []
        inter_pred_bboxes = []
        curr_bboxes_fwd = curr_bboxes
        curr_proposals = None

        for head_idx, head in enumerate(bbox_head.head_series):
            with timer(stats, f"6a. head[{head_idx}] total"):
                # ROI extraction
                from projects.LDMDet.mods.utils import bbox2roi
                with timer(stats, f"6a1. head[{head_idx}] bbox2roi"):
                    rois = bbox2roi([curr_bboxes_fwd[i] for i in range(len(img_metas))])

                with timer(stats, f"6a2. head[{head_idx}] roi_extractor"):
                    roi_features = bbox_head.roi_extractor(features, rois)

                bs = len(img_metas)
                num_boxes = curr_bboxes_fwd.shape[1]

                if curr_proposals is None:
                    curr_proposals = (
                        roi_features.flatten(2)
                        .mean(-1)
                        .view(bs, num_boxes, head.feat_channels)
                    )

                roi_features = roi_features.view(
                    bs * num_boxes, head.feat_channels, -1
                ).permute(2, 0, 1)

                # Conditioned forward
                with timer(stats, f"6a3. head[{head_idx}] _conditioned_forward"):
                    fc_feature = head._conditioned_forward(
                        curr_proposals, roi_features, time_emb, bs, num_boxes
                    )

                # Predict
                with timer(stats, f"6a4. head[{head_idx}] _predict"):
                    cls_logits, pred_bboxes, curr_proposals = head._predict(
                        fc_feature, curr_bboxes_fwd, bs, num_boxes
                    )

                inter_cls_logits.append(cls_logits)
                inter_pred_bboxes.append(pred_bboxes)
                curr_bboxes_fwd = pred_bboxes.detach()

    all_cls_logits = torch.stack(inter_cls_logits)
    all_pred_bboxes = torch.stack(inter_pred_bboxes)

    # 7. Normalize pred bboxes
    with timer(stats, "7. _normalize_pred_bboxes"):
        norm_pred_bboxes = bbox_head._normalize_pred_bboxes(all_pred_bboxes, img_metas)

    # 8. Build outputs
    with timer(stats, "8. _build_outputs"):
        outputs = bbox_head._build_outputs(all_cls_logits, norm_pred_bboxes)

    # 9. Criterion (matcher + loss)
    with timer(stats, "9. criterion (matcher+loss)"):
        losses = bbox_head.criterion(outputs, targets)

    # 细分 matcher
    with timer(stats, "9a. matcher"):
        indices = bbox_head.criterion.matcher(outputs, targets)

    # 10. Auxiliary losses
    x_starts_batch = torch.stack(x_starts)
    x_noises_batch = torch.stack(x_noises)

    if bbox_head.use_velocity_loss and bbox_head.diffusion_type == 'rectified_flow':
        with timer(stats, "10. velocity_loss"):
            bbox_head._add_velocity_loss(
                all_pred_bboxes, x_starts_batch, x_noises_batch,
                t, img_metas,
            )

    if bbox_head.use_cat:
        with timer(stats, "10. cat_loss"):
            bbox_head._add_cat_loss(
                features, all_pred_bboxes, x_starts_batch, x_noises_batch,
                t, img_metas,
            )

    if bbox_head.use_trd:
        with timer(stats, "10. trd_loss"):
            bbox_head._add_trd_loss(
                features, all_pred_bboxes, x_starts_batch, x_noises_batch,
                t, img_metas,
            )

    return losses


# ============================================================
# 主 Profile 流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Profile LDMDet training")
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--num-warmup", type=int, default=3,
        help="warmup iterations (not profiled)",
    )
    parser.add_argument(
        "--num-profile", type=int, default=5,
        help="profiled iterations",
    )
    parser.add_argument(
        "--use-tb-profiler", action="store_true",
        help="enable PyTorch TensorBoard Profiler (detailed kernel trace)",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device id",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # ---- 构建模型和数据 ----
    print("=" * 60)
    print("LDMDet Training Profiler")
    print("=" * 60)

    cfg = Config.fromfile(args.config)
    # 禁用 SwanLab 等可视化后端以加速
    if hasattr(cfg, 'visualizer'):
        cfg.visualizer.vis_backends = [dict(type='LocalVisBackend')]

    # 减少到 1 epoch 用于 profile
    if hasattr(cfg, 'train_cfg'):
        cfg.train_cfg.max_epochs = 1

    print(f"\nConfig: {args.config}")
    print(f"Warmup iterations: {args.num_warmup}")
    print(f"Profile iterations: {args.num_profile}")
    print(f"TensorBoard Profiler: {args.use_tb_profiler}")

    from mmengine.runner import Runner
    runner = Runner.from_cfg(cfg)

    # 获取模型和数据加载器
    model = runner.model
    if hasattr(model, 'module'):
        model = model.module
    model.eval()  # 先 eval 避免BN等影响，后面切回train
    model.train()

    device = next(model.parameters()).device
    print(f"Device: {device}")

    # 获取一个 batch 的数据
    dataloader = runner.train_dataloader
    data_iter = iter(dataloader)

    # ---- Warmup ----
    print(f"\n--- Warmup ({args.num_warmup} iters) ---")
    for i in range(args.num_warmup):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch_inputs = batch['inputs'].to(device)
        batch_data_samples = batch['data_samples']

        with torch.no_grad():
            _ = model.loss(batch_inputs, batch_data_samples)

        if (i + 1) % 1 == 0:
            print(f"  Warmup iter {i+1}/{args.num_warmup} done")

    # ---- Profile with TimingStats ----
    print(f"\n--- Profiling ({args.num_profile} iters) ---")
    stats = TimingStats()

    tb_profiler = None
    if args.use_tb_profiler:
        from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity, schedule as prof_schedule
        tb_profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=tensorboard_trace_handler('./profiler_logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        tb_profiler.start()

    for i in range(args.num_profile):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch_inputs = batch['inputs'].to(device)
        batch_data_samples = batch['data_samples']

        # 整体计时
        with timer(stats, "0. FULL ITERATION (loss + backward)"):
            # 细粒度前向
            with timer(stats, "0a. forward (loss computation)"):
                losses = profile_model_forward(
                    model, batch_inputs, batch_data_samples, stats, device
                )

            # 反向传播
            total_loss = sum(losses.values())
            with timer(stats, "0b. backward"):
                total_loss.backward()

            # 模拟 optimizer step
            with timer(stats, "0c. optimizer step (zero_grad + step)"):
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = None  # 模拟 zero_grad

        if tb_profiler is not None:
            tb_profiler.step()

        print(f"  Profile iter {i+1}/{args.num_profile} done, loss={total_loss.item():.4f}")

    if tb_profiler is not None:
        tb_profiler.stop()

    # ---- 输出结果 ----
    print("\n" + "=" * 60)
    print("PROFILE RESULTS")
    print("=" * 60)
    print(stats.summary())

    # ---- GPU 内存统计 ----
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
        print(f"  Reserved:  {torch.cuda.max_memory_reserved()/1024**2:.1f} MB")

    # ---- 模型参数统计 ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # ---- 各子模块参数量 ----
    print(f"\nSub-module Parameters:")
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:<20} {n_params:>12,}")

    if hasattr(model, 'bbox_head'):
        for name, module in model.bbox_head.named_children():
            n_params = sum(p.numel() for p in module.parameters())
            print(f"  bbox_head.{name:<15} {n_params:>12,}")

    # ---- 关键配置信息 ----
    print(f"\nKey Config:")
    bh = model.bbox_head
    print(f"  num_proposals:   {bh.num_proposals}")
    print(f"  num_heads:       {bh.num_heads}")
    print(f"  deep_supervision:{bh.deep_supervision}")
    print(f"  diffusion_type:  {bh.diffusion_type}")
    print(f"  ot_coupling:     {bh.ot_coupling}")
    print(f"  ot_epsilon:      {bh.ot_epsilon}")
    print(f"  ot_num_iters:    {bh.ot_num_iters}")
    print(f"  ot_group_hier:   {bh.ot_module.ot_group_hierarchical}")
    print(f"  use_trd:         {bh.use_trd}")
    print(f"  use_cat:         {bh.use_cat}")
    print(f"  use_velocity:    {bh.use_velocity_loss}")
    print(f"  use_flash_attn:  {bh.use_flash_attn}")
    print(f"  time_conditioning:{bh.head_series[0].time_conditioning if bh.head_series else 'N/A'}")

    if args.use_tb_profiler:
        print(f"\nTensorBoard Profiler logs saved to: ./profiler_logs/")
        print(f"View with: tensorboard --logdir ./profiler_logs")


if __name__ == '__main__':
    main()
