"""推理 sanity check — 加载真实 checkpoint, 检查推理时的 pred_boxes 分布.

诊断动机:
  梯度诊断显示各模块梯度健康 (decoder 子层都有合理梯度), planA/B 让所有
  slot 有监督, 但 Epoch 1 val mAP=0.0000. 这说明问题可能不在训练梯度,
  而在推理 pipeline 或训练-推理分布不匹配.

本脚本加载真实 Epoch 1 checkpoint, 做合成推理, 检查:
  1. 每步 Euler 的 x_t 范围 (是否发散到 OOD)
  2. 最终 pred_boxes 在 diffusion [-s,s] / norm [0,1] 空间的分布
  3. pred_logits sigmoid max score 分布 (是否有高置信度 slot)
  4. box_renewal 每步重置了多少 slot
  5. matched_mask 在推理时是否正确处理 (推理无 GT, 应为 None)

这能区分:
  - 训练问题: pred_boxes 发散或全在边界 → box_head 没学好
  - 推理问题: pred_boxes 合理但 score 全低 → 后处理/score_thr 问题
  - 评估问题: pred_boxes + score 都合理 → CocoMetric 配置问题
"""

import argparse
import copy
from typing import Dict

import torch
from torch import Tensor


def load_head_from_checkpoint(
    config_path: str,
    checkpoint_path: str,
    device: str = 'cpu',
):
    """从完整模型 checkpoint 加载 bbox_head.

    Args:
        config_path: 配置文件 (用于构建 head 结构)
        checkpoint_path: 完整模型 checkpoint (.pth)
        device: 'cpu' or 'cuda'
    """
    from mmengine.config import Config
    from setdiff.models.set_head import JointDiffusionHead

    cfg = Config.fromfile(config_path)
    head_cfg = copy.deepcopy(cfg.model.bbox_head)
    head_cfg.pop('type', None)
    head = JointDiffusionHead(**head_cfg)

    # 加载 checkpoint, 提取 bbox_head 的 state_dict
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)

    # 提取 bbox_head.* 的 key, 去掉前缀
    head_keys = {
        k[len('bbox_head.'):]: v
        for k, v in state_dict.items()
        if k.startswith('bbox_head.')
    }

    if not head_keys:
        raise RuntimeError(
            f"checkpoint 中没有 bbox_head.* 的 key. "
            f"可用前缀: {set(k.split('.')[0] for k in state_dict.keys())}"
        )

    missing, unexpected = head.load_state_dict(head_keys, strict=False)
    if missing:
        print(f"  ⚠️ missing keys: {missing[:5]}... ({len(missing)} total)")
    if unexpected:
        print(f"  ⚠️ unexpected keys: {unexpected[:5]}... ({len(unexpected)} total)")

    head = head.to(device)
    head.eval()
    print(f"  ✓ 加载 checkpoint: {checkpoint_path}")
    print(f"  ✓ bbox_head 参数数: {sum(p.numel() for p in head.parameters())}")
    return head


def run_inference_sanity(
    config_path: str,
    checkpoint_path: str,
    device: str = 'cpu',
    label: str = '',
):
    """运行推理 sanity check."""
    print(f"\n{'=' * 80}")
    print(f"推理 sanity check: {label}")
    print(f"config: {config_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {device}")
    print(f"{'=' * 80}")

    head = load_head_from_checkpoint(config_path, checkpoint_path, device)
    snr_scale = head.snr_scale
    num_steps = head.num_sample_steps

    # 合成 image_features (固定 seed 可复现)
    torch.manual_seed(42)
    B = 2
    HW = 64
    feat_channels = head.feat_channels
    image_features = torch.randn(B, HW, feat_channels, device=device)

    print(f"\n--- 推理配置 ---")
    print(f"  num_queries: {head.num_queries}")
    print(f"  num_sample_steps: {num_steps}")
    print(f"  sampler: {head.sampler}")
    print(f"  snr_scale: {snr_scale}")
    print(f"  box_renewal: {head.box_renewal}")
    print(f"  score_thr: {head.score_thr}")
    print(f"  min_keep: {head.min_keep}")

    # ============================================================
    # 手动执行推理循环, 记录每步统计
    # ============================================================
    with torch.no_grad():
        # 1. 初始 noise
        x_t = torch.randn(B, head.num_queries, 4, device=device)
        print(f"\n--- 初始 x_t (t=1.0, 纯 noise) ---")
        _print_tensor_stats('x_t', x_t, snr_scale)

        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t_curr = float(timesteps[i].item())
            t_next = float(timesteps[i + 1].item())

            t = torch.full((B,), t_curr, device=device)
            t_scaled = t * 1000.0
            t_emb = head.time_embed(t_scaled)

            cls_logits, pred_boxes = head.encoder(x_t, t_emb, image_features)

            print(f"\n--- Step {i+1}/{num_steps}: t={t_curr:.3f} → {t_next:.3f} ---")
            _print_tensor_stats('pred_boxes (diffusion)', pred_boxes, snr_scale)
            _print_tensor_stats('cls_logits', cls_logits, snr_scale)

            # sigmoid score 分布
            scores = torch.sigmoid(cls_logits)
            max_scores = scores.max(dim=-1)[0]
            print(
                f"  sigmoid max score: mean={max_scores.mean():.6f} "
                f"max={max_scores.max():.6f} min={max_scores.min():.6f} "
                f">0.3: {(max_scores > 0.3).sum().item()}/{max_scores.numel()} "
                f">0.5: {(max_scores > 0.5).sum().item()}/{max_scores.numel()}"
            )

            # Euler step
            x_t_new = head.rf.step(x_t, pred_boxes, t_curr, t_next)

            if head.box_renewal and i < num_steps - 1:
                x_t_before = x_t_new.clone()
                x_t_new = head._apply_box_renewal(x_t_new, cls_logits)
                num_renewed = (x_t_new != x_t_before).any(-1).sum().item()
                total = x_t_new.shape[0] * x_t_new.shape[1]
                print(f"  box_renewal: 重置 {num_renewed}/{total} slot 为 randn")

            print(f"  x_t after step (t={t_next:.3f}):")
            _print_tensor_stats('x_t_new', x_t_new, snr_scale)

            x_t = x_t_new

    # ============================================================
    # 最终预测分析
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"--- 最终预测分析 (最后一步 pred_boxes/pred_logits) ---")
    print(f"{'=' * 80}")

    # diffusion → norm space
    s = snr_scale
    pred_boxes_norm = (pred_boxes.clamp(-s, s) / s + 1.0) / 2.0
    print(f"\n最终 pred_boxes (norm [0,1] space):")
    _print_norm_box_stats(pred_boxes_norm)

    # score 分布
    scores = torch.sigmoid(cls_logits)
    max_scores = scores.max(dim=-1)[0]
    labels = scores.argmax(dim=-1)
    print(f"\n最终预测 score 分布:")
    print(f"  全部 {max_scores.numel()} 个 slot:")
    print(f"    score mean={max_scores.mean():.6f} max={max_scores.max():.6f}")
    print(f"    >0.1: {(max_scores > 0.1).sum().item()}")
    print(f"    >0.3: {(max_scores > 0.3).sum().item()}")
    print(f"    >0.5: {(max_scores > 0.5).sum().item()}")
    print(f"    >0.9: {(max_scores > 0.9).sum().item()}")

    # 高置信度 slot 的 box 分布
    high_conf = max_scores > 0.3
    if high_conf.any():
        print(f"\n  高置信度 slot (score>0.3) 的 box 分布:")
        _print_norm_box_stats(pred_boxes_norm[high_conf])
        print(f"  高置信度 slot 的 label 分布:")
        high_labels = labels[high_conf]
        for lab in high_labels.unique():
            count = (high_labels == lab).sum().item()
            print(f"    label {lab.item()}: {count} slot")
    else:
        print(f"\n  ⚠️ 没有任何 slot score > 0.3!")
        print(f"  这解释了 mAP=0: 所有预测都被 score_thr 过滤掉")

    # 与 GT 空间对比 (GT 通常 w,h ∈ [0.05, 0.3])
    print(f"\n  pred_boxes w,h 分布 (GT 通常 w,h ∈ [0.05, 0.3]):")
    wh = pred_boxes_norm[..., 2:]
    print(
        f"    w: mean={wh[..., 0].mean():.4f} min={wh[..., 0].min():.4f} "
        f"max={wh[..., 0].max():.4f}"
    )
    print(
        f"    h: mean={wh[..., 1].mean():.4f} min={wh[..., 1].min():.4f} "
        f"max={wh[..., 1].max():.4f}"
    )
    neg_wh = (wh <= 0).any(-1).sum().item()
    huge_wh = (wh > 0.8).any(-1).sum().item()
    print(f"    w或h<=0 (框翻转): {neg_wh}/{wh.shape[0] * wh.shape[1]}")
    print(f"    w或h>0.8 (异常大): {huge_wh}/{wh.shape[0] * wh.shape[1]}")


def _print_tensor_stats(name: str, t: Tensor, snr_scale: float):
    """打印张量统计."""
    print(
        f"  {name}: shape={list(t.shape)} "
        f"mean={t.mean():.4f} std={t.std():.4f} "
        f"min={t.min():.4f} max={t.max():.4f}"
    )
    # 检查是否在合理范围
    s = snr_scale
    out_of_range = (t.abs() > s * 1.5).sum().item()
    if out_of_range > 0:
        total = t.numel()
        print(f"    ⚠️ {out_of_range}/{total} 元素超出 [-{s*1.5}, {s*1.5}] (OOD!)")


def _print_norm_box_stats(boxes_norm: Tensor):
    """打印 [0,1] 空间 box 统计."""
    if boxes_norm.numel() == 0:
        print(f"  (空)")
        return
    print(
        f"  shape={list(boxes_norm.shape)} "
        f"mean={boxes_norm.mean():.4f} std={boxes_norm.std():.4f} "
        f"min={boxes_norm.min():.4f} max={boxes_norm.max():.4f}"
    )
    # 检查是否在 [0,1]
    out_of_range = ((boxes_norm < 0) | (boxes_norm > 1)).sum().item()
    if out_of_range > 0:
        total = boxes_norm.numel()
        print(
            f"  ⚠️ {out_of_range}/{total} 元素超出 [0,1] "
            f"(clamp 前的 pred_boxes 异常!)"
        )
    # cx, cy 应在 [0,1], w, h 应在 [0, 0.5] (合理范围)
    cx, cy = boxes_norm[..., 0], boxes_norm[..., 1]
    w, h = boxes_norm[..., 2], boxes_norm[..., 3]
    print(
        f"  cx: [{cx.min():.4f}, {cx.max():.4f}] "
        f"cy: [{cy.min():.4f}, {cy.max():.4f}]"
    )
    print(
        f"  w:  [{w.min():.4f}, {w.max():.4f}] "
        f"h:  [{h.min():.4f}, {h.max():.4f}]"
    )


def main():
    parser = argparse.ArgumentParser(description='推理 sanity check')
    parser.add_argument(
        '--config', type=str,
        default='experiments/configs/setdiff/setdiff_24obj_planA_20ep.py',
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default='/tmp/setdiff_ckpts/planA_epoch1.pth',
    )
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--label', type=str, default='planA_epoch1')
    args = parser.parse_args()

    run_inference_sanity(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        label=args.label,
    )


if __name__ == '__main__':
    main()
