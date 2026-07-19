"""encoder 输出 hs 区分度诊断 — 检查 box_head 输入是否有多样性.

诊断动机:
  推理 sanity check 发现 box_head 模式坍缩 (所有 slot 输出相同的中心大框).
  但梯度诊断显示 box_head 梯度健康. 这说明问题可能在 box_head 的输入
  (encoder 输出 hs) 缺乏区分度 — 如果 hs 对所有 slot 都相似, box_head
  自然无法输出不同的 pred_boxes.

本脚本检查:
  1. hs [B, N, C] 的 per-slot 方差: 不同 slot 的 hs 是否不同?
  2. hs 的有效秩: hs 矩阵的实际秩 (反映信息多样性)
  3. cross_attn attention map: 是否聚焦到图像特定区域? 还是均匀分布?
  4. 对比: 随机初始化 head vs 加载 checkpoint 的 head, hs 区分度差异
"""

import argparse
import copy

import torch
from torch import Tensor


def _build_head(config_path, device='cpu', matcher_type=None,
                unmatched_strategy=None):
    from mmengine.config import Config
    from setdiff.models.set_head import JointDiffusionHead

    cfg = Config.fromfile(config_path)
    head_cfg = copy.deepcopy(cfg.model.bbox_head)
    head_cfg.pop('type', None)
    if matcher_type:
        head_cfg['matcher_type'] = matcher_type
    if unmatched_strategy:
        head_cfg['unmatched_strategy'] = unmatched_strategy
    return JointDiffusionHead(**head_cfg).to(device).eval()


def _load_checkpoint(head, checkpoint_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    head_keys = {
        k[len('bbox_head.'):]: v
        for k, v in state_dict.items()
        if k.startswith('bbox_head.')
    }
    head.load_state_dict(head_keys, strict=False)
    return head


def analyze_hs(head, image_features, x_t, t, label=''):
    """分析 encoder 输出 hs 的区分度."""
    B, N, C = x_t.shape[0], x_t.shape[1], head.feat_channels

    t_tensor = torch.full((B,), t, device=x_t.device)
    t_emb = head.time_embed(t_tensor * 1000.0)

    # 注册 hook 捕获 hs (encoder decoder 输出, box_head 输入)
    hs_captured = {}

    def hook(module, inp, out):
        # decoder 输出 hs [B, N, C]
        hs_captured['hs'] = out.detach()

    h = head.encoder.decoder.register_forward_hook(hook)

    with torch.no_grad():
        cls_logits, pred_boxes = head.encoder(x_t, t_emb, image_features)

    h.remove()

    hs = hs_captured.get('hs', None)
    if hs is None:
        print(f"  [{label}] 未能捕获 hs")
        return

    print(f"\n  [{label}] t={t:.2f} hs 分析:")
    print(f"    shape={list(hs.shape)} mean={hs.mean():.4f} std={hs.std():.4f}")

    # per-slot 方差: 每个 slot 跨 batch 的 hs 方差
    # hs [B, N, C] → 看 N 个 slot 之间的差异
    hs_per_slot_mean = hs.mean(dim=0)  # [N, C] 跨 batch 平均
    slot_diff = hs_per_slot_mean.std(dim=0)  # [C] 每个 channel 跨 slot 的 std
    print(
        f"    per-slot 差异 (channel-wise std across slots): "
        f"mean={slot_diff.mean():.6f} max={slot_diff.max():.6f}"
    )

    # slot 间余弦相似度 (反映 slot 多样性)
    # 取 batch 0, [N, C]
    hs_b0 = hs[0]  # [N, C]
    hs_norm = hs_b0 / (hs_b0.norm(dim=-1, keepdim=True) + 1e-8)
    cos_sim = hs_norm @ hs_norm.T  # [N, N]
    off_diag = cos_sim[~torch.eye(N, dtype=torch.bool, device=hs.device)]
    print(
        f"    slot 间余弦相似度: mean={off_diag.mean():.4f} "
        f"max={off_diag.max():.4f} min={off_diag.min():.4f}"
    )
    print(
        f"    (mean≈1.0 = 所有 slot 几乎相同 = 坍缩; "
        f"mean<0.5 = slot 有多样性)"
    )

    # hs 有效秩 (通过奇异值)
    try:
        hs_b0_np = hs_b0.cpu().numpy()
        import numpy as np
        sv = np.linalg.svd(hs_b0_np, compute_uv=False)
        sv_normalized = sv / (sv.sum() + 1e-8)
        entropy = -np.sum(sv_normalized * np.log(sv_normalized + 1e-8))
        eff_rank = np.exp(entropy)
        print(
            f"    有效秩 (effective rank): {eff_rank:.2f} / {N} "
            f"(越接近 N 越多样)"
        )
        print(
            f"    top-5 奇异值: {sv[:5].tolist()}"
        )
    except Exception as e:
        print(f"    有效秩计算失败: {e}")

    # pred_boxes 多样性
    print(
        f"    pred_boxes (diffusion): std={pred_boxes.std():.6f} "
        f"mean={pred_boxes.mean():.4f}"
    )
    box_per_slot = pred_boxes[0]  # [N, 4]
    box_std = box_per_slot.std(dim=0)  # [4] 每个 dim 跨 slot 的 std
    print(
        f"    pred_boxes per-dim std (across slots): "
        f"{box_std.tolist()}"
    )


def main():
    parser = argparse.ArgumentParser(description='hs 区分度诊断')
    parser.add_argument(
        '--config', type=str,
        default='experiments/configs/setdiff/setdiff_24obj_planA_20ep.py',
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default='/tmp/setdiff_ckpts/planA_epoch1.pth',
    )
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print(f"{'=' * 80}")
    print(f"hs 区分度诊断")
    print(f"{'=' * 80}")

    torch.manual_seed(42)
    B, HW = 2, 64
    image_features = torch.randn(B, HW, 256, device=args.device)
    N = 300

    # 1. 随机初始化 head
    print(f"\n--- 1. 随机初始化 head ---")
    head_random = _build_head(args.config, args.device)
    for t in [1.0, 0.5, 0.0]:
        x_t = torch.randn(B, N, 4, device=args.device)
        analyze_hs(head_random, image_features, x_t, t, label='random_init')

    # 2. 加载 checkpoint head
    print(f"\n--- 2. 加载 Epoch 1 checkpoint ---")
    head_ckpt = _build_head(args.config, args.device)
    head_ckpt = _load_checkpoint(head_ckpt, args.checkpoint, args.device)
    for t in [1.0, 0.5, 0.0]:
        x_t = torch.randn(B, N, 4, device=args.device)
        analyze_hs(head_ckpt, image_features, x_t, t, label='epoch1_ckpt')

    print(f"\n{'=' * 80}")
    print(f"诊断说明:")
    print(f"  - 若 epoch1_ckpt 的 slot 间余弦相似度 ≈ 1.0 (远高于 random_init),")
    print(f"    说明 encoder 训练后 hs 坍缩 (所有 slot 表示趋同), box_head 无法")
    print(f"    输出多样化预测 → mAP=0 根因.")
    print(f"  - 若有效秩远低于 N, 说明 hs 信息维度不足.")
    print(f"  - 若 pred_boxes per-dim std ≈ 0, 说明 box_head 输出无多样性.")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
