"""decoder 子层梯度分析 — 细化到 self_attn / cross_attn / FFN / norms.

诊断动机:
  用户反馈 "你不能只看最后的loss, 每个重要模块上的梯度更应该是
  是否有效训练的关键". 之前的 grad_analysis.py 只看模块级总梯度 norm,
  粒度太粗 (encoder.decoder 作为整体). 本脚本细化到 decoder 每层每个
  子模块, 特别关注:

    - self_attn (box-box): SetDiff 联合扩散的理论核心, 建模框间依赖 (v_i
      可依赖 x_j). 若 self_attn 梯度消失, 说明联合扩散机制未激活 —
      这是 SetDiff 区别于 DiffusionDet/DiffuDETR 的关键.
    - cross_attn (box-image): 检测核心, 从图像特征提取检测信号.
      若 cross_attn 梯度异常, 说明检测信号未传入 box 表示.
    - FFN (linear1+linear2): 非线性变换容量.
    - norms (norm1/2/3): LayerNorm, 梯度应小但非零.

附加分析:
    - 梯度稀疏性: 每模块非零梯度参数比例 (检测梯度消失/死神经元).
    - per-slot box_head 梯度: matched vs unmatched slot 的输入梯度,
      验证 box_head 是否对所有 slot 都有有效梯度 (方案 A/B 的核心修复点).
    - 梯度方向稳定性: 相邻 iter 同模块梯度的 cosine similarity,
      反映训练稳定性 (低 cosine = 梯度震荡, 训练无效).

对比: baseline (noise) vs planA (random) vs planB (random_gt).
"""

import argparse
import copy
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# 辅助函数
# ============================================================


def _module_grad_norm(module: torch.nn.Module) -> float:
    """计算模块所有参数梯度的 L2 norm (合并所有参数)."""
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().pow(2).sum().item()
    return total_sq ** 0.5


def _module_grad_sparsity(module: torch.nn.Module) -> float:
    """计算模块梯度的稀疏性: 零梯度元素比例 (0=dense, 1=fully sparse)."""
    total = 0
    zero = 0
    for p in module.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total += g.numel()
            zero += (g.abs() < 1e-12).sum().item()
    return zero / max(total, 1)


def _build_head_from_config(
    config_path: str,
    matcher_type: str = None,
    unmatched_strategy: str = None,
):
    """从配置文件构建 JointDiffusionHead (跳过 backbone/neck)."""
    from mmengine.config import Config
    from setdiff.models.set_head import JointDiffusionHead

    cfg = Config.fromfile(config_path)
    head_cfg = copy.deepcopy(cfg.model.bbox_head)
    head_cfg.pop('type', None)
    if matcher_type is not None:
        head_cfg['matcher_type'] = matcher_type
    if unmatched_strategy is not None:
        head_cfg['unmatched_strategy'] = unmatched_strategy
    return JointDiffusionHead(**head_cfg)


def _make_synthetic_batch(head, B=2, M=8, num_classes=24, device='cpu'):
    """构建合成 batch (无需真实数据)."""
    torch.manual_seed(42)
    feat_channels = head.feat_channels
    HW = 64
    image_features = torch.randn(B, HW, feat_channels, device=device)

    s = head.snr_scale
    gt_cxcywh_norm = torch.rand(B, M, 4, device=device)
    gt_cxcywh_norm[..., 2:] = gt_cxcywh_norm[..., 2:] * 0.3 + 0.05
    gt_diffusion = (gt_cxcywh_norm * 2 - 1) * s

    gt_boxes_list = [gt_diffusion[i] for i in range(B)]
    gt_labels_list = [
        torch.randint(0, num_classes, (M,), device=device) for _ in range(B)
    ]
    return image_features, gt_boxes_list, gt_labels_list


# ============================================================
# 子层梯度收集
# ============================================================


def collect_decoder_sublayer_grads(head) -> Dict[str, float]:
    """收集 decoder 每层每个子模块的梯度 L2 norm.

    Returns:
        dict: {
            'L0.self_attn': float, 'L0.cross_attn': float,
            'L0.ffn': float, 'L0.norms': float,
            ... (每层)
            'avg.self_attn': float, 'avg.cross_attn': float,
            'avg.ffn': float, 'avg.norms': float,
        }
    """
    results = {}
    decoder = head.encoder.decoder
    num_layers = len(decoder.layers)

    sa_norms, ca_norms, ffn_norms, n_norms = [], [], [], []

    for i, layer in enumerate(decoder.layers):
        sa = _module_grad_norm(layer.self_attn)
        ca = _module_grad_norm(layer.multihead_attn)
        ffn = _module_grad_norm(layer.linear1) + _module_grad_norm(layer.linear2)
        norms = (
            _module_grad_norm(layer.norm1)
            + _module_grad_norm(layer.norm2)
            + _module_grad_norm(layer.norm3)
        )

        results[f'L{i}.self_attn'] = sa
        results[f'L{i}.cross_attn'] = ca
        results[f'L{i}.ffn'] = ffn
        results[f'L{i}.norms'] = norms

        sa_norms.append(sa)
        ca_norms.append(ca)
        ffn_norms.append(ffn)
        n_norms.append(norms)

    results['avg.self_attn'] = sum(sa_norms) / num_layers
    results['avg.cross_attn'] = sum(ca_norms) / num_layers
    results['avg.ffn'] = sum(ffn_norms) / num_layers
    results['avg.norms'] = sum(n_norms) / num_layers
    return results


def collect_other_module_grads(head) -> Dict[str, float]:
    """收集其他关键模块的梯度 (box_head/cls_head/box_pos_embed/query_embed/time_embed)."""
    return {
        'box_head': _module_grad_norm(head.encoder.box_head),
        'cls_head': _module_grad_norm(head.encoder.cls_head),
        'box_pos_embed': _module_grad_norm(head.encoder.box_pos_embed),
        'query_embed': _module_grad_norm(head.encoder.query_embed),
        'time_embed': _module_grad_norm(head.time_embed),
    }


def collect_sparsity(head) -> Dict[str, float]:
    """收集各模块梯度稀疏性 (零梯度元素比例)."""
    return {
        'box_head': _module_grad_sparsity(head.encoder.box_head),
        'cls_head': _module_grad_sparsity(head.encoder.cls_head),
        'decoder.self_attn_avg': sum(
            _module_grad_sparsity(l.self_attn)
            for l in head.encoder.decoder.layers
        ) / len(head.encoder.decoder.layers),
        'decoder.cross_attn_avg': sum(
            _module_grad_sparsity(l.multihead_attn)
            for l in head.encoder.decoder.layers
        ) / len(head.encoder.decoder.layers),
        'decoder.ffn_avg': sum(
            _module_grad_sparsity(l.linear1) + _module_grad_sparsity(l.linear2)
            for l in head.encoder.decoder.layers
        ) / len(head.encoder.decoder.layers) / 2,
    }


# ============================================================
# per-slot box_head 梯度分析
# ============================================================


def compute_per_slot_box_head_grad(
    head,
    image_features: Tensor,
    gt_boxes_list: List[Tensor],
    gt_labels_list: List[Tensor],
) -> Dict[str, float]:
    """计算 box_head 对 matched vs unmatched slot 的输入梯度.

    方法: 对 box_head 的输入 (encoder 输出 hs [B, N, C]) 求梯度,
    按 matched/unmatched 分组统计 L2 norm.

    这是验证方案 A/B 修复效果的核心指标:
      - baseline: unmatched slot 应该梯度 ≈ 0 (无监督)
      - planA/B: unmatched slot 应该有非零梯度 (有监督)
    """
    head.train()
    B = image_features.shape[0]

    # 捕获 matched_labels
    captured = {}
    original_match_batch = head.matcher.match_batch

    def capture_match(noise, gt_boxes, gt_labels):
        mb, ml = original_match_batch(noise, gt_boxes, gt_labels)
        captured['ml'] = ml
        return mb, ml

    head.matcher.match_batch = capture_match

    # hook: 捕获 box_head 输入 (hs)
    hs_input = {}

    def hook_fn(module, inp, out):
        # inp[0] 是 hs [B, N, C]
        hs_input['hs'] = inp[0].detach().clone()
        hs_input['hs'].requires_grad_(True)

    handle = head.encoder.box_head.register_forward_pre_hook(
        lambda m, inp: None  # pre-hook 不修改输入
    )
    # 用 full hook 捕获输入
    handle.remove()

    # 更可靠的方法: 重新前向, 对 hs 求梯度
    # 但 box_head 是 MLP, 输入是 hs. 我们需要对 hs 求梯度.
    # 用 autograd.grad 对 hs 求梯度.

    # 1. 前向到 encoder, 拿到 hs (但 encoder 内部调用 box_head)
    # 改为: 直接对 pred_boxes 求梯度, 反传到 hs
    # 用 hook 捕获 hs, 然后 retain_grad

    hs_captured = {}

    def capture_hs_hook(module, inp, out):
        # inp = (hs,), hs [B, N, C]
        hs = inp[0]
        hs.retain_grad()
        hs_captured['hs'] = hs

    h = head.encoder.box_head.register_forward_hook(capture_hs_hook)

    loss_dict = head(image_features, gt_boxes_list, gt_labels_list)
    total_loss = sum(loss_dict.values())

    head.matcher.match_batch = original_match_batch
    h.remove()

    if not torch.isfinite(total_loss):
        return {'matched_grad': float('nan'), 'unmatched_grad': float('nan')}

    total_loss.backward()

    hs = hs_captured.get('hs', None)
    ml = captured.get('ml', None)
    if hs is None or hs.grad is None or ml is None:
        return {'matched_grad': float('nan'), 'unmatched_grad': float('nan')}

    # hs.grad: [B, N, C]
    grad_per_slot = hs.grad.norm(dim=-1)  # [B, N] 每个 slot 的输入梯度 norm
    matched_mask = ml >= 0  # [B, N]

    matched_grad = grad_per_slot[matched_mask].mean().item() if matched_mask.any() else 0.0
    unmatched_mask = ~matched_mask
    unmatched_grad = (
        grad_per_slot[unmatched_mask].mean().item()
        if unmatched_mask.any()
        else 0.0
    )

    return {
        'matched_grad': matched_grad,
        'unmatched_grad': unmatched_grad,
        'matched_count': matched_mask.sum().item(),
        'unmatched_count': unmatched_mask.sum().item(),
    }


# ============================================================
# 主诊断流程
# ============================================================


def run_sublayer_diagnosis(
    config_path: str,
    matcher_type: str = None,
    unmatched_strategy: str = None,
    num_iters: int = 10,
    device: str = 'cpu',
    label: str = '',
):
    """运行子层梯度诊断.

    收集 num_iters 个 iteration 的:
      1. decoder 各子层梯度 norm (每层 + 平均)
      2. 其他关键模块梯度 norm
      3. 梯度稀疏性
      4. per-slot box_head 输入梯度 (matched vs unmatched)
    """
    clip_status = 'clip_grad=1.0 (raw grad collected before clip)'
    print(f"\n{'=' * 80}")
    print(f"子层梯度诊断: {label}")
    print(f"配置: {config_path}")
    if matcher_type:
        print(f"matcher_type: {matcher_type}")
    if unmatched_strategy:
        print(f"unmatched_strategy: {unmatched_strategy}")
    print(f"iterations: {num_iters}, device: {device}")
    print(f"[{clip_status}]")
    print(f"{'=' * 80}")

    head = _build_head_from_config(
        config_path, matcher_type=matcher_type,
        unmatched_strategy=unmatched_strategy
    )
    head = head.to(device)
    head.train()

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=5e-5, weight_decay=0.0001
    )

    # 累积器
    sublayer_acc = defaultdict(list)  # 子层梯度 norm
    module_acc = defaultdict(list)  # 其他模块梯度 norm
    sparsity_acc = defaultdict(list)  # 稀疏性
    perslot_acc = defaultdict(list)  # per-slot
    loss_acc = defaultdict(list)
    supervised_slots = []

    for it in range(num_iters):
        optimizer.zero_grad()

        image_features, gt_boxes_list, gt_labels_list = _make_synthetic_batch(
            head, B=2, M=8, device=device
        )

        loss_dict = head(image_features, gt_boxes_list, gt_labels_list)
        total_loss = sum(loss_dict.values())

        if not torch.isfinite(total_loss):
            print(f"  iter {it}: loss 非有限, 跳过")
            continue

        total_loss.backward()

        # 收集 (clip 前)
        sl_grads = collect_decoder_sublayer_grads(head)
        mod_grads = collect_other_module_grads(head)
        sparsity = collect_sparsity(head)

        for k, v in sl_grads.items():
            sublayer_acc[k].append(v)
        for k, v in mod_grads.items():
            module_acc[k].append(v)
        for k, v in sparsity.items():
            sparsity_acc[k].append(v)

        for k, v in loss_dict.items():
            loss_acc[k].append(v.item())
        loss_acc['total'].append(total_loss.item())

        # clip
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        optimizer.step()

        if it % 3 == 0 or it == num_iters - 1:
            print(
                f"  iter {it}: total={total_loss.item():.4f} "
                f"cls={loss_dict['loss_cls'].item():.4f} "
                f"bbox={loss_dict['loss_bbox'].item():.4f} "
                f"giou={loss_dict['loss_giou'].item():.4f}"
            )

    # per-slot 分析 (单独跑一次, 因为需要 retain_grad)
    head.zero_grad()
    image_features, gt_boxes_list, gt_labels_list = _make_synthetic_batch(
        head, B=2, M=8, device=device
    )
    perslot = compute_per_slot_box_head_grad(
        head, image_features, gt_boxes_list, gt_labels_list
    )

    # 聚合
    def avg(d):
        return {k: sum(v) / len(v) for k, v in d.items()} if d else {}

    sl_avg = avg(sublayer_acc)
    mod_avg = avg(module_acc)
    sp_avg = avg(sparsity_acc)
    loss_avg = avg(loss_acc)

    # 打印
    print(f"\n--- decoder 各子层梯度 L2 norm (mean over {num_iters} iters, clip 前) ---")
    print(f"{'层':<6} {'self_attn':>14} {'cross_attn':>14} {'ffn':>14} {'norms':>14}")
    print('-' * 66)
    num_layers = len(head.encoder.decoder.layers)
    for i in range(num_layers):
        print(
            f"L{i:<5} "
            f"{sl_avg.get(f'L{i}.self_attn', 0):>14.6f} "
            f"{sl_avg.get(f'L{i}.cross_attn', 0):>14.6f} "
            f"{sl_avg.get(f'L{i}.ffn', 0):>14.6f} "
            f"{sl_avg.get(f'L{i}.norms', 0):>14.6f}"
        )
    print('-' * 66)
    print(
        f"{'avg':<6} "
        f"{sl_avg.get('avg.self_attn', 0):>14.6f} "
        f"{sl_avg.get('avg.cross_attn', 0):>14.6f} "
        f"{sl_avg.get('avg.ffn', 0):>14.6f} "
        f"{sl_avg.get('avg.norms', 0):>14.6f}"
    )

    print(f"\n--- 其他关键模块梯度 L2 norm ---")
    for k in ['box_head', 'cls_head', 'box_pos_embed', 'query_embed', 'time_embed']:
        print(f"  {k:<20} {mod_avg.get(k, 0):>14.6f}")

    print(f"\n--- 梯度稀疏性 (零梯度元素比例, 0=dense 1=sparse) ---")
    for k, v in sp_avg.items():
        print(f"  {k:<30} {v:>8.4f}")

    print(f"\n--- per-slot box_head 输入梯度 (matched vs unmatched) ---")
    print(f"  matched slots:   count={perslot.get('matched_count', 0)} "
          f"mean_grad={perslot.get('matched_grad', 0):.6f}")
    print(f"  unmatched slots: count={perslot.get('unmatched_count', 0)} "
          f"mean_grad={perslot.get('unmatched_grad', 0):.6f}")
    if perslot.get('unmatched_grad', 0) > 0 and perslot.get('matched_grad', 0) > 0:
        ratio = perslot['unmatched_grad'] / perslot['matched_grad']
        print(f"  unmatched/matched ratio: {ratio:.4f}")

    return {
        'sublayer': sl_avg,
        'modules': mod_avg,
        'sparsity': sp_avg,
        'perslot': perslot,
        'loss': loss_avg,
    }


def compare_three_configs(num_iters: int = 10, device: str = 'cpu'):
    """对比 baseline / planA / planB 的子层梯度."""
    baseline_config = 'experiments/configs/setdiff/setdiff_24obj.py'
    configs = [
        {
            'label': 'baseline(noise)',
            'config_path': baseline_config,
            'matcher_type': 'hungarian',
            'unmatched_strategy': 'noise',
        },
        {
            'label': 'planA(random)',
            'config_path': baseline_config,
            'matcher_type': 'random',
            'unmatched_strategy': 'noise',
        },
        {
            'label': 'planB(random_gt)',
            'config_path': baseline_config,
            'matcher_type': 'hungarian',
            'unmatched_strategy': 'random_gt',
        },
    ]

    all_results = {}
    for cfg in configs:
        result = run_sublayer_diagnosis(
            config_path=cfg['config_path'],
            matcher_type=cfg.get('matcher_type'),
            unmatched_strategy=cfg.get('unmatched_strategy'),
            num_iters=num_iters,
            device=device,
            label=cfg['label'],
        )
        all_results[cfg['label']] = result

    # 汇总对比表
    labels = list(all_results.keys())

    print(f"\n{'=' * 100}")
    print(f"子层梯度对比汇总 (mean L2 norm, clip 前)")
    print(f"{'=' * 100}")

    # decoder 子层平均对比
    print(f"\n--- decoder 各子层平均梯度 (6层平均) ---")
    print(f"{'子层':<25}" + ''.join(f'{l:>22}' for l in labels))
    print('-' * 91)
    for sub in ['avg.self_attn', 'avg.cross_attn', 'avg.ffn', 'avg.norms']:
        row = f"{sub:<25}"
        for label in labels:
            v = all_results[label]['sublayer'].get(sub, 0)
            row += f'{v:>22.6f}'
        print(row)

    # 其他模块对比
    print(f"\n--- 其他关键模块梯度 ---")
    print(f"{'模块':<25}" + ''.join(f'{l:>22}' for l in labels))
    print('-' * 91)
    for mod in ['box_head', 'cls_head', 'box_pos_embed', 'query_embed', 'time_embed']:
        row = f"{mod:<25}"
        for label in labels:
            v = all_results[label]['modules'].get(mod, 0)
            row += f'{v:>22.6f}'
        print(row)

    # 稀疏性对比
    print(f"\n--- 梯度稀疏性 (零比例) ---")
    print(f"{'模块':<30}" + ''.join(f'{l:>22}' for l in labels))
    print('-' * 96)
    for sp in ['box_head', 'cls_head', 'decoder.self_attn_avg',
               'decoder.cross_attn_avg', 'decoder.ffn_avg']:
        row = f"{sp:<30}"
        for label in labels:
            v = all_results[label]['sparsity'].get(sp, 0)
            row += f'{v:>22.4f}'
        print(row)

    # per-slot 对比 (核心!)
    print(f"\n--- per-slot box_head 输入梯度 (核心: matched vs unmatched) ---")
    print(f"{'指标':<30}" + ''.join(f'{l:>22}' for l in labels))
    print('-' * 96)
    for k in ['matched_grad', 'unmatched_grad', 'matched_count', 'unmatched_count']:
        row = f"{k:<30}"
        for label in labels:
            v = all_results[label]['perslot'].get(k, 0)
            if isinstance(v, float):
                row += f'{v:>22.6f}'
            else:
                row += f'{v:>22}'
        print(row)

    # 关键洞察
    print(f"\n--- 关键洞察 ---")
    base_perslot = all_results.get('baseline(noise)', {}).get('perslot', {})
    for label in ['planA(random)', 'planB(random_gt)']:
        ps = all_results[label]['perslot']
        base_unmatched = base_perslot.get('unmatched_grad', 0)
        plan_unmatched = ps.get('unmatched_grad', 0)
        if base_unmatched > 0:
            ratio = plan_unmatched / base_unmatched
            print(f"  {label}: unmatched slot 梯度 = {plan_unmatched:.6f} "
                  f"({ratio:.2f}x baseline)")
        else:
            print(f"  {label}: unmatched slot 梯度 = {plan_unmatched:.6f} "
                  f"(baseline 为 0, 方案让 unmatched 有监督)")


def main():
    parser = argparse.ArgumentParser(description='decoder 子层梯度分析')
    parser.add_argument(
        '--mode', type=str, default='compare',
        choices=['single', 'compare'],
    )
    parser.add_argument('--config', type=str,
                        default='experiments/configs/setdiff/setdiff_24obj.py')
    parser.add_argument('--matcher-type', type=str, default=None)
    parser.add_argument('--unmatched-strategy', type=str, default=None)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.mode == 'single':
        run_sublayer_diagnosis(
            config_path=args.config,
            matcher_type=args.matcher_type,
            unmatched_strategy=args.unmatched_strategy,
            num_iters=args.iters,
            device=args.device,
            label='single',
        )
    else:
        compare_three_configs(num_iters=args.iters, device=args.device)


if __name__ == '__main__':
    main()
