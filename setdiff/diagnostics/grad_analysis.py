"""梯度诊断工具 — 检查各模块梯度是否有效训练.

诊断目标:
  总 loss 下降 ≠ 各模块有效学习. 特别关注 box_head 的梯度:
  - baseline (Hungarian + unmatched=noise): box_head 对 ~85% slot 无监督,
    梯度应极小 (只来自 M 个 matched slot)
  - 方案 A (RandomMatcher): 所有 300 slot 有监督, box_head 梯度应显著增大
  - 方案 B (Hungarian + random_gt): 同方案 A, 所有 slot 有监督

关键模块:
  - backbone (ResNet50 各 stage) — 通常冻结 stage1+2, 只训练 stage3+4
  - neck (FPN)
  - encoder.box_pos_embed — box 位置编码
  - encoder.query_embed — content query (learnable)
  - encoder.decoder — Transformer 各层 (self-attn / cross-attn / FFN)
  - encoder.cls_head — 分类头
  - encoder.box_head — 回归头 ← mAP=0 根因所在, 最关键
  - time_embed — 时间嵌入

输出: 各模块梯度 L2 norm 统计 (mean / max / std over N iterations).
"""

import argparse
import copy
import sys
from collections import defaultdict
from typing import Dict, List

import torch
from torch import Tensor


def _build_head_from_config(config_path: str, matcher_type: str = None,
                             unmatched_strategy: str = None):
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
    # image_features: [B, HW, C]
    feat_channels = head.feat_channels
    HW = 64
    image_features = torch.randn(B, HW, feat_channels, device=device)

    # GT boxes: 扩散空间 [-snr_scale, +snr_scale], cxcywh
    s = head.snr_scale
    # 生成合理的 cxcywh 归一化 [0, 1] 再缩放到 [-s, s]
    gt_cxcywh_norm = torch.rand(B, M, 4, device=device)
    gt_cxcywh_norm[..., 2:] = gt_cxcywh_norm[..., 2:] * 0.3 + 0.05  # w,h ∈ [0.05, 0.35]
    gt_diffusion = (gt_cxcywh_norm * 2 - 1) * s  # [B, M, 4] in [-s, s]

    gt_boxes_list = [gt_diffusion[i] for i in range(B)]
    gt_labels_list = [
        torch.randint(0, num_classes, (M,), device=device) for _ in range(B)
    ]
    return image_features, gt_boxes_list, gt_labels_list


def collect_grad_norms(module: torch.nn.Module, prefix: str = '') -> Dict[str, float]:
    """收集模块中所有参数的梯度 L2 norm.

    Returns:
        dict: {param_name: grad_norm}
    """
    norms = {}
    for name, param in module.named_parameters(prefix=prefix):
        if param.grad is not None:
            norms[name] = param.grad.detach().norm(2).item()
        else:
            norms[name] = 0.0
    return norms


def aggregate_module_grads(head, iter_grads: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """聚合各模块的梯度统计.

    Args:
        head: JointDiffusionHead
        iter_grads: list of per-iteration grad norms dict

    Returns:
        dict: {
            module_name: {
                'mean': float, 'max': float, 'std': float, 'count': int
            }
        }
    """
    # 定义模块分组 (按路径前缀聚合)
    module_groups = {
        'encoder.box_pos_embed': head.encoder.box_pos_embed,
        'encoder.query_embed': head.encoder.query_embed,
        'encoder.cls_head': head.encoder.cls_head,
        'encoder.box_head': head.encoder.box_head,  # ← 关键
        'encoder.decoder': head.encoder.decoder,
        'time_embed': head.time_embed,
    }

    # 收集每个模块下所有参数的梯度
    module_stats = defaultdict(lambda: {'values': []})

    for iter_norms in iter_grads:
        for mod_name, mod in module_groups.items():
            # 跳过非 nn.Module (如 matcher 是普通类)
            if not isinstance(mod, torch.nn.Module):
                continue
            mod_params = dict(mod.named_parameters())
            if not mod_params:
                continue
            # 该模块下所有参数梯度 norm 的平均 (模块级代表值)
            mod_grad_norms = []
            for p_name in mod_params:
                full_name = f'{mod_name}.{p_name}' if mod_name else p_name
                if full_name in iter_norms:
                    mod_grad_norms.append(iter_norms[full_name])
                elif p_name in iter_norms:
                    mod_grad_norms.append(iter_norms[p_name])
            if mod_grad_norms:
                # 模块级: 取所有参数 grad norm 的均值
                module_stats[mod_name]['values'].append(
                    sum(mod_grad_norms) / len(mod_grad_norms)
                )

    # 计算统计量
    result = {}
    for mod_name, data in module_stats.items():
        vals = data['values']
        if not vals:
            continue
        tensor_vals = torch.tensor(vals)
        result[mod_name] = {
            'mean': tensor_vals.mean().item(),
            'max': tensor_vals.max().item(),
            'std': tensor_vals.std().item() if len(vals) > 1 else 0.0,
            'count': len(vals),
        }
    return result


def run_grad_diagnosis(
    config_path: str,
    matcher_type: str = None,
    unmatched_strategy: str = None,
    num_iters: int = 10,
    device: str = 'cpu',
    label: str = '',
    clip_grad: bool = True,
):
    """运行梯度诊断.

    Args:
        config_path: 配置文件路径
        matcher_type: 覆盖 matcher_type ('hungarian' | 'random' | None=用配置)
        unmatched_strategy: 覆盖 unmatched_strategy
        num_iters: 运行多少个 iteration 收集梯度
        device: 'cpu' or 'cuda'
        label: 输出标签
        clip_grad: 是否应用 clip_grad=1.0 (False=看原始梯度, 更能反映差异)

    Returns:
        dict: 各模块梯度统计 (clip 前 + clip 后)
    """
    clip_status = 'clip_grad=1.0' if clip_grad else 'NO clip (raw grad)'
    print(f"\n{'=' * 70}")
    print(f"梯度诊断: {label}  [{clip_status}]")
    print(f"配置: {config_path}")
    if matcher_type:
        print(f"matcher_type: {matcher_type}")
    if unmatched_strategy:
        print(f"unmatched_strategy: {unmatched_strategy}")
    print(f"iterations: {num_iters}, device: {device}")
    print(f"{'=' * 70}")

    head = _build_head_from_config(
        config_path, matcher_type=matcher_type,
        unmatched_strategy=unmatched_strategy
    )
    head = head.to(device)
    head.train()

    # 优化器 (对齐训练配置: AdamW lr=5e-5)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=5e-5, weight_decay=0.0001
    )

    iter_grads_raw = []   # clip 前的原始梯度
    iter_grads_clipped = []  # clip 后的梯度
    loss_stats = defaultdict(list)
    global_grad_norms = []  # 全局梯度 norm (clip 前)
    supervised_slot_counts = []  # 有效监督 slot 数 (label >= 0)

    for it in range(num_iters):
        optimizer.zero_grad()

        image_features, gt_boxes_list, gt_labels_list = _make_synthetic_batch(
            head, B=2, M=8, device=device
        )

        # 捕获 matched_labels 统计有效监督 slot 数
        original_match_batch = head.matcher.match_batch
        captured_labels = {}

        def capture_match(noise, gt_boxes, gt_labels):
            mb, ml = original_match_batch(noise, gt_boxes, gt_labels)
            captured_labels['ml'] = ml
            return mb, ml

        head.matcher.match_batch = capture_match

        loss_dict = head(image_features, gt_boxes_list, gt_labels_list)
        total_loss = sum(loss_dict.values())

        head.matcher.match_batch = original_match_batch  # 恢复

        # 统计有效监督 slot 数
        if 'ml' in captured_labels:
            ml = captured_labels['ml']
            num_supervised = (ml >= 0).sum().item()
            total_slots = ml.numel()
            supervised_slot_counts.append(num_supervised)

        # 检查 NaN/Inf
        if not torch.isfinite(total_loss):
            print(f"  iter {it}: loss 不是有限值: {total_loss.item()}")
            continue

        total_loss.backward()

        # 收集 clip 前的原始梯度
        grad_norms_raw = collect_grad_norms(head)
        iter_grads_raw.append(grad_norms_raw)

        # 计算全局梯度 norm (clip 前)
        global_norm = 0.0
        for p in head.parameters():
            if p.grad is not None:
                global_norm += p.grad.detach().norm(2).item() ** 2
        global_norm = global_norm ** 0.5
        global_grad_norms.append(global_norm)

        # 梯度裁剪 (对齐训练配置 clip_grad=1.0)
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)

        # 收集 clip 后的梯度
        grad_norms_clipped = collect_grad_norms(head)
        iter_grads_clipped.append(grad_norms_clipped)

        # 记录 loss
        for k, v in loss_dict.items():
            loss_stats[k].append(v.item())
        loss_stats['total'].append(total_loss.item())

        optimizer.step()

        if it % 2 == 0 or it == num_iters - 1:
            num_sup = supervised_slot_counts[-1] if supervised_slot_counts else -1
            print(
                f"  iter {it}: total={total_loss.item():.4f} "
                f"cls={loss_dict['loss_cls'].item():.4f} "
                f"bbox={loss_dict['loss_bbox'].item():.4f} "
                f"giou={loss_dict['loss_giou'].item():.4f} "
                f"| supervised_slots={num_sup}/{total_slots} "
                f"| global_grad_norm={global_norm:.4f}"
            )

    # 聚合统计
    module_stats_raw = aggregate_module_grads(head, iter_grads_raw)
    module_stats_clipped = aggregate_module_grads(head, iter_grads_clipped)

    print(f"\n--- 各模块梯度 L2 norm 统计 [原始梯度, clip 前] ---")
    print(f"{'模块':<30} {'mean':>12} {'max':>12} {'std':>12}")
    print('-' * 70)
    for mod_name in [
        'encoder.box_head',  # ← 最关键, 排第一
        'encoder.cls_head',
        'encoder.box_pos_embed',
        'encoder.query_embed',
        'encoder.decoder',
        'time_embed',
    ]:
        if mod_name in module_stats_raw:
            s = module_stats_raw[mod_name]
            print(
                f"{mod_name:<30} {s['mean']:>12.6f} "
                f"{s['max']:>12.6f} {s['std']:>12.6f}"
            )

    # 全局梯度 norm 统计
    if global_grad_norms:
        gn = global_grad_norms
        print(f"\n--- 全局梯度 norm (clip 前) ---")
        print(
            f"  mean={sum(gn)/len(gn):.4f} "
            f"max={max(gn):.4f} min={min(gn):.4f}"
        )
        print(f"  (训练配置 clip_grad=1.0, 故实际训练时全局 norm 被裁剪到 1.0)")

    if clip_grad:
        print(f"\n--- 各模块梯度 L2 norm 统计 [clip 后, 训练实际使用] ---")
        print(f"{'模块':<30} {'mean':>12} {'max':>12} {'std':>12}")
        print('-' * 70)
        for mod_name in [
            'encoder.box_head',
            'encoder.cls_head',
            'encoder.box_pos_embed',
            'encoder.query_embed',
            'encoder.decoder',
            'time_embed',
        ]:
            if mod_name in module_stats_clipped:
                s = module_stats_clipped[mod_name]
                print(
                    f"{mod_name:<30} {s['mean']:>12.6f} "
                    f"{s['max']:>12.6f} {s['std']:>12.6f}"
                )

    # loss 统计
    print(f"\n--- Loss 统计 ---")
    print(f"{'loss':<15} {'mean':>10} {'last':>10}")
    for k in ['loss_cls', 'loss_bbox', 'loss_giou', 'total']:
        if k in loss_stats:
            vals = loss_stats[k]
            print(f"{k:<15} {sum(vals)/len(vals):>10.4f} {vals[-1]:>10.4f}")

    return {
        'raw': module_stats_raw,
        'clipped': module_stats_clipped,
        'global_norms': global_grad_norms,
        'supervised_slots': supervised_slot_counts,
    }, dict(loss_stats)


def compare_configs(configs: List[Dict], num_iters: int = 10, device: str = 'cpu'):
    """对比多个配置的梯度分布.

    Args:
        configs: list of dict, 每个 dict 包含 config_path, matcher_type,
                 unmatched_strategy, label
        num_iters: 每个配置运行的 iteration 数
        device: 'cpu' or 'cuda'
    """
    all_stats = {}
    all_losses = {}
    all_global_norms = {}
    all_supervised = {}
    for cfg in configs:
        result, losses = run_grad_diagnosis(
            config_path=cfg['config_path'],
            matcher_type=cfg.get('matcher_type'),
            unmatched_strategy=cfg.get('unmatched_strategy'),
            num_iters=num_iters,
            device=device,
            label=cfg['label'],
            clip_grad=True,
        )
        all_stats[cfg['label']] = result['raw']  # 用原始梯度对比
        all_losses[cfg['label']] = losses
        all_global_norms[cfg['label']] = result['global_norms']
        all_supervised[cfg['label']] = result['supervised_slots']

    # 对比表 (原始梯度, clip 前)
    print(f"\n{'=' * 100}")
    print(f"梯度对比汇总 [原始梯度 mean norm, clip 前] — 反映真实梯度差异")
    print(f"{'=' * 100}")
    modules = [
        'encoder.box_head',  # ← 最关键
        'encoder.cls_head',
        'encoder.box_pos_embed',
        'encoder.query_embed',
        'encoder.decoder',
        'time_embed',
    ]
    labels = list(all_stats.keys())
    header = f"{'模块':<30}" + ''.join(f'{l:>22}' for l in labels)
    print(header)
    print('-' * 100)
    for mod_name in modules:
        row = f"{mod_name:<30}"
        for label in labels:
            if mod_name in all_stats[label]:
                v = all_stats[label][mod_name]['mean']
                row += f'{v:>22.6f}'
            else:
                row += f'{"N/A":>22}'
        print(row)

    # box_head 详细对比 (最关键)
    print(f"\n--- box_head 梯度详细对比 [原始, clip 前] (mAP=0 根因模块) ---")
    box_head_vals = {}
    for label in labels:
        if 'encoder.box_head' in all_stats[label]:
            s = all_stats[label]['encoder.box_head']
            box_head_vals[label] = s['mean']
            print(f"  {label}: mean={s['mean']:.6f} max={s['max']:.6f}")

    # 计算相对差异
    if 'baseline(noise)' in box_head_vals:
        base_v = box_head_vals['baseline(noise)']
        print(f"\n--- box_head 梯度相对差异 (vs baseline) ---")
        for label, v in box_head_vals.items():
            if label == 'baseline(noise)':
                continue
            ratio = v / base_v if base_v > 0 else float('inf')
            pct = (ratio - 1) * 100
            print(f"  {label} / baseline = {ratio:.4f}x  ({pct:+.1f}%)")

    # 全局梯度 norm 对比
    print(f"\n--- 全局梯度 norm 对比 [clip 前] ---")
    for label in labels:
        gn = all_global_norms[label]
        if gn:
            print(
                f"  {label}: mean={sum(gn)/len(gn):.4f} "
                f"max={max(gn):.4f} min={min(gn):.4f}"
            )

    # 有效监督 slot 数对比 (关键指标!)
    print(f"\n--- 有效监督 slot 数对比 [loss_bbox/giou 覆盖的 slot] ---")
    print(f"  (关键: baseline 只有 M 个 matched slot 有监督, "
          f"方案 A/B 所有 N 个 slot 都有监督)")
    print(f"  (loss 归一化 /max(num_pos,1) 使总梯度 norm 相近, "
          f"但有效监督 slot 数差异巨大)")
    for label in labels:
        ss = all_supervised[label]
        if ss:
            print(
                f"  {label}: mean={sum(ss)/len(ss):.1f} "
                f"(total_slots={300 * 2})"  # B=2, N=300
            )

    # loss 对比
    print(f"\n--- Loss 对比 (last iter) ---")
    print(f"{'配置':<35} {'loss_cls':>10} {'loss_bbox':>10} {'loss_giou':>10}")
    for label in labels:
        l = all_losses[label]
        print(
            f"{label:<35} {l['loss_cls'][-1]:>10.4f} "
            f"{l['loss_bbox'][-1]:>10.4f} {l['loss_giou'][-1]:>10.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description='梯度诊断工具')
    parser.add_argument(
        '--mode', type=str, default='compare',
        choices=['single', 'compare'],
        help='single: 单配置诊断; compare: 对比 baseline/A/B',
    )
    parser.add_argument(
        '--config', type=str,
        default='experiments/configs/setdiff/setdiff_24obj_planA_20ep.py',
        help='single 模式的配置文件',
    )
    parser.add_argument(
        '--matcher-type', type=str, default=None,
        help='覆盖 matcher_type (hungarian | random)',
    )
    parser.add_argument(
        '--unmatched-strategy', type=str, default=None,
        help='覆盖 unmatched_strategy (noise | random_gt)',
    )
    parser.add_argument(
        '--iters', type=int, default=10,
        help='运行 iteration 数',
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='设备 (cpu | cuda:0 | cuda:1)',
    )
    args = parser.parse_args()

    if args.mode == 'single':
        run_grad_diagnosis(
            config_path=args.config,
            matcher_type=args.matcher_type,
            unmatched_strategy=args.unmatched_strategy,
            num_iters=args.iters,
            device=args.device,
            label='single',
        )
    else:
        # 对比 baseline / 方案A / 方案B
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
        compare_configs(configs, num_iters=args.iters, device=args.device)


if __name__ == '__main__':
    main()
