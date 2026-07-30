"""白盒插桩分析运行脚本

对指定 checkpoint 运行 3 个白盒分析器, 生成 JSON 报告.

Usage:
    # 单个 ckpt
    python experiments/analysis/instrumentation/run_instrumentation.py \
        --config experiments/configs/ldmdet/directions/nonlinear_trajectory/rf_heun_adaln.py \
        --ckpt work_dirs/multi_seed_aug/rf_heun_adaln/seed_42/best_coco_bbox_mAP_epoch_102.pth \
        --name baseline_aug \
        --analyzers trajectory roi_feature head_output \
        --num-samples 50

    # 批量 (按方案矩阵)
    python experiments/analysis/instrumentation/run_instrumentation.py --batch
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 触发 mmdet 模块注册
import mmdet.models  # noqa: F401
import mmdet.datasets  # noqa: F401
from mmengine.config import Config
from mmengine.registry import init_default_scope
init_default_scope('mmdet')

from experiments.mmdet_bridge.registry import register_all
register_all()

from experiments.analysis.instrumentation import (
    TrajectoryCollector,
    TrajectoryAnalyzer,
    RoIFeatureCollector,
    RoIFeatureAnalyzer,
    HeadOutputCollector,
    HeadOutputAnalyzer,
)

# 染色体分组映射 (24 类: RF+Heun-3, B4-5, C6-12, D13-15, E16-18, F19-20, G21-22, X, Y)
CHROMOSOME_GROUPS = {
    0: [0, 1, 2],                  # A 组
    1: [3, 4],                      # B 组
    2: [5, 6, 7, 8, 9, 10, 11],    # C 组
    3: [12, 13, 14],               # D 组
    4: [15, 16, 17],               # E 组
    5: [18, 19],                    # F 组
    6: [20, 21],                    # G 组
    7: [22],                        # X
    8: [23],                        # Y
}

# 方案矩阵: name -> (config, ckpt, analyzers)
BATCH_PLAN = {
    'baseline_aug': {
        'config': 'experiments/configs/ldmdet/directions/nonlinear_trajectory/rf_heun_adaln.py',
        'ckpt': 'work_dirs/multi_seed_aug/rf_heun_adaln/seed_42/best_coco_bbox_mAP_epoch_102.pth',
        'analyzers': ['trajectory', 'roi_feature', 'head_output'],
    },
    'direction_d': {
        'config': 'experiments/configs/ldmdet/direction_d_box_refine.py',
        'ckpt': 'work_dirs/direction_exps/direction_d_box_refine/best_coco_bbox_mAP_epoch_55.pth',
        'analyzers': ['trajectory'],
    },
    'focal_gamma_3': {
        'config': 'experiments/configs/bottleneck/focal_gamma_3.py',
        'ckpt': 'work_dirs/bottleneck/ablation/focal_gamma_3/best_coco_bbox_mAP_epoch_88.pth',
        'analyzers': ['head_output'],
    },
    # 24obj 数据集对照 (架构/损失与新数据集一致, 仅数据集不同)
    'ghss_24obj': {
        'config': 'experiments/configs/multiset/chromo_24obj.py',
        'ckpt': 'work_dirs/24obj_ablation/ghss/seed_42/best_coco_bbox_mAP_epoch_83.pth',
        'analyzers': ['trajectory', 'roi_feature', 'head_output'],
    },
    'random_24obj': {
        'config': 'experiments/configs/multiset/chromo_24obj_random.py',
        'ckpt': 'work_dirs/24obj_ablation/random/seed_42/best_coco_bbox_mAP_epoch_59.pth',
        'analyzers': ['trajectory', 'roi_feature', 'head_output'],
    },
}


def load_model(config_path: str, ckpt_path: str, device: str = 'cuda'):
    """加载模型和 checkpoint."""
    cfg = Config.fromfile(config_path)
    # 减小 num_proposals 加速 (分析不需要全量 proposals)
    # 注意: 不能改, 否则 ckpt 加载会 mismatch. 保持原配置.
    from mmdet.registry import MODELS
    model = MODELS.build(cfg.model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, cfg


def load_val_samples(cfg, num_samples: int = 50):
    """加载验证集样本 (返回原始图片和标注)."""
    from pycocotools.coco import COCO
    from mmcv import imread

    val_cfg = cfg.val_dataloader.dataset
    ann_file = osp.join(val_cfg.data_root, val_cfg.ann_file)
    img_prefix = osp.join(val_cfg.data_root, val_cfg.data_prefix['img'])

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    # 采样 (均匀采样, 覆盖各类)
    if num_samples < len(img_ids):
        indices = np.linspace(0, len(img_ids) - 1, num_samples, dtype=int)
        img_ids = [img_ids[i] for i in indices]

    samples = []
    cat_ids = coco.getCatIds()
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = osp.join(img_prefix, img_info['file_name'])
        img = imread(img_path)
        if img is None:
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = []
        gt_labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(cat_id_to_idx[ann['category_id']])

        samples.append({
            'img_id': img_id,
            'img': img,
            'gt_boxes': torch.tensor(gt_boxes).float() if gt_boxes else torch.zeros(0, 4),
            'gt_labels': torch.tensor(gt_labels).long() if gt_labels else torch.zeros(0, dtype=torch.long),
            'img_meta': {
                'img_shape': img.shape,
                'ori_shape': img.shape,
                'scale_factor': 1.0,
                'img_id': img_id,
            },
        })

    return samples, cat_id_to_idx


def run_trajectory_analysis(model, samples, device, output_path):
    """运行扩散采样轨迹分析.

    修复: 每张图单独分析 (单图 sampling_timesteps 步), 再聚合统计量。
    之前错误地把所有图的所有步混入同一个 collector, 导致 n_steps=N_images*steps,
    且跨图 IoU 计算无意义。
    """
    print('  [Trajectory] 采集轨迹 (每图单独分析)...')
    head = model.bbox_head

    per_image_reports = []
    n_collected = 0
    for sample in samples:
        img = torch.from_numpy(sample['img']).to(device).float()
        if img.dim() == 3:
            img = img.unsqueeze(0)
        # 归一化
        img = img.permute(0, 3, 1, 2) / 255.0

        img_metas = [sample['img_meta']]

        with torch.no_grad():
            # backbone + neck
            feats = model.backbone(img)
            feats = model.neck(feats)

            # 采集轨迹
            try:
                results, trajectory = head.predict(feats, img_metas, rescale=False, return_trajectory=True)
                # 每图独立 collector + analyzer
                img_collector = TrajectoryCollector()
                for step_idx, (cls_logits, pred_bboxes) in enumerate(trajectory):
                    # 模拟 x0_raw (predict 不直接返回, 用 pred_bboxes 近似)
                    # 注意: 真实 x0_raw 需要修改 predict 内部, 这里用 pred_bboxes 作为近似
                    img_collector.record(
                        step_idx=step_idx,
                        t_curr=1.0 - step_idx * (1.0 / max(len(trajectory), 1)),
                        t_next=1.0 - (step_idx + 1) * (1.0 / max(len(trajectory), 1)),
                        cls_logits=cls_logits,
                        pred_bboxes=pred_bboxes,
                        x0_raw=pred_bboxes,  # 近似
                        x_raw_before_step=pred_bboxes,
                        x_raw_after_step=pred_bboxes,
                    )
                img_report = TrajectoryAnalyzer(img_collector.trajectory).full_report()
                per_image_reports.append(img_report)
                n_collected += 1
            except Exception as e:
                print(f'    [WARN] img_id={sample["img_id"]} 轨迹采集失败: {e}')
                continue

    print(f'  [Trajectory] 采集 {n_collected}/{len(samples)} 张图')

    # 聚合每图报告 → 跨图统计量
    report = _aggregate_trajectory_reports(per_image_reports)
    report['n_images'] = n_collected
    report['n_steps'] = per_image_reports[0]['n_steps'] if per_image_reports else 0
    report['per_image_count'] = len(per_image_reports)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f'  [Trajectory] 报告保存到 {output_path}')
    return report


def _aggregate_trajectory_reports(per_image_reports: list) -> dict:
    """聚合每图 trajectory 报告为跨图统计量.

    每图 n_steps (= sampling_timesteps) 单独计算后, 对同名字段取均值/比例。
    """
    if not per_image_reports:
        return {}

    import numpy as np

    def _safe_mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    def _safe_ratio(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    # box_evolution
    box_conv = [r['box_evolution']['convergence_step'] for r in per_image_reports]
    box_iou_final = [r['box_evolution']['per_step_iou_to_final'][-1]
                     for r in per_image_reports if r['box_evolution']['per_step_iou_to_final']]

    # cls_convergence
    cls_conv = [r['cls_convergence']['convergence_step'] for r in per_image_reports]
    cls_converged = [r['cls_convergence']['is_converged'] for r in per_image_reports]

    # x0_quality
    early_x0 = [r['x0_quality']['early_x0_quality'] for r in per_image_reports]
    x0_stab = [r['x0_quality']['x0_stability'] for r in per_image_reports]

    # renewal
    renew_eff = [r['renewal']['renewal_effective'] for r in per_image_reports]
    renew_dir = [r['renewal']['renewal_direction'] for r in per_image_reports]

    aggregated = {
        'box_evolution': {
            'convergence_step_mean': _safe_mean(box_conv),
            'convergence_step_distribution': {
                str(k): int(v) for k, v in __import__('collections').Counter(box_conv).items()
            },
            'final_iou_to_final_mean': _safe_mean(box_iou_final),
            'n_steps_per_image': per_image_reports[0]['n_steps'],
        },
        'cls_convergence': {
            'convergence_step_mean': _safe_mean(cls_conv),
            'cls_converged_ratio': _safe_ratio(cls_converged),
        },
        'x0_quality': {
            'early_x0_quality_mean': _safe_mean(early_x0),
            'x0_stability_mean': _safe_mean(x0_stab),
        },
        'renewal': {
            'renewal_effective_ratio': _safe_ratio(renew_eff),
            'renewal_direction_mean': _safe_mean(renew_dir),
        },
    }

    # 保留文本结论 (取第一图的作为示例, 标注为示例)
    if per_image_reports:
        aggregated['analysis'] = per_image_reports[0].get('analysis', '')
        aggregated['note'] = 'analysis 字段为示例 (第1张图), 聚合统计量见各 *_mean/*_ratio 字段'

    return aggregated


def run_roi_feature_analysis(model, samples, device, output_path):
    """运行 RoI 特征区分度分析."""
    print('  [RoIFeature] 采集 RoI 特征...')
    collector = RoIFeatureCollector()
    head = model.bbox_head

    # 注册 hook
    captured = {}
    def roi_hook(module, input, output):
        captured['roi_features'] = output
    handle = head.roi_extractor.register_forward_hook(roi_hook)

    n_collected = 0
    for sample in samples:
        img = torch.from_numpy(sample['img']).to(device).float()
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) / 255.0
        img_metas = [sample['img_meta']]

        with torch.no_grad():
            feats = model.backbone(img)
            feats = model.neck(feats)
            try:
                _ = head.predict(feats, img_metas, rescale=False)
            except Exception as e:
                continue

        # 从 captured 获取 RoI 特征
        if 'roi_features' not in captured:
            continue

        roi_feat = captured['roi_features']
        if roi_feat.dim() == 4:
            roi_feat_pooled = roi_feat.mean(dim=[2, 3])
        else:
            roi_feat_pooled = roi_feat

        # 用 GT 标签近似 (RoI 对应 proposal, 这里用 GT 标签做近似分析)
        # 注意: 真实分析应匹配 proposal 到 GT, 这里简化用 GT 数量截断
        n_roi = roi_feat_pooled.shape[0]
        n_gt = sample['gt_labels'].shape[0]
        if n_gt == 0:
            continue
        # 循环使用 GT 标签 (简化)
        labels = sample['gt_labels'][torch.arange(n_roi) % n_gt]
        # 尺度 = GT 框面积 (近似)
        if sample['gt_boxes'].shape[0] > 0:
            areas = ((sample['gt_boxes'][:, 2] - sample['gt_boxes'][:, 0]) *
                      (sample['gt_boxes'][:, 3] - sample['gt_boxes'][:, 1]))
            scales = areas[torch.arange(n_roi) % n_gt]
        else:
            scales = torch.ones(n_roi) * 1000

        collector.record(
            roi_features=roi_feat_pooled,
            labels=labels,
            scales=scales,
        )
        n_collected += 1

    handle.remove()
    print(f'  [RoIFeature] 采集 {n_collected}/{len(samples)} 张图, 共 {collector.features.shape[0] if collector.features is not None else 0} 个 RoI')

    if collector.features is None or collector.features.shape[0] == 0:
        print('  [RoIFeature] 无数据, 跳过')
        return None

    analyzer = RoIFeatureAnalyzer(collector)
    report = analyzer.full_report(group_mapping=CHROMOSOME_GROUPS)
    report['n_images'] = n_collected

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f'  [RoIFeature] 报告保存到 {output_path}')
    return report


def run_head_output_analysis(model, samples, device, output_path, head_index: int = 0):
    """运行 cls/reg 头输出分析.

    D'1 验证增强: 同时采集 proposal boxes, 用 GT IoU 匹配构建 fg_masks,
    使 analyze_reg_distribution 能区分正/负样本的 delta 统计。
    教训来自 D'2 失败: 对所有 proposal 求均值会掩盖正样本行为 (被~90%背景主导)。

    Args:
        head_index: 采集哪个 head 的输出 (0=第一个, -1=最后一个). 默认 0.
                    D'1 验证建议同时运行 0 和 -1 对比: 第 0 个 head 输入是初始噪声
                    proposal, 最后一个 head 输入是经迭代回归的 proposal.
    """
    print(f'  [HeadOutput] 采集头输出 (head_index={head_index})...')
    collector = HeadOutputCollector()
    head = model.bbox_head
    sh = head.head_series[head_index]

    # 注册 hook
    cls_captured = {}
    reg_captured = {}
    proposal_captured = {}  # D'1: 采集 proposal boxes
    def cls_hook(m, i, o): cls_captured['out'] = o
    def reg_hook(m, i, o): reg_captured['out'] = o
    # D'1: forward pre hook 获取 single_head 输入的 bboxes (proposal boxes)
    def sh_pre_hook(module, args):
        # single_head.forward(features, bboxes, proposals, pooler, time_emb)
        # bboxes 是第 2 个参数 (args[1])
        if len(args) >= 2:
            proposal_captured['bboxes'] = args[1]
    h1 = sh.cls_head.register_forward_hook(cls_hook)
    h2 = sh.reg_head.register_forward_hook(reg_hook)
    h3 = sh.register_forward_pre_hook(sh_pre_hook)

    n_collected = 0
    for sample in samples:
        img = torch.from_numpy(sample['img']).to(device).float()
        if img.dim() == 3:
            img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) / 255.0
        img_metas = [sample['img_meta']]

        with torch.no_grad():
            feats = model.backbone(img)
            feats = model.neck(feats)
            try:
                _ = head.predict(feats, img_metas, rescale=False)
            except Exception as e:
                continue

        if 'out' not in cls_captured or 'out' not in reg_captured:
            continue

        cls_out = cls_captured['out']
        reg_out = reg_captured['out']
        if cls_out.dim() == 3:
            cls_out = cls_out.reshape(-1, cls_out.shape[-1])
            reg_out = reg_out.reshape(-1, reg_out.shape[-1])

        n = cls_out.shape[0]
        n_gt = sample['gt_labels'].shape[0]
        if n_gt == 0:
            labels = torch.randint(0, cls_out.shape[1], (n,))
        else:
            labels = sample['gt_labels'][torch.arange(n) % n_gt]

        # D'1: 用 GT IoU 匹配构建 fg_masks (IoU > 0.5 为正样本)
        fg_masks = None
        if 'bboxes' in proposal_captured and n_gt > 0:
            prop_bboxes = proposal_captured['bboxes']  # [bs, N, 4]
            if prop_bboxes.dim() == 3:
                prop_bboxes_flat = prop_bboxes.reshape(-1, 4)  # [bs*N, 4]
            else:
                prop_bboxes_flat = prop_bboxes
            # 只取与 cls_out 对应数量的 proposal (bs*N)
            prop_bboxes_flat = prop_bboxes_flat[:n].cpu()
            gt_boxes = sample['gt_boxes'].cpu()  # [n_gt, 4]
            # 计算 IoU: [n, 4] vs [n_gt, 4] → [n, n_gt]
            from torchvision.ops import box_iou
            iou_matrix = box_iou(prop_bboxes_flat, gt_boxes)  # [n, n_gt]
            max_iou, _ = iou_matrix.max(dim=1)  # [n]
            fg_masks = (max_iou > 0.5)  # [n] bool

        collector.record(
            fc_feature=cls_out,  # 近似 (实际应 hook fc_feature)
            cls_logits=cls_out,
            reg_deltas=reg_out,
            labels=labels,
            fg_masks=fg_masks,
        )
        n_collected += 1

    h1.remove()
    h2.remove()
    h3.remove()
    print(f'  [HeadOutput] 采集 {n_collected}/{len(samples)} 张图, 共 {collector.cls_logits.shape[0] if collector.cls_logits is not None else 0} 个样本')

    if collector.cls_logits is None or collector.cls_logits.shape[0] == 0:
        print('  [HeadOutput] 无数据, 跳过')
        return None

    analyzer = HeadOutputAnalyzer(collector)
    report = analyzer.full_report(group_mapping=CHROMOSOME_GROUPS)
    report['n_images'] = n_collected

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f'  [HeadOutput] 报告保存到 {output_path}')
    return report


def run_single(name, config_path, ckpt_path, analyzers, num_samples, device, output_dir, head_index: int = 0):
    """运行单个 ckpt 的分析."""
    print(f'\n{"="*60}')
    print(f'分析: {name}')
    print(f'  config: {config_path}')
    print(f'  ckpt: {ckpt_path}')
    print(f'  analyzers: {analyzers}')
    print(f'  head_index: {head_index}')
    print(f'{"="*60}')

    # 输出目录
    out_dir = Path(output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print('加载模型...')
    model, cfg = load_model(config_path, ckpt_path, device)

    # 加载样本
    print(f'加载 {num_samples} 个验证集样本...')
    samples, _ = load_val_samples(cfg, num_samples)
    print(f'  实际加载 {len(samples)} 个样本')

    reports = {}
    for analyzer_name in analyzers:
        if analyzer_name == 'trajectory':
            reports['trajectory'] = run_trajectory_analysis(
                model, samples, device, out_dir / 'trajectory_report.json')
        elif analyzer_name == 'roi_feature':
            reports['roi_feature'] = run_roi_feature_analysis(
                model, samples, device, out_dir / 'roi_feature_report.json')
        elif analyzer_name == 'head_output':
            out_name = 'head_output_report.json' if head_index == 0 else f'head_output_idx{head_index}_report.json'
            reports['head_output'] = run_head_output_analysis(
                model, samples, device, out_dir / out_name, head_index=head_index)

    # 保存汇总
    summary = {
        'name': name,
        'config': config_path,
        'ckpt': ckpt_path,
        'num_samples': len(samples),
        'analyzers_run': analyzers,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return reports


def main():
    parser = argparse.ArgumentParser(description='白盒插桩分析')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--ckpt', help='checkpoint 路径')
    parser.add_argument('--name', help='实验名称')
    parser.add_argument('--analyzers', nargs='+',
                        choices=['trajectory', 'roi_feature', 'head_output'],
                        help='要运行的分析器')
    parser.add_argument('--num-samples', type=int, default=50, help='验证集采样数')
    parser.add_argument('--device', default='cuda:0', help='设备')
    parser.add_argument('--head-index', type=int, default=0,
                        help='采集哪个 head 的输出 (0=第一个, -1=最后一个). D\'1 验证用')
    parser.add_argument('--output-dir', default='work_dirs/instrumentation', help='输出目录')
    parser.add_argument('--batch', action='store_true', help='批量运行方案矩阵')
    parser.add_argument('--only', nargs='+', default=None,
                        help='批量模式下只运行指定的 name (空格分隔)')
    args = parser.parse_args()

    if args.batch:
        # 批量运行方案矩阵
        all_reports = {}
        for name, plan in BATCH_PLAN.items():
            if args.only is not None and name not in args.only:
                continue
            if not osp.exists(plan['ckpt']):
                print(f'[SKIP] {name}: ckpt 不存在 {plan["ckpt"]}')
                continue
            reports = run_single(
                name=name,
                config_path=plan['config'],
                ckpt_path=plan['ckpt'],
                analyzers=plan['analyzers'],
                num_samples=args.num_samples,
                device=args.device,
                output_dir=args.output_dir,
            )
            all_reports[name] = reports

        # 保存汇总对比 (--only 模式下合并已有 comparison.json, 避免丢失历史结果)
        comparison_path = Path(args.output_dir) / 'comparison.json'
        comparison = {}
        if args.only is not None and comparison_path.exists():
            with open(comparison_path) as f:
                comparison = json.load(f)
        for name, reports in all_reports.items():
            comparison[name] = {}
            for analyzer, report in reports.items():
                if report is not None:
                    # 提取关键指标
                    if analyzer == 'trajectory':
                        # 修复后字段: 每图独立分析后聚合
                        be = report['box_evolution']
                        cc = report['cls_convergence']
                        xq = report['x0_quality']
                        rn = report['renewal']
                        comparison[name]['trajectory'] = {
                            'n_steps_per_image': be.get('n_steps_per_image', be.get('n_steps', 0)),
                            'box_convergence_step_mean': be.get('convergence_step_mean',
                                                                be.get('convergence_step')),
                            'box_convergence_dist': be.get('convergence_step_distribution', {}),
                            'cls_convergence_step_mean': cc.get('convergence_step_mean'),
                            'cls_converged_ratio': cc.get('cls_converged_ratio',
                                                          (1.0 if cc.get('is_converged') else 0.0)),
                            'early_x0_quality_mean': xq.get('early_x0_quality_mean',
                                                            xq.get('early_x0_quality')),
                            'x0_stability_mean': xq.get('x0_stability_mean'),
                            'renewal_effective_ratio': rn.get('renewal_effective_ratio',
                                                              (1.0 if rn.get('renewal_effective') else 0.0)),
                        }
                    elif analyzer == 'roi_feature':
                        comparison[name]['roi_feature'] = {
                            'same_group_too_similar': report['same_group_similarity']['same_group_too_similar'],
                            'cross_group_sim': report['same_group_similarity']['cross_group_similarity'],
                            'scale_affects': report['scale_feature_norm']['scale_affects_feature'],
                        }
                    elif analyzer == 'head_output':
                        comparison[name]['head_output'] = {
                            'class_collapse': report['cls_distribution']['class_collapse'],
                            'reg_conservative': report['reg_distribution']['is_conservative'],
                            'hard_samples': report['hard_samples']['hard_sample_count'],
                            'overconfident': report['hard_samples']['overconfident'],
                        }

        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)
        print(f'\n对比汇总保存到 {comparison_path}')

    else:
        # 单个运行
        if not args.config or not args.ckpt or not args.name:
            parser.error('单次运行需要 --config, --ckpt, --name')
        analyzers = args.analyzers or ['trajectory', 'roi_feature', 'head_output']
        run_single(
            name=args.name,
            config_path=args.config,
            ckpt_path=args.ckpt,
            analyzers=analyzers,
            num_samples=args.num_samples,
            device=args.device,
            output_dir=args.output_dir,
            head_index=args.head_index,
        )


if __name__ == '__main__':
    main()
