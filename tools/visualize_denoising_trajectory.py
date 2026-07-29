"""去噪轨迹可视化：展示 KaryoFlow 每个目标匹配的预测框在各步的位置变化

核心思路：
  1. 用 Hungarian 匹配将每个 GT 框与最终步的预测框一一对应
  2. 追踪每个 GT 匹配到的 proposal 在所有步中的位置
  3. 每步只画这些匹配到的框，用颜色区分不同目标，展示从随机→目标的收敛过程

用法:
    # 批量生成所有组合
    conda run -n chromo python3 tools/visualize_denoising_trajectory.py --batch-all

    # 单次生成
    conda run -n chromo python3 tools/visualize_denoising_trajectory.py \
        experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \
        --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \
        --image-id 131 --solver dpm_solver_pp --steps 4 \
        --output docs/paper/latex/figures/trajectory/dpm_pp_4step_24obj.png
"""

import argparse
import json
import math
import os
import sys
import random

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# 确保项目根目录在 Python path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mmengine.config import Config
from mmengine.runner import Runner


def set_seed(seed: int):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_gt_boxes(ann_file, image_id):
    """从 COCO 标注文件加载指定图像的 GT 框 (xyxy 像素坐标)"""
    with open(ann_file) as f:
        coco = json.load(f)
    gt_boxes = []
    for ann in coco['annotations']:
        if ann['image_id'] == image_id:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
    return np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))


def load_image(ann_file, image_id, data_prefix='valid/'):
    """从 COCO 标注加载指定图像"""
    with open(ann_file) as f:
        coco = json.load(f)
    for img_info in coco['images']:
        if img_info['id'] == image_id:
            data_root = os.path.dirname(os.path.dirname(ann_file))
            img_path = os.path.join(data_root, data_prefix, img_info['file_name'])
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), img_info
    raise ValueError(f"Image ID {image_id} not found in {ann_file}")


def get_ann_file_from_config(cfg):
    """从配置中提取标注文件路径"""
    val_dataloader = cfg.val_dataloader
    if isinstance(val_dataloader, dict):
        dataset_cfg = val_dataloader['dataset']
        if 'ann_file' in dataset_cfg:
            data_root = dataset_cfg.get('data_root', '')
            return os.path.join(data_root, dataset_cfg['ann_file'])
    return None


def build_runner_and_model(cfg_path, gpu_id=0):
    """构建 Runner 和模型，返回 (runner, model, head, dataset, cfg)"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cfg = Config.fromfile(cfg_path)
    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = cfg.val_evaluator

    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer', vis_backends=cfg.vis_backends, name='visualizer'
    )
    cfg.work_dir = '/tmp/trajectory_vis'

    runner = Runner.from_cfg(cfg)

    model = runner.model
    if hasattr(model, 'module'):
        model = model.module
    head = model.bbox_head
    dataset = runner.test_dataloader.dataset

    return runner, model, head, dataset, cfg


def find_image_index(dataset, image_id):
    """在数据集中找到指定 image_id 的索引"""
    for i in range(len(dataset)):
        sample = dataset[i]
        img_id = sample['data_samples'].metainfo.get('img_id')
        if img_id == image_id:
            return i
    return None


def run_inference_with_trajectory(model, head, dataset, image_idx,
                                  solver_type, steps, seed=42):
    """运行推理并获取轨迹"""
    set_seed(seed)

    head.solver_type = solver_type
    head._sampler.solver_type = solver_type
    head._sampler.sampling_timesteps = steps
    head.eval()

    sample = dataset[image_idx]
    dp = model.data_preprocessor
    dp.eval()
    data = dict(
        inputs=sample['inputs'].unsqueeze(0),
        data_samples=[sample['data_samples']],
    )
    processed = dp(data)

    with torch.no_grad():
        x = model.extract_feat(processed['inputs'])
        results, trajectory = head.predict(
            x, processed['data_samples'], rescale=False, return_trajectory=True
        )

    meta = processed['data_samples'][0].metainfo
    return trajectory, meta


def trajectory_to_original_coords(trajectory, meta):
    """将轨迹中的 bboxes 从 resized 空间转换到原始图像空间"""
    scale_factor = meta['scale_factor']  # (w_scale, h_scale)
    sf = np.array([scale_factor[0], scale_factor[1],
                   scale_factor[0], scale_factor[1]])

    converted = []
    for cls_logits, pred_bboxes in trajectory:
        boxes = pred_bboxes[0].cpu().numpy() / sf
        scores = torch.sigmoid(cls_logits[0]).max(-1)[0].cpu().numpy()
        converted.append((boxes, scores))

    return converted


def match_gt_to_proposals(gt_boxes, final_boxes, final_scores):
    """用 Hungarian 匹配将 GT 框与最终步的预测框一一对应

    代价矩阵: 1 - IoU (越大越差)

    返回:
        matched_indices: List[int] — 长度=len(gt_boxes), 每个 GT 对应的 proposal 索引
                         如果 GT 没有匹配到任何 proposal, 则为 -1
    """
    n_gt = len(gt_boxes)
    n_pred = len(final_boxes)

    if n_gt == 0 or n_pred == 0:
        return [-1] * n_gt

    # 计算代价矩阵
    cost_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
    for i in range(n_gt):
        for j in range(n_pred):
            iou = _compute_iou(gt_boxes[i], final_boxes[j])
            cost_matrix[i, j] = 1.0 - iou

    # Hungarian 匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 构建匹配结果
    matched_indices = [-1] * n_gt
    for r, c in zip(row_ind, col_ind):
        # 只保留 IoU > 0.01 的匹配 (过滤掉完全不重叠的)
        if cost_matrix[r, c] < 0.99:
            matched_indices[r] = c

    return matched_indices


def _compute_iou(box1, box2):
    """计算两个 xyxy 框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def compute_crop_region(gt_boxes, img_h, img_w, margin=60, min_size=256):
    """计算 GT 框的包围盒 + margin 作为裁剪区域"""
    if len(gt_boxes) == 0:
        return None
    x1 = max(int(gt_boxes[:, 0].min()) - margin, 0)
    y1 = max(int(gt_boxes[:, 1].min()) - margin, 0)
    x2 = min(int(gt_boxes[:, 2].max()) + margin, img_w)
    y2 = min(int(gt_boxes[:, 3].max()) + margin, img_h)
    if x2 - x1 < min_size:
        cx = (x1 + x2) // 2
        x1 = max(cx - min_size // 2, 0)
        x2 = min(cx + min_size // 2, img_w)
    if y2 - y1 < min_size:
        cy = (y1 + y2) // 2
        y1 = max(cy - min_size // 2, 0)
        y2 = min(cy + min_size // 2, img_h)
    return (x1, y1, x2, y2)


def _clip_box_to_crop(box, crop_region):
    """将框坐标裁剪到 crop 区域内"""
    bx1, by1, bx2, by2 = box
    if crop_region is not None:
        cx1, cy1, cx2, cy2 = crop_region
        bx1 = max(bx1 - cx1, 0)
        by1 = max(by1 - cy1, 0)
        bx2 = min(bx2 - cx1, cx2 - cx1)
        by2 = min(by2 - cy1, cy2 - cy1)
    if bx2 <= bx1 or by2 <= by1:
        return None
    return (bx1, by1, bx2, by2)


def visualize_trajectory(img, gt_boxes, trajectory_converted, solver_type, steps,
                         output_path, seed=42, dpi=200):
    """生成去噪轨迹可视化图

    每步只画 GT 匹配到的预测框，用颜色区分不同目标。
    GT 用绿色虚线，预测框用不同颜色实线，颜色随步数变化展示收敛过程。

    Args:
        img: 原始图像 (H, W, 3) RGB
        gt_boxes: GT 框 (N, 4) xyxy 原始坐标
        trajectory_converted: List[(boxes, scores)] 已转换到原始坐标空间
        solver_type: solver 名称
        steps: 步数
    """
    n_steps = len(trajectory_converted)
    n_gt = len(gt_boxes)

    # === 匹配 GT → 最终步的 proposal ===
    final_boxes, final_scores = trajectory_converted[-1]
    matched_indices = match_gt_to_proposals(gt_boxes, final_boxes, final_scores)
    n_matched = sum(1 for idx in matched_indices if idx >= 0)
    print(f"  matched {n_matched}/{n_gt} GT boxes to proposals")

    # === 为每个 GT 目标分配颜色 ===
    # 使用 tab20 colormap 区分不同目标
    gt_colors = plt.colormaps['tab20'](
        np.linspace(0, 1, max(n_gt, 1))
    )

    # === 布局: (steps+1) 列 ===
    n_cols = n_steps + 1  # +1 for GT only panel
    fig_w = min(3.0 * n_cols, 22)
    fig_h = 3.8
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h))
    if n_cols == 1:
        axes = [axes]

    # 计算裁剪区域
    img_h, img_w = img.shape[:2]
    crop = compute_crop_region(gt_boxes, img_h, img_w)

    # Solver 显示名称
    solver_display = {
        'euler': 'Euler',
        'heun': 'Heun',
        'dpm_solver_pp': 'DPM-Solver++',
    }.get(solver_type, solver_type)

    # 时间步值 (RF: t 从 1.0 → 0)
    t_vals = [1.0 - i / steps for i in range(steps)]

    # --- (a) GT only: 只画 GT 框，每个框用不同颜色 ---
    if crop is not None:
        cx1, cy1, cx2, cy2 = crop
        axes[0].imshow(img[cy1:cy2, cx1:cx2])
    else:
        axes[0].imshow(img)

    for gt_idx, gt_box in enumerate(gt_boxes):
        clipped = _clip_box_to_crop(gt_box, crop)
        if clipped is None:
            continue
        bx1, by1, bx2, by2 = clipped
        color = gt_colors[gt_idx]
        rect = patches.Rectangle(
            (bx1, by1), bx2 - bx1, by2 - by1,
            linewidth=1.5, edgecolor=color, facecolor='none',
            linestyle='--', alpha=0.85
        )
        axes[0].add_patch(rect)

    axes[0].set_title('GT', fontsize=9, pad=2)
    axes[0].set_axis_off()

    # --- (b~) 每步预测: 只画 GT 匹配到的 proposal 框 ---
    for step_idx, (boxes, scores) in enumerate(trajectory_converted):
        t_curr = t_vals[step_idx] if step_idx < len(t_vals) else 0.0
        if step_idx == 0:
            title = f'Step 1 ($t$=1.0)'
        elif step_idx == n_steps - 1:
            title = f'Step {step_idx+1} ($t$=0)'
        else:
            title = f'Step {step_idx+1} ($t$={t_curr:.2f})'

        if crop is not None:
            axes[step_idx + 1].imshow(img[cy1:cy2, cx1:cx2])
        else:
            axes[step_idx + 1].imshow(img)

        # 画 GT 框 (灰色淡虚线作为背景参考)
        for gt_box in gt_boxes:
            clipped = _clip_box_to_crop(gt_box, crop)
            if clipped is None:
                continue
            bx1, by1, bx2, by2 = clipped
            rect = patches.Rectangle(
                (bx1, by1), bx2 - bx1, by2 - by1,
                linewidth=0.5, edgecolor='gray', facecolor='none',
                linestyle=':', alpha=0.4
            )
            axes[step_idx + 1].add_patch(rect)

        # 画匹配到的预测框 (每个 GT 用不同颜色)
        for gt_idx, prop_idx in enumerate(matched_indices):
            if prop_idx < 0 or prop_idx >= len(boxes):
                continue
            pred_box = boxes[prop_idx]
            clipped = _clip_box_to_crop(pred_box, crop)
            if clipped is None:
                continue
            bx1, by1, bx2, by2 = clipped
            color = gt_colors[gt_idx]
            # 透明度随步数递增 (早期模糊→后期清晰)
            alpha = 0.4 + 0.5 * (step_idx / max(n_steps - 1, 1))
            lw = 0.8 + 1.0 * (step_idx / max(n_steps - 1, 1))
            rect = patches.Rectangle(
                (bx1, by1), bx2 - bx1, by2 - by1,
                linewidth=lw, edgecolor=color, facecolor='none',
                linestyle='-', alpha=alpha
            )
            axes[step_idx + 1].add_patch(rect)

        axes[step_idx + 1].set_title(title, fontsize=9, pad=2)
        axes[step_idx + 1].set_axis_off()

    # === Legend ===
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle=':', linewidth=0.5,
               label='GT (reference)'),
        Line2D([0], [0], color='tab:blue', linestyle='-', linewidth=1.5,
               label='Matched pred (per-target color)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=2, fontsize=8, framealpha=0.7,
               bbox_to_anchor=(0.5, 0.99))

    plt.suptitle(
        f'{solver_display} × {steps} steps',
        fontsize=11, y=1.08
    )
    plt.subplots_adjust(left=0.01, right=0.97, top=0.88, bottom=0.02, wspace=0.05)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"[SAVED] {output_path}")


def generate_one(runner, model, head, dataset, cfg,
                 image_id, solver_type, steps, ann_file, output_path,
                 seed=42, dpi=200):
    """生成单张轨迹可视化图"""
    img_idx = find_image_index(dataset, image_id)
    if img_idx is None:
        print(f"[WARN] image_id={image_id} not found, skipping")
        return False

    img, img_info = load_image(ann_file, image_id)
    gt_boxes = load_gt_boxes(ann_file, image_id)
    print(f"  image_id={image_id}, img_size={img.shape}, n_gt={len(gt_boxes)}")

    trajectory, meta = run_inference_with_trajectory(
        model, head, dataset, img_idx, solver_type, steps, seed=seed
    )
    print(f"  trajectory: {len(trajectory)} steps")

    trajectory_converted = trajectory_to_original_coords(trajectory, meta)

    visualize_trajectory(
        img, gt_boxes, trajectory_converted, solver_type, steps,
        output_path, seed=seed, dpi=dpi
    )
    return True


def main():
    parser = argparse.ArgumentParser(description='Denoising Trajectory Visualization')
    parser.add_argument('config', nargs='?', help='Config file path')
    parser.add_argument('--checkpoint', help='Checkpoint path')
    parser.add_argument('--image-id', type=int, default=None, help='COCO image_id')
    parser.add_argument('--solver', type=str, default=None,
                        choices=['euler', 'heun', 'dpm_solver_pp'])
    parser.add_argument('--steps', type=int, default=None, help='Sampling steps')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--dpi', type=int, default=200, help='Output DPI')
    parser.add_argument('--batch-all', action='store_true',
                        help='Generate all solver×steps for both datasets')
    args = parser.parse_args()

    if args.batch_all:
        run_batch_all(args)
        return

    if args.config is None or args.checkpoint is None:
        parser.error('config and --checkpoint are required (unless --batch-all)')

    runner, model, head, dataset, cfg = build_runner_and_model(
        args.config, gpu_id=args.gpu_id
    )
    runner.load_checkpoint(args.checkpoint)

    solver_type = args.solver or cfg.model.bbox_head.get('solver_type', 'euler')
    steps = args.steps or cfg.model.bbox_head.get('sampling_timesteps', 4)

    ann_file = get_ann_file_from_config(cfg)
    if ann_file is None:
        raise RuntimeError("Cannot determine annotation file from config")

    image_id = args.image_id
    if image_id is None:
        with open(ann_file) as f:
            coco = json.load(f)
        from collections import Counter
        img_box_count = Counter(ann['image_id'] for ann in coco['annotations'])
        image_id = min(img_box_count, key=img_box_count.get)
        print(f"[AUTO] Selected image_id={image_id}")

    output = args.output or f'docs/paper/latex/figures/trajectory/{solver_type}_{steps}step_vis.png'
    generate_one(runner, model, head, dataset, cfg,
                 image_id, solver_type, steps, ann_file, output,
                 seed=args.seed, dpi=args.dpi)


def run_batch_all(args):
    """批量生成所有组合"""
    outdir = 'docs/paper/latex/figures/trajectory'
    os.makedirs(outdir, exist_ok=True)

    datasets = [
        {
            'name': '24obj',
            'cfg_path': 'experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py',
            'ckpt': 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth',
            'image_id': 131,
        },
        {
            'name': 'chr2024',
            'cfg_path': 'experiments/configs/ldmdet/a4_dpm_pp_chr2024.py',
            'ckpt': 'work_dirs/a4_dpm_pp_chr2024_seed42/best_coco_bbox_mAP_epoch_49.pth',
            'image_id': 198,
        },
    ]

    combos = [
        ('euler', 1),
        ('euler', 4),
        ('euler', 8),
        ('heun', 1),
        ('heun', 4),
        ('dpm_solver_pp', 1),
        ('dpm_solver_pp', 4),
    ]

    for ds_info in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_info['name']}")
        print(f"{'='*60}")

        runner, model, head, dataset, cfg = build_runner_and_model(
            ds_info['cfg_path'], gpu_id=args.gpu_id
        )
        runner.load_checkpoint(ds_info['ckpt'])

        ann_file = get_ann_file_from_config(cfg)
        if ann_file is None:
            print(f"[ERROR] Cannot get ann_file for {ds_info['name']}")
            continue

        for solver_type, steps in combos:
            output = os.path.join(
                outdir, f'{solver_type}_{steps}step_{ds_info["name"]}.png'
            )
            print(f"\n--- {solver_type} × {steps} steps ---")
            try:
                generate_one(
                    runner, model, head, dataset, cfg,
                    ds_info['image_id'], solver_type, steps, ann_file, output,
                    seed=args.seed, dpi=args.dpi,
                )
            except Exception as e:
                print(f"  [ERROR] {solver_type}×{steps}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()
