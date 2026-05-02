"""
KaryoFlow 核型图可视化工具

功能:
1. visualize_karyotype_grid  — 将 46 个染色体 crop 按标准核型图布局排列显示
2. visualize_prediction_vs_gt — 并排对比预测排列与 GT 排列
3. visualize_assignment_matrix — 显示模型输出的 slot×detection 分配热图
4. visualize_trajectory        — 显示迭代 unmask 过程的动画

用法:
    python tools/visualize_karyotype.py \
        --coco-json /data/.../train/_annotations.coco.json \
        --kf-json   /data/.../train_karyoflow.json \
        --image-dir /data/.../train \
        --image-id  123 \
        --output    karyotype_123.png
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # 无头模式，服务器上不报错
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from karyoflow.constants import (
    CLASS_TO_SLOTS,
    DENVER_GROUPS,
    ID_TO_CLASS,
    NUM_SLOTS,
    SLOT_ORDER,
    SLOT_TO_GROUP,
)

# ============================================================
# Denver 分组颜色
# ============================================================

GROUP_COLORS: Dict[str, str] = {
    "A":   "#E63946",   # 红
    "B":   "#F4A261",   # 橙
    "C":   "#2A9D8F",   # 青绿
    "D":   "#457B9D",   # 蓝
    "E":   "#6A0572",   # 紫
    "F":   "#84A98C",   # 绿
    "G":   "#BC6C25",   # 棕
    "Sex": "#555555",   # 灰
}


def _load_image_and_anns(
    coco_json: str, kf_json: str, image_dir: str, image_id: int
) -> Tuple[Image.Image, List[dict], List[int]]:
    """加载图像、标注、排列"""
    with open(coco_json) as f:
        coco = json.load(f)
    with open(kf_json) as f:
        kf = json.load(f)

    # 找图像
    img_info = next((i for i in coco["images"] if i["id"] == image_id), None)
    if img_info is None:
        raise ValueError(f"image_id={image_id} not found")

    img_path = os.path.join(image_dir, img_info["file_name"])
    image = Image.open(img_path).convert("RGB")

    # 找该图的标注
    anns = [a for a in coco["annotations"] if a["image_id"] == image_id]
    anns.sort(key=lambda a: a["id"])

    # 排列
    kf_ann = kf["karyoflow_annotations"].get(str(image_id))
    permutation = kf_ann["detection_indices"] if kf_ann and kf_ann.get("valid") else None

    return image, anns, permutation


def _crop_chromosome(image: Image.Image, bbox: List[float], pad: float = 0.05) -> Image.Image:
    """从图像裁剪单个染色体，带少量 padding"""
    iw, ih = image.size
    x, y, w, h = bbox
    pad_x = w * pad
    pad_y = h * pad
    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = min(iw, int(x + w + pad_x))
    y2 = min(ih, int(y + h + pad_y))
    return image.crop((x1, y1, x2, y2))


# ============================================================
# 核心可视化函数
# ============================================================

def visualize_karyotype_grid(
    image: Image.Image,
    anns: List[dict],
    permutation: List[int],
    title: str = "Karyotype",
    crop_h: int = 80,
    crop_w: int = 48,
    figsize: Tuple[int, int] = (18, 10),
) -> plt.Figure:
    """将 46 个染色体按标准核型图布局排列显示

    布局:
    - 行 = Denver 分组 (A~G + Sex)
    - 列 = 组内每条染色体 (成对排列)
    """
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    axes.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Denver 分组 → slot 列表
    group_slot_list = []
    for group_name, classes in DENVER_GROUPS.items():
        slots = []
        for cls in classes:
            slots.extend(CLASS_TO_SLOTS[cls])
        group_slot_list.append((group_name, slots))

    n_rows = len(group_slot_list)
    max_cols = max(len(slots) for _, slots in group_slot_list)

    # 计算画布大小
    cell_w = crop_w + 8
    cell_h = crop_h + 20
    total_w = max_cols * cell_w + 60  # 左边留 60px 给组名
    total_h = n_rows * cell_h + 30

    canvas = Image.new("RGB", (total_w, total_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    y_offset = 20
    for row_idx, (group_name, slots) in enumerate(group_slot_list):
        color_hex = GROUP_COLORS[group_name]
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)

        # 组名标签
        draw.text((5, y_offset + crop_h // 2 - 6), group_name, fill=(r, g, b))

        for col_idx, slot_idx in enumerate(slots):
            if permutation is None or slot_idx >= len(permutation):
                continue
            ann_idx = permutation[slot_idx]
            if ann_idx < 0 or ann_idx >= len(anns):
                continue

            ann = anns[ann_idx]
            cls_name = ID_TO_CLASS.get(ann["category_id"], "?")

            # 裁剪染色体
            crop = _crop_chromosome(image, ann["bbox"])
            crop_resized = crop.resize((crop_w, crop_h), Image.LANCZOS)

            x = 60 + col_idx * cell_w
            y = y_offset

            # 粘贴 crop
            canvas.paste(crop_resized, (x, y))

            # 边框（组颜色）
            draw.rectangle([x, y, x + crop_w, y + crop_h], outline=(r, g, b), width=2)

            # 类别标签
            label = f"{cls_name}\n#{slot_idx}"
            draw.text((x + 2, y + crop_h + 2), label, fill=(60, 60, 60))

        y_offset += cell_h

    axes.imshow(np.array(canvas))
    axes.set_title(title, pad=8)

    return fig


def visualize_prediction_vs_gt(
    image: Image.Image,
    anns: List[dict],
    pred_perm: List[int],
    gt_perm: List[int],
    output_path: str,
    crop_h: int = 60,
    crop_w: int = 36,
):
    """并排对比预测排列 vs GT 排列

    - 绿色边框: 预测正确
    - 红色边框: 预测错误
    """
    iw, ih = image.size
    group_slot_list = []
    for group_name, classes in DENVER_GROUPS.items():
        slots = []
        for cls in classes:
            slots.extend(CLASS_TO_SLOTS[cls])
        group_slot_list.append((group_name, slots))

    n_rows = len(group_slot_list)
    max_cols = max(len(slots) for _, slots in group_slot_list)

    cell_w = crop_w + 6
    cell_h = crop_h + 18
    panel_w = max_cols * cell_w + 50
    panel_h = n_rows * cell_h + 30

    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2 / 80, panel_h / 80 + 1))
    fig.suptitle("Prediction (left) vs Ground Truth (right)", fontsize=13, fontweight="bold")

    for ax_idx, (perm, label) in enumerate([(pred_perm, "Prediction"), (gt_perm, "Ground Truth")]):
        canvas = Image.new("RGB", (panel_w, panel_h), color=(248, 248, 248))
        draw = ImageDraw.Draw(canvas)

        y_offset = 20
        for group_name, slots in group_slot_list:
            gh = GROUP_COLORS[group_name]
            gr_rgb = (int(gh[1:3], 16), int(gh[3:5], 16), int(gh[5:7], 16))
            draw.text((4, y_offset + crop_h // 2 - 5), group_name, fill=gr_rgb)

            for col_idx, slot_idx in enumerate(slots):
                if perm is None or slot_idx >= len(perm):
                    continue
                ann_idx = perm[slot_idx]
                if ann_idx < 0 or ann_idx >= len(anns):
                    continue

                ann = anns[ann_idx]
                crop = _crop_chromosome(image, ann["bbox"])
                crop_resized = crop.resize((crop_w, crop_h), Image.LANCZOS)

                x = 45 + col_idx * cell_w
                y = y_offset
                canvas.paste(crop_resized, (x, y))

                # 边框颜色：正确=绿，错误=红（仅预测侧显示）
                if ax_idx == 0 and gt_perm is not None:
                    is_correct = (slot_idx < len(perm) and slot_idx < len(gt_perm)
                                  and perm[slot_idx] == gt_perm[slot_idx])
                    border_color = (0, 180, 0) if is_correct else (220, 0, 0)
                else:
                    border_color = gr_rgb

                draw.rectangle([x, y, x + crop_w, y + crop_h], outline=border_color, width=2)

                cls_name = ID_TO_CLASS.get(ann["category_id"], "?")
                draw.text((x + 1, y + crop_h + 2), cls_name, fill=(60, 60, 60))

            y_offset += cell_h

        axes[ax_idx].imshow(np.array(canvas))
        axes[ax_idx].axis("off")
        axes[ax_idx].set_title(label, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def visualize_assignment_matrix(
    logits: "torch.Tensor",
    slot_labels: Optional[List[str]] = None,
    det_labels: Optional[List[str]] = None,
    title: str = "Assignment Matrix (slot × detection)",
    output_path: str = "assignment_matrix.png",
):
    """显示模型输出的 slot×detection 匹配热图

    Args:
        logits: (N, N) 匹配分数矩阵 (单样本)
    """
    import torch
    import torch.nn.functional as F

    if hasattr(logits, "detach"):
        probs = F.softmax(logits.detach().cpu().float(), dim=-1).numpy()
    else:
        probs = np.array(logits)

    N = probs.shape[0]
    slot_labels = slot_labels or [f"{SLOT_ORDER[i]}({i})" for i in range(N)]
    det_labels = det_labels or [str(i) for i in range(N)]

    fig, ax = plt.subplots(figsize=(max(12, N * 0.25), max(10, N * 0.22)))
    im = ax.imshow(probs, cmap="Blues", aspect="auto", vmin=0, vmax=probs.max())

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(det_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(slot_labels, fontsize=7)
    ax.set_xlabel("Detection Index", fontsize=10)
    ax.set_ylabel("Slot (Standard Karyotype Position)", fontsize=10)
    ax.set_title(title, fontsize=12)

    # 彩色分组边框
    slot_boundaries = {}
    prev_group = None
    start = 0
    for i, slot_cls in enumerate(SLOT_ORDER[:N]):
        group = SLOT_TO_GROUP.get(i, "?")
        if group != prev_group:
            if prev_group is not None:
                slot_boundaries[prev_group] = (start, i)
            start = i
            prev_group = group
    if prev_group:
        slot_boundaries[prev_group] = (start, N)

    for group_name, (s, e) in slot_boundaries.items():
        color = GROUP_COLORS.get(group_name, "#888888")
        rect = patches.Rectangle(
            (-0.5, s - 0.5), N, e - s,
            linewidth=1.5, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(-1.5, (s + e) / 2, group_name, color=color,
                fontsize=8, va="center", ha="right", fontweight="bold")

    plt.colorbar(im, ax=ax, label="Probability", fraction=0.02, pad=0.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def visualize_unmask_trajectory(
    image: Image.Image,
    anns: List[dict],
    trajectory: List[List[int]],
    gt_perm: Optional[List[int]] = None,
    output_dir: str = "trajectory_frames",
    crop_h: int = 50,
    crop_w: int = 30,
):
    """将迭代 unmask 过程保存为逐帧图像（可拼成 GIF）

    Args:
        trajectory: list of permutation snapshots at each step
                    masked positions indicated by -1
    """
    os.makedirs(output_dir, exist_ok=True)
    n_steps = len(trajectory)

    group_slot_list = []
    for group_name, classes in DENVER_GROUPS.items():
        slots = []
        for cls in classes:
            slots.extend(CLASS_TO_SLOTS[cls])
        group_slot_list.append((group_name, slots))

    max_cols = max(len(slots) for _, slots in group_slot_list)
    cell_w = crop_w + 6
    cell_h = crop_h + 18
    panel_w = max_cols * cell_w + 50
    panel_h = len(group_slot_list) * cell_h + 40

    frame_paths = []
    for step_idx, perm_snapshot in enumerate(trajectory):
        canvas = Image.new("RGB", (panel_w, panel_h), color=(248, 248, 248))
        draw = ImageDraw.Draw(canvas)

        n_revealed = sum(1 for v in perm_snapshot if v >= 0)
        title_text = f"Step {step_idx + 1}/{n_steps}  |  Revealed: {n_revealed}/{NUM_SLOTS}"
        draw.text((5, 5), title_text, fill=(30, 30, 30))

        y_offset = 25
        for group_name, slots in group_slot_list:
            gh = GROUP_COLORS[group_name]
            gr_rgb = (int(gh[1:3], 16), int(gh[3:5], 16), int(gh[5:7], 16))
            draw.text((4, y_offset + crop_h // 2 - 5), group_name, fill=gr_rgb)

            for col_idx, slot_idx in enumerate(slots):
                x = 45 + col_idx * cell_w
                y = y_offset

                val = perm_snapshot[slot_idx] if slot_idx < len(perm_snapshot) else -1

                if val < 0:
                    # 未揭示: 灰色方块
                    draw.rectangle([x, y, x + crop_w, y + crop_h],
                                   fill=(200, 200, 200), outline=(150, 150, 150), width=1)
                    draw.text((x + crop_w // 2 - 3, y + crop_h // 2 - 5), "?",
                              fill=(120, 120, 120))
                else:
                    ann = anns[val] if val < len(anns) else None
                    if ann:
                        crop = _crop_chromosome(image, ann["bbox"])
                        crop_resized = crop.resize((crop_w, crop_h), Image.LANCZOS)
                        canvas.paste(crop_resized, (x, y))

                        # 边框
                        if gt_perm and slot_idx < len(gt_perm):
                            is_correct = (val == gt_perm[slot_idx])
                            border = (0, 180, 0) if is_correct else (220, 0, 0)
                        else:
                            border = gr_rgb
                        draw.rectangle([x, y, x + crop_w, y + crop_h],
                                       outline=border, width=2)

            y_offset += cell_h

        frame_path = os.path.join(output_dir, f"step_{step_idx:03d}.png")
        canvas.save(frame_path)
        frame_paths.append(frame_path)

    # 尝试合成 GIF
    try:
        frames = [Image.open(p) for p in frame_paths]
        gif_path = os.path.join(output_dir, "trajectory.gif")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=400,
            loop=0,
        )
        print(f"GIF saved: {gif_path}")
    except Exception as e:
        print(f"GIF generation skipped: {e}")

    return frame_paths


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="KaryoFlow Visualization")
    parser.add_argument("--coco-json", type=str, required=True)
    parser.add_argument("--kf-json", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--image-id", type=int, required=True)
    parser.add_argument("--output", type=str, default="karyotype.png",
                        help="输出文件路径 (.png)")
    parser.add_argument("--mode", type=str,
                        choices=["grid", "compare", "matrix"],
                        default="grid",
                        help="可视化模式: grid=核型图, compare=预测vs GT, matrix=分配热图")
    args = parser.parse_args()

    image, anns, permutation = _load_image_and_anns(
        args.coco_json, args.kf_json, args.image_dir, args.image_id
    )
    print(f"Image: {image.size}, Annotations: {len(anns)}, Valid perm: {permutation is not None}")

    if args.mode == "grid":
        if permutation is None:
            print("No valid permutation found for this image.")
            return
        fig = visualize_karyotype_grid(
            image, anns, permutation,
            title=f"Karyotype (image_id={args.image_id})",
        )
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {args.output}")

    elif args.mode == "compare":
        if permutation is None:
            print("No valid permutation found.")
            return
        # 模拟一个 "预测" (随机打乱) 用于演示
        import random
        pred = permutation[:]
        random.shuffle(pred)
        visualize_prediction_vs_gt(
            image, anns, pred, permutation, args.output
        )

    elif args.mode == "matrix":
        # 用随机 logits 演示
        import torch
        N = len(anns)
        logits = torch.randn(N, N)
        slot_labels = [f"{SLOT_ORDER[min(i, NUM_SLOTS-1)]}({i})" for i in range(N)]
        det_labels = [ID_TO_CLASS.get(a["category_id"], "?") + f"({i})"
                      for i, a in enumerate(anns)]
        visualize_assignment_matrix(
            logits, slot_labels, det_labels,
            title=f"Assignment Matrix (image_id={args.image_id})",
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
