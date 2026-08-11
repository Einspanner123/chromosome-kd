"""Figure 1: image evidence, box-state transport, and detector architecture.

Microscopy panels are raster evidence. Feature maps, boxes, coordinate traces,
architecture, labels, and connectors remain vector-editable. Intermediate box
states visualize the prescribed linear rectified-flow path; they are not
claimed as exported hidden activations.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from PIL import Image

from figure_style_v2 import (
    C_FOUNDATION, C_LIGHT, C_LINE, C_LQCR, C_LQCR_LIGHT, C_MUTED,
    C_OUTPUT, C_RF, C_RF_LIGHT, C_TEXT, configure_style, save_vector_figure,
)


if "KARYOFLOW_ROOT" in os.environ:
    ROOT = Path(os.environ["KARYOFLOW_ROOT"])
else:
    ROOT = Path(__file__).resolve().parents[6]
ANN_FILE = ROOT / "data/24_chromosomes_object/coco/valid/_annotations.coco.json"
IMAGE_DIR = ROOT / "data/24_chromosomes_object/JEPG"
PRED_FILE = Path(os.environ.get(
    "KARYOFLOW_PRED_FILE",
    ROOT / "experiments/analysis/baseline_vs_sota_cache"
    / "24obj_SOTA_seed42_preds.json",
))
IMAGE_ID = 2
CAT = {1:"A1", 2:"A2", 3:"A3", 4:"B4", 5:"B5", 6:"C6", 7:"C7",
       8:"C8", 9:"C9", 10:"C10", 11:"C11", 12:"C12", 13:"D13",
       14:"D14", 15:"D15", 16:"E16", 17:"E17", 18:"E18", 19:"F19",
       20:"F20", 21:"G21", 22:"G22", 23:"X", 24:"Y"}


def load_scene() -> tuple[Image.Image, list[dict], list[dict]]:
    with ANN_FILE.open(encoding="utf-8") as stream:
        coco = json.load(stream)
    info = next(item for item in coco["images"] if int(item["id"]) == IMAGE_ID)
    gt = [item for item in coco["annotations"] if int(item["image_id"]) == IMAGE_ID]
    with PRED_FILE.open(encoding="utf-8") as stream:
        preds = [item for item in json.load(stream)
                 if int(item["image_id"]) == IMAGE_ID and float(item["score"]) >= 0.30]
    preds.sort(key=lambda item: float(item["score"]), reverse=True)
    return Image.open(IMAGE_DIR / info["file_name"]).convert("RGB"), gt, preds


def dense_crop(annotations: list[dict], width: int, height: int) -> tuple[int, int, int, int]:
    side = int(0.36 * min(width, height))
    centers = np.array([[a["bbox"][0] + a["bbox"][2] / 2,
                         a["bbox"][1] + a["bbox"][3] / 2] for a in annotations])
    best_center, best_count = centers[0], -1
    for center in centers:
        count = np.sum((np.abs(centers[:, 0] - center[0]) <= side / 2)
                       & (np.abs(centers[:, 1] - center[1]) <= side / 2))
        if count > best_count:
            best_center, best_count = center, int(count)
    x1 = int(np.clip(best_center[0] - side / 2, 0, width - side))
    y1 = int(np.clip(best_center[1] - side / 2, 0, height - side))
    return x1, y1, x1 + side, y1 + side


def center_inside(bbox: list[float], crop: tuple[int, int, int, int]) -> bool:
    x, y, w, h = bbox
    x1, y1, x2, y2 = crop
    return x1 <= x + w / 2 <= x2 and y1 <= y + h / 2 <= y2


def image_axis(fig: plt.Figure, rect: tuple[float, float, float, float],
               image: Image.Image) -> plt.Axes:
    ax = fig.add_axes(rect)
    ax.imshow(image, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#D0D5DD"); spine.set_linewidth(0.65)
    return ax


def draw_bbox(ax: plt.Axes, bbox: list[float], color: str, *,
              origin: tuple[float, float] = (0, 0), linewidth: float = 0.75,
              linestyle: str = "-", label: str | None = None) -> None:
    x, y, w, h = bbox
    x -= origin[0]; y -= origin[1]
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color,
                           linewidth=linewidth, linestyle=linestyle, zorder=5))
    if label:
        ax.text(x, y, label, ha="left", va="bottom", fontsize=4.4,
                color="white", zorder=6,
                bbox={"facecolor": color, "edgecolor": "none", "pad": 0.45})


def noisy_bbox(final: np.ndarray, index: int, crop: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = crop
    width, height = x2 - x1, y2 - y1
    rng = np.random.default_rng(9107 + index)
    x, y, w, h = final
    cx = x + w / 2 + rng.uniform(-0.31, 0.31) * width
    cy = y + h / 2 + rng.uniform(-0.31, 0.31) * height
    nw = np.clip(w * rng.uniform(0.55, 1.9), 0.035 * width, 0.30 * width)
    nh = np.clip(h * rng.uniform(0.55, 1.9), 0.035 * height, 0.30 * height)
    return np.array([np.clip(cx - nw / 2, x1, x2 - nw),
                     np.clip(cy - nh / 2, y1, y2 - nh), nw, nh])


def panel_label(fig: plt.Figure, x: float, y: float, label: str) -> None:
    fig.text(x, y, f"({label})", ha="left", va="top", fontsize=8.6,
             weight="bold", color=C_TEXT)


def rounded(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float],
            text: str, *, fc: str = "white", ec: str = C_LINE,
            color: str = C_TEXT, size: float = 5.6, weight: str = "normal") -> None:
    x, y = xy; w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.012,rounding_size=0.018",
                 facecolor=fc, edgecolor=ec, linewidth=0.75))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color=color, weight=weight)


def feature_pyramid(fig: plt.Figure, rect: tuple[float, float, float, float]) -> None:
    """Abstract multi-scale maps, intentionally not resized copies of the image."""
    ax = fig.add_axes(rect); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.98, "ResNet-50", ha="center", va="top", fontsize=5.8,
            color=C_FOUNDATION, weight="bold")
    levels = [
        (0.08, 0.10, 0.80, 0.23, "P2", "#E5F3FA"),
        (0.17, 0.34, 0.65, 0.21, "P3", "#D3EAF6"),
        (0.27, 0.56, 0.49, 0.18, "P4", "#B9DDEE"),
        (0.37, 0.75, 0.34, 0.15, "P5", "#98CBE3"),
    ]
    for x, y, w, h, label, face in levels:
        skew = 0.055
        poly = Polygon([[x, y], [x+w, y], [x+w+skew, y+h], [x+skew, y+h]],
                       closed=True, facecolor=face, edgecolor=C_RF, linewidth=0.7)
        ax.add_patch(poly)
        for frac in (0.33, 0.66):
            ax.plot([x+frac*w, x+frac*w+skew], [y, y+h], color="white", lw=0.35)
        ax.plot([x+skew/2, x+w+skew/2], [y+h/2, y+h/2], color="white", lw=0.35)
        ax.text(x+w+skew+0.025, y+h/2, label, va="center", fontsize=5.2,
                color=C_RF, weight="bold")
    ax.text(0.50, 0.01, "FPN: multi-scale feature maps", ha="center",
            va="bottom", fontsize=5.2, color=C_MUTED)


def coordinate_path(fig: plt.Figure, rect: tuple[float, float, float, float],
                    noise: np.ndarray, final: np.ndarray,
                    crop: tuple[int, int, int, int]) -> None:
    ax = fig.add_axes(rect); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    x1, y1, x2, y2 = crop; cw, ch = x2-x1, y2-y1
    times = [1.0, 0.67, 0.33, 0.0]
    colors = [C_LINE, "#7EA6C2", "#3B88B8", C_RF]
    tile_x = [0.00, 0.255, 0.51, 0.765]
    for idx, (left, t, color) in enumerate(zip(tile_x, times, colors)):
        state = t * noise + (1-t) * final
        nx, ny, nw, nh = ((state[0]-x1)/cw, (state[1]-y1)/ch,
                          state[2]/cw, state[3]/ch)
        # The physical axes aspect makes 0.19 x 0.32 approximately square.
        tile_y, tile_w, tile_h = 0.44, 0.19, 0.32
        ax.add_patch(Rectangle((left, tile_y), tile_w, tile_h,
                               facecolor="white", edgecolor=C_LINE, lw=0.55))
        for frac in (1/3, 2/3):
            ax.plot([left+frac*tile_w, left+frac*tile_w], [tile_y, tile_y+tile_h],
                    color=C_LIGHT, lw=0.45)
            ax.plot([left, left+tile_w], [tile_y+frac*tile_h, tile_y+frac*tile_h],
                    color=C_LIGHT, lw=0.45)
        ax.add_patch(Rectangle((left+nx*tile_w, tile_y+(1-ny-nh)*tile_h),
                               nw*tile_w, nh*tile_h, fill=False,
                               edgecolor=color, lw=0.82,
                               linestyle=(0, (2, 1.5)) if t == 1 else "-"))
        ax.plot(left+(nx+nw/2)*tile_w, tile_y+(1-ny-nh/2)*tile_h,
                marker="o", ms=1.65, color=color)
        ax.text(left+tile_w/2, 0.86, f"t={t:.2f}", ha="center", va="center",
                fontsize=5.4, color=color, weight="bold" if t == 0 else "normal")
        ax.text(left+tile_w/2, 0.23,
                f"c=({nx+nw/2:.2f},{ny+nh/2:.2f})\ns=({nw:.2f},{nh:.2f})",
                ha="center", va="center", fontsize=4.5, color=C_MUTED)
        if idx < 3:
            ax.annotate("", xy=(left+0.245, 0.60), xytext=(left+0.202, 0.60),
                        arrowprops=dict(arrowstyle="-|>", color=C_FOUNDATION,
                                        lw=0.65, mutation_scale=6))
    ax.text(0.0, 0.02, "noise source", fontsize=4.8, color=C_MUTED)
    ax.text(1.0, 0.02, "predicted target box", ha="right", fontsize=4.8,
            color=C_RF)


def architecture_panel(fig: plt.Figure, rect: tuple[float, float, float, float]) -> None:
    ax = fig.add_axes(rect); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    rounded(ax, (0.025, 0.49), (0.085, 0.17), "FPN\nP2-P5", fc=C_RF_LIGHT,
            ec=C_RF, color=C_RF, size=5.5, weight="bold")
    rounded(ax, (0.025, 0.17), (0.085, 0.16), "boxes\n$x_t$", fc="white",
            ec=C_FOUNDATION, color=C_FOUNDATION, size=5.5, weight="bold")
    rounded(ax, (0.155, 0.36), (0.095, 0.17), "RoIAlign\n$7\\times7$", fc=C_LIGHT,
            ec=C_LINE, size=5.5, weight="bold")
    ax.annotate("", xy=(0.142, 0.455), xytext=(0.122, 0.575),
                arrowprops=dict(arrowstyle="-|>", color=C_FOUNDATION, lw=0.65,
                                mutation_scale=6))
    ax.annotate("", xy=(0.142, 0.395), xytext=(0.122, 0.25),
                arrowprops=dict(arrowstyle="-|>", color=C_FOUNDATION, lw=0.65,
                                mutation_scale=6))

    # One actual cascade head: self-attention, DynamicConv and FFN.
    ax.add_patch(FancyBboxPatch((0.31, 0.27), 0.39, 0.45,
                 boxstyle="round,pad=0.012,rounding_size=0.02",
                 facecolor="#F8FAFC", edgecolor=C_RF, linewidth=0.9))
    ax.text(0.327, 0.675, "single cascade head  (repeated $H=6$)",
            fontsize=5.6, color=C_RF, weight="bold", va="center")
    stages = [
        (0.33, "multi-head\nself-attention"),
        (0.448, "DynamicConv\nproposal-RoI interaction"),
        (0.585, "feed-forward\nnetwork"),
    ]
    widths = [0.088, 0.105, 0.080]
    for (x, label), width in zip(stages, widths):
        rounded(ax, (x, 0.39), (width, 0.16), label, fc="white", ec=C_LINE,
                size=4.45)
    for start, end in ((0.430, 0.436), (0.565, 0.573)):
        ax.annotate("", xy=(end, 0.47), xytext=(start, 0.47),
                    arrowprops=dict(arrowstyle="-|>", color=C_FOUNDATION,
                                    lw=0.6, mutation_scale=5,
                                    shrinkA=0, shrinkB=0))

    rounded(ax, (0.425, 0.035), (0.16, 0.115), "time embedding $t$\nAdaLN-Zero",
            fc="#F3EEF8", ec=C_OUTPUT, color=C_OUTPUT, size=4.9, weight="bold")
    for target in (0.375, 0.625):
        ax.annotate("", xy=(target, 0.378), xytext=(0.505, 0.162),
                    arrowprops=dict(arrowstyle="-|>", color=C_OUTPUT,
                                    lw=0.65, mutation_scale=5,
                                    shrinkA=0, shrinkB=0))

    ax.annotate("", xy=(0.298, 0.47), xytext=(0.262, 0.445),
                arrowprops=dict(arrowstyle="-|>", color=C_FOUNDATION, lw=0.7,
                                mutation_scale=6, shrinkA=0, shrinkB=0))
    rounded(ax, (0.755, 0.55), (0.125, 0.15), "class logits\n$p$", fc="white",
            ec=C_FOUNDATION, color=C_FOUNDATION, size=5.2, weight="bold")
    rounded(ax, (0.755, 0.31), (0.125, 0.15), "target box\n$\\hat{x}_0$", fc=C_RF_LIGHT,
            ec=C_RF, color=C_RF, size=5.2, weight="bold")
    rounded(ax, (0.755, 0.075), (0.125, 0.14), "quality $q$\nfinal head only",
            fc=C_LQCR_LIGHT, ec=C_LQCR, color=C_LQCR, size=4.8, weight="bold")
    # All prediction branches share the same straight-arrow grammar.
    for target_y, color in ((0.625, C_FOUNDATION), (0.385, C_RF),
                            (0.145, C_LQCR)):
        ax.annotate("", xy=(0.742, target_y), xytext=(0.712, 0.47),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.65,
                                    mutation_scale=5,
                                    connectionstyle="arc3,rad=0",
                                    shrinkA=0, shrinkB=0))
    ax.text(0.942, 0.57, "solver update", ha="center", fontsize=4.8,
            color=C_RF, weight="bold")
    ax.text(0.942, 0.47, r"$x_t \,\leftarrow\, \hat{x}_0$ history",
            ha="center", fontsize=5.0, color=C_RF)
    ax.plot([0.895, 0.99], [0.34, 0.34], color=C_LIGHT, lw=0.6)
    ax.text(0.942, 0.25, "final output", ha="center", fontsize=4.8,
            color=C_LQCR, weight="bold")
    ax.text(0.942, 0.15, r"rank by $p q^2$", ha="center", fontsize=5.7,
            color=C_LQCR, weight="bold")


def main() -> None:
    configure_style()
    image, gt, preds = load_scene()
    crop = dense_crop(gt, image.width, image.height)
    cropped = image.crop(crop)
    local_preds = [item for item in preds if center_inside(item["bbox"], crop)][:7]
    if len(local_preds) < 4:
        raise RuntimeError("Selected scene does not contain enough local predictions")

    fig = plt.figure(figsize=(7.2, 3.62), facecolor="white")
    panel_label(fig, 0.022, 0.958, "a")
    panel_label(fig, 0.302, 0.958, "b")
    panel_label(fig, 0.838, 0.958, "c")
    panel_label(fig, 0.022, 0.425, "d")
    panel_label(fig, 0.325, 0.425, "e")

    input_ax = image_axis(fig, (0.022, 0.49, 0.158, 0.405), image)
    input_ax.add_patch(Rectangle((crop[0], crop[1]), crop[2]-crop[0], crop[3]-crop[1],
                                 fill=False, edgecolor=C_LQCR, linewidth=1.0))
    input_ax.text(0.03, 0.04, "input image", transform=input_ax.transAxes,
                  color="white", fontsize=5.6, weight="bold",
                  bbox={"facecolor":"#202124CC", "edgecolor":"none", "pad":1.1})

    feature_pyramid(fig, (0.194, 0.51, 0.085, 0.36))

    output_ax = image_axis(fig, (0.838, 0.49, 0.142, 0.405), image)
    for rank, item in enumerate(preds[:42]):
        draw_bbox(output_ax, item["bbox"], C_RF, linewidth=0.52,
                  label=CAT.get(int(item["category_id"])) if rank < 8 else None)
    output_ax.text(0.03, 0.04, "ranked output", transform=output_ax.transAxes,
                   color="white", fontsize=5.6, weight="bold",
                   bbox={"facecolor":"#006DA8DD", "edgecolor":"none", "pad":1.1})

    state_x = [0.302, 0.432, 0.562, 0.692]
    times = [1.0, 0.67, 0.33, 0.0]
    labels = ["noise", "NFE 1", "NFE 2", "NFE 4"]
    final_boxes = [np.array(item["bbox"], dtype=float) for item in local_preds]
    noise_boxes = [noisy_bbox(box, idx, crop) for idx, box in enumerate(final_boxes)]
    colors = [C_LINE, "#7EA6C2", "#3B88B8", C_RF]
    for x, time_value, state_label, color in zip(state_x, times, labels, colors):
        ax = image_axis(fig, (x, 0.565, 0.104, 0.30), cropped)
        for noise, final in zip(noise_boxes, final_boxes):
            state = time_value * noise + (1-time_value) * final
            draw_bbox(ax, state.tolist(), color, origin=(crop[0], crop[1]),
                      linewidth=0.75, linestyle=(0, (2, 1.5)) if time_value == 1 else "-")
        ax.set_title(rf"$t={time_value:.2f}$", fontsize=6.2, color=color, pad=1.5)
        fig.text(x+0.052, 0.545, state_label, ha="center", va="top",
                 fontsize=5.3, color=color, weight="bold" if time_value == 0 else "normal")

    crop_center = np.array([(crop[0] + crop[2]) / 2,
                            (crop[1] + crop[3]) / 2])
    final_centers = np.array([[box[0] + box[2] / 2,
                               box[1] + box[3] / 2] for box in final_boxes])
    noise_centers = np.array([[box[0] + box[2] / 2,
                               box[1] + box[3] / 2] for box in noise_boxes])
    # Keep the explanatory trajectory spatially central, then prefer the
    # candidate with the clearest center displacement among the four nearest.
    center_distance = np.linalg.norm(final_centers - crop_center, axis=1)
    central_candidates = np.argsort(center_distance)[:4]
    displacement = np.linalg.norm(final_centers - noise_centers, axis=1)
    trajectory_index = int(central_candidates[
        np.argmax(displacement[central_candidates])])
    coordinate_path(fig, (0.022, 0.075, 0.275, 0.32),
                    noise_boxes[trajectory_index], final_boxes[trajectory_index], crop)
    architecture_panel(fig, (0.325, 0.065, 0.655, 0.335))
    save_vector_figure(fig, "fig01_system_overview")


if __name__ == "__main__":
    main()
