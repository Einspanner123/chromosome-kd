"""Figure 1: effect-first KaryoFlow overview with real input and output.

Raster microscopy is embedded only where visual evidence is required. Text,
trajectory curves, proposal boxes, labels, and layout remain vector-editable.
Intermediate RF states are a visualization of the prescribed linear path, not
an export of hidden inference states; the final detections are real predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from figure_style_v2 import (
    C_FOUNDATION,
    C_LIGHT,
    C_LINE,
    C_LQCR,
    C_MUTED,
    C_OUTPUT,
    C_RF,
    C_TEXT,
    configure_style,
    save_vector_figure,
)


ROOT = Path(__file__).resolve().parents[6]
ANN_FILE = ROOT / "data/24_chromosomes_object/coco/valid/_annotations.coco.json"
IMAGE_DIR = ROOT / "data/24_chromosomes_object/JEPG"
PRED_FILE = (
    ROOT / "experiments/analysis/baseline_vs_sota_cache"
    / "24obj_SOTA_seed42_preds.json"
)
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
        ax.text(x, y, label, ha="left", va="bottom", fontsize=4.6,
                color="white", zorder=6,
                bbox={"facecolor": color, "edgecolor": "none", "pad": 0.5})


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


def panel_label(fig: plt.Figure, x: float, text: str) -> None:
    fig.text(x, 0.945, text, ha="left", va="top", fontsize=8.2,
             weight="bold", color=C_TEXT)


def main() -> None:
    configure_style()
    image, gt, preds = load_scene()
    crop = dense_crop(gt, image.width, image.height)
    cropped = image.crop(crop)
    local_preds = [item for item in preds if center_inside(item["bbox"], crop)][:7]
    if len(local_preds) < 4:
        raise RuntimeError("Selected scene does not contain enough local predictions")

    fig = plt.figure(figsize=(7.2, 3.55), facecolor="white")
    panel_label(fig, 0.022, "(a)  Metaphase input")
    panel_label(fig, 0.285, "(b)  KaryoFlow: box flow inside the detector")
    panel_label(fig, 0.825, "(c)  Detected chromosomes")

    # Large, real input and output anchor the visual narrative.
    input_ax = image_axis(fig, (0.022, 0.37, 0.165, 0.52), image)
    input_ax.add_patch(Rectangle((crop[0], crop[1]), crop[2]-crop[0], crop[3]-crop[1],
                                 fill=False, edgecolor=C_LQCR, linewidth=1.0))
    input_ax.text(0.03, 0.04, "input image", transform=input_ax.transAxes,
                  color="white", fontsize=5.8, weight="bold",
                  bbox={"facecolor":"#202124CC", "edgecolor":"none", "pad":1.2})

    output_ax = image_axis(fig, (0.825, 0.37, 0.153, 0.52), image)
    for item in preds[:42]:
        draw_bbox(output_ax, item["bbox"], C_RF, linewidth=0.55,
                  label=CAT.get(int(item["category_id"])) if float(item["score"]) > 0.78 else None)
    output_ax.text(0.03, 0.04, "KaryoFlow output", transform=output_ax.transAxes,
                   color="white", fontsize=5.8, weight="bold",
                   bbox={"facecolor":"#006DA8DD", "edgecolor":"none", "pad":1.2})

    # Actual crop appears as a small feature pyramid rather than abstract FPN boxes.
    fig.text(0.215, 0.835, "image\nfeatures", ha="center", va="center",
             fontsize=6.0, color=C_MUTED)
    pyramid = [(0.198, 0.60, 0.071, 0.20), (0.207, 0.57, 0.061, 0.17),
               (0.216, 0.54, 0.051, 0.14)]
    for level, rect in enumerate(pyramid):
        ax = image_axis(fig, rect, cropped.resize((max(24, cropped.width // (level + 1)),
                                                   max(24, cropped.height // (level + 1)))))
        ax.patch.set_alpha(0.94)
    fig.text(0.233, 0.515, "ResNet-50 + FPN", ha="center", va="top",
             fontsize=5.7, color=C_FOUNDATION)

    # Prescribed RF path shown directly on microscopy, from noisy proposals to
    # actual terminal predictions. This is deliberately not a module flowchart.
    state_x = [0.285, 0.414, 0.543, 0.672]
    times = [1.0, 0.67, 0.33, 0.0]
    labels = ["noise", "step 1", "step 2", "step 4"]
    final_boxes = [np.array(item["bbox"], dtype=float) for item in local_preds]
    noise_boxes = [noisy_bbox(box, idx, crop) for idx, box in enumerate(final_boxes)]
    colors = [C_LINE, "#7EA6C2", "#3B88B8", C_RF]
    for x, time_value, state_label, color in zip(state_x, times, labels, colors):
        ax = image_axis(fig, (x, 0.50, 0.105, 0.36), cropped)
        for idx, (noise, final) in enumerate(zip(noise_boxes, final_boxes)):
            state = time_value * noise + (1 - time_value) * final
            draw_bbox(ax, state.tolist(), color, origin=(crop[0], crop[1]),
                      linewidth=0.75, linestyle=(0, (2, 1.5)) if time_value == 1 else "-")
        ax.set_title(rf"$t={time_value:.2f}$", fontsize=6.5, color=color, pad=2)
        fig.text(x + 0.0525, 0.475, state_label, ha="center", va="top",
                 fontsize=5.6, color=color, weight="bold" if time_value == 0 else "normal")

    # A quiet time axis and direct model annotation replace chains of boxes.
    timeline = fig.add_axes((0.294, 0.425, 0.474, 0.035))
    timeline.plot([0, 1], [0.5, 0.5], color=C_FOUNDATION, linewidth=0.8)
    for position in np.linspace(0, 1, 4):
        timeline.plot([position, position], [0.36, 0.64], color=C_FOUNDATION, linewidth=0.7)
    timeline.set_xlim(-0.02, 1.02); timeline.set_ylim(0, 1); timeline.axis("off")
    fig.text(0.531, 0.405,
             r"shared cascade head $H_{1:6}$ estimates $v_\theta(x_t,t)$; DPM-Solver++ uses 4 NFE",
             ha="center", va="top", fontsize=6.1, color=C_FOUNDATION)

    # Training view: a coordinate trajectory plot is a scientific depiction of
    # flow matching, distinct from the image-space forward pass above.
    flow_ax = fig.add_axes((0.285, 0.075, 0.48, 0.225))
    selected = list(zip(noise_boxes[:6], final_boxes[:6]))
    tau = np.linspace(0, 1, 30)
    for coordinate, alpha in ((0, 0.85), (1, 0.50)):
        for noise, final in selected:
            values = (1 - tau) * noise[coordinate] + tau * final[coordinate]
            values = (values - (crop[coordinate] if coordinate < 2 else 0)) / (
                (crop[2] - crop[0]) if coordinate == 0 else (crop[3] - crop[1]))
            flow_ax.plot(tau, values, color=C_RF if coordinate == 0 else C_OUTPUT,
                         linewidth=0.75, alpha=alpha)
    flow_ax.set_xlim(0, 1); flow_ax.set_ylim(-0.08, 1.08)
    flow_ax.set_xticks([0, 0.5, 1], [r"source $\epsilon$", r"$x_t$", r"target $x_0$"])
    flow_ax.set_yticks([])
    flow_ax.spines[["top", "right", "left"]].set_visible(False)
    flow_ax.spines["bottom"].set_color(C_LINE)
    flow_ax.tick_params(axis="x", labelsize=5.6, colors=C_MUTED, length=2)
    flow_ax.grid(axis="x", color=C_LIGHT, linewidth=0.7)
    flow_ax.text(0.0, 1.09, "Flow-matching supervision in box space",
                 transform=flow_ax.transAxes, ha="left", va="bottom",
                 fontsize=6.8, weight="bold", color=C_TEXT)
    flow_ax.text(1.0, 1.09,
                 r"$x_t=(1-t)x_0+t\epsilon$   $\mathcal{L}_{FM}=\|v_\theta-(\epsilon-x_0)\|_2^2$",
                 transform=flow_ax.transAxes, ha="right", va="bottom",
                 fontsize=6.2, color=C_FOUNDATION)

    # Output ranking is shown as compact evidence encoding, not a process box.
    rank_ax = fig.add_axes((0.825, 0.075, 0.153, 0.225))
    rank_ax.set_xlim(0, 1); rank_ax.set_ylim(0, 1); rank_ax.axis("off")
    rank_ax.text(0, 1.09, "Final decision", transform=rank_ax.transAxes,
                 ha="left", va="bottom", fontsize=6.8, weight="bold")
    rank_ax.text(0, 0.82, r"class $p$", fontsize=5.8, color=C_FOUNDATION)
    rank_ax.plot([0.36, 0.92], [0.84, 0.84], color=C_LIGHT, linewidth=6)
    rank_ax.plot([0.36, 0.78], [0.84, 0.84], color=C_FOUNDATION, linewidth=6)
    rank_ax.text(0, 0.54, r"quality $q$", fontsize=5.8, color=C_LQCR)
    rank_ax.plot([0.36, 0.92], [0.56, 0.56], color=C_LIGHT, linewidth=6)
    rank_ax.plot([0.36, 0.68], [0.56, 0.56], color=C_LQCR, linewidth=6)
    rank_ax.text(0, 0.19, r"rank by $s=pq^2$", fontsize=6.1,
                 color=C_LQCR, weight="bold")
    rank_ax.text(0, 0.02, "boxes and classes unchanged", fontsize=5.1,
                 color=C_MUTED)

    save_vector_figure(fig, "fig01_system_overview")


if __name__ == "__main__":
    main()
