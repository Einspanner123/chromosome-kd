"""Figure 7: compact difficult-case comparison with vector annotations."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from figure_style_v2 import (
    C_FOUNDATION, C_LQCR, C_OUTPUT, C_RF, C_TEXT, configure_style,
    save_vector_figure,
)

ROOT = Path(__file__).resolve().parents[6]
ANN_FILE = ROOT / "data/24_chromosomes_object/coco/valid/_annotations.coco.json"
IMAGE_DIR = ROOT / "data/24_chromosomes_object/JEPG"
SOTA_DIR = ROOT / "experiments/analysis/baseline_vs_sota_cache"
BASELINE_DIR = ROOT / "experiments/analysis/baseline_inference_24obj_cache"

CAT = {1:"A1", 2:"A2", 3:"A3", 4:"B4", 5:"B5", 6:"C6", 7:"C7",
       8:"C8", 9:"C9", 10:"C10", 11:"C11", 12:"C12", 13:"D13",
       14:"D14", 15:"D15", 16:"E16", 17:"E17", 18:"E18", 19:"F19",
       20:"F20", 21:"G21", 22:"G22", 23:"X", 24:"Y"}

MODELS = [
    ("gt", "Ground truth", C_FOUNDATION),
    ("ours", "KaryoFlow", C_RF),
    ("diffusiondet", "DiffusionDet", C_OUTPUT),
    ("dino", "DINO R50", C_LQCR),
]

CASES = [
    (1, [24], "rare / small Y"),
    (2, [19, 20, 21, 22], "F--G small target"),
    (4, [13, 14], "D-group local overlap"),
]


def group_predictions(path: Path, dict_key: str | None = None) -> dict[int, list[dict]]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if dict_key is not None:
        data = data[dict_key]
    grouped: dict[int, list[dict]] = {}
    for pred in data:
        grouped.setdefault(int(pred["image_id"]), []).append(pred)
    return grouped


def crop_from_targets(annotations: list[dict], width: int, height: int,
                      target_categories: list[int]) -> tuple[int, int, int, int]:
    targets = [ann for ann in annotations if ann["category_id"] in target_categories]
    if not targets:
        return 0, 0, width, height
    # Anchor each row on one hard instance.  Using the union of all instances
    # from a chromosome group can span the whole metaphase image and makes the
    # comparison unreadable.  The smallest target is the most demanding local
    # case; a 3.4x square neighborhood retains nearby overlap context.
    anchor = min(targets, key=lambda ann: ann["bbox"][2] * ann["bbox"][3])
    x, y, w, h = anchor["bbox"]
    side = max(w, h) * 3.4
    cx, cy = x + w / 2, y + h / 2
    return (max(0, int(cx - side / 2)), max(0, int(cy - side / 2)),
            min(width, int(cx + side / 2)), min(height, int(cy + side / 2)))


def center_inside(bbox: list[float], crop: tuple[int, int, int, int]) -> bool:
    x, y, w, h = bbox; x1, y1, x2, y2 = crop
    return x1 <= x + w / 2 <= x2 and y1 <= y + h / 2 <= y2


def draw_box(ax: plt.Axes, bbox: list[float], crop: tuple[int, int, int, int],
             color: str, label: str) -> None:
    x, y, w, h = bbox; x1, y1, x2, y2 = crop
    sx, sy = 1 / (x2 - x1), 1 / (y2 - y1)
    rx, ry, rw, rh = (x - x1) * sx, (y - y1) * sy, w * sx, h * sy
    ax.add_patch(Rectangle((rx, ry), rw, rh, transform=ax.transAxes,
                           fill=False, edgecolor=color, linewidth=0.9, zorder=5))
    ax.text(rx, max(0.01, ry - 0.01), label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=5.1, color="white", zorder=6,
            bbox={"facecolor": color, "edgecolor": "none", "pad": 0.65})


def main() -> None:
    configure_style()
    with ANN_FILE.open(encoding="utf-8") as stream:
        coco = json.load(stream)
    images = {int(item["id"]): item for item in coco["images"]}
    gt: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        gt.setdefault(int(ann["image_id"]), []).append(ann)
    predictions = {
        "ours": group_predictions(SOTA_DIR / "24obj_SOTA_seed42_preds.json"),
        "diffusiondet": group_predictions(SOTA_DIR / "24obj_DiffusionDet_seed42_preds.json"),
        "dino": group_predictions(BASELINE_DIR / "DINO_R50_seed42_preds.json"),
    }

    fig, axes = plt.subplots(len(CASES), len(MODELS), figsize=(7.2, 4.15),
                             facecolor="white")
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.025, top=0.90,
                        wspace=0.035, hspace=0.07)

    for row, (image_id, target_cats, row_name) in enumerate(CASES):
        info = images[image_id]
        image_path = IMAGE_DIR / info["file_name"]
        image = Image.open(image_path).convert("RGB")
        crop = crop_from_targets(gt.get(image_id, []), image.width, image.height, target_cats)
        cropped = image.crop(crop)

        for col, (model, title, color) in enumerate(MODELS):
            ax = axes[row, col]
            ax.imshow(cropped, extent=(0, 1, 1, 0), interpolation="lanczos")
            ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("white"); spine.set_linewidth(1.0)
            if row == 0:
                ax.set_title(title, fontsize=7.2, weight="bold", color=color, pad=4)
            if col == 0:
                ax.text(-0.10, 0.50, row_name, transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=6.2,
                        weight="bold", color=C_TEXT)

            items = gt.get(image_id, []) if model == "gt" else predictions[model].get(image_id, [])
            # Show only the row's target chromosome group.  This is a local
            # localization comparison, not a full-image false-positive audit;
            # rendering every neighboring category obscures the hard instance.
            visible = [item for item in items
                       if int(item["category_id"]) in target_cats
                       and center_inside(item["bbox"], crop)]
            if model != "gt":
                visible = [item for item in visible if float(item.get("score", 1.0)) >= 0.30]
                visible.sort(key=lambda item: float(item.get("score", 1.0)), reverse=True)
            for item in visible[:8]:
                draw_box(ax, item["bbox"], crop, color, CAT.get(int(item["category_id"]), "?"))

    save_vector_figure(fig, "fig07_qualitative")


if __name__ == "__main__":
    main()
