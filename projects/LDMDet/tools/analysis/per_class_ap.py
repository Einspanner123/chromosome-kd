#!/usr/bin/env python3
"""Per-class AP analysis for comparing coupling strategies.

Computes per-class AP for multiple LDMDet checkpoints and produces
a comparison table showing the delta from the random-coupling baseline.

Usage:
    python projects/LDMDet/tools/analysis/per_class_ap.py

The script is configured via the MODELS dict below. Edit it to add/remove models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.registry import DefaultScope

from mmdet.apis import init_detector
from mmdet.registry import DATASETS, METRICS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# COCO annotation file for the chromosome validation set
ANN_FILE = "data/Chromosome20240904_NoAug_NoResize_coco/valid/_annotations.coco.json"

# Class names (must match the dataset)
CLASS_NAMES = [
    "A1", "A2", "A3", "B4", "B5",
    "C10", "C11", "C12", "C6", "C7", "C8", "C9",
    "D13", "D14", "D15", "E16", "E17", "E18",
    "F19", "F20", "G21", "G22", "X", "Y",
]

# Models to evaluate: (label, config_path, checkpoint_path)
# Paths are relative to the repo root.
REPO = Path(__file__).resolve().parents[3]  # chromo-kd root

MODELS: list[tuple[str, str, str]] = [
    (
        "Random (AdaLN)",
        "work_dirs/ldmdet_flowdet_adaln/ldmdet_flowdet_adaln.py",
        "work_dirs/ldmdet_flowdet_adaln/best_coco_bbox_mAP_epoch_79.pth",
    ),
    (
        "Hard OT",
        "work_dirs/ldmdet_flowdet_adaln_ot/ldmdet_flowdet_adaln_ot.py",
        "work_dirs/ldmdet_flowdet_adaln_ot/best_coco_bbox_mAP_epoch_52.pth",
    ),
    (
        "Sinkhorn argmax eps=5",
        "work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_eps5/ldmdet_flowdet_adaln_ot_sinkhorn_eps5.py",
        "work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_eps5/best_coco_bbox_mAP_epoch_67.pth",
    ),
    (
        "Sinkhorn sample eps=1",
        "work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps1/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps1.py",
        "work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps1/best_coco_bbox_mAP_epoch_70.pth",
    ),
    (
        "Sinkhorn sample eps=5",
        "work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py",
        "work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5/best_coco_bbox_mAP_epoch_86.pth",
    ),
]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def _load_model(config_path: Path, checkpoint_path: Path, device: str):
    cfg = Config.fromfile(str(config_path))
    # DefaultScope is needed so that MMEngine registries resolve correctly
    DefaultScope.get_default_scope()
    model = init_detector(cfg, str(checkpoint_path), device=device)
    return model, cfg


def _evaluate_model(
    model,
    cfg,
    ann_file: str,
    device: str,
) -> dict:
    """Run inference on the val dataset and compute classwise COCO metrics."""
    # Temporarily override evaluator to enable classwise
    cfg_copy = Config.fromfile(str(cfg))
    test_dataset_cfg = cfg_copy.test_dataloader.dataset
    val_dataset_cfg = cfg_copy.val_dataloader.dataset
    # Prefer val dataset, fall back to test
    dataset_cfg = val_dataset_cfg if hasattr(cfg_copy, "val_dataloader") else test_dataset_cfg

    dataset = DATASETS.build(dataset_cfg)

    results: list = []
    for i in range(len(dataset)):
        data = dataset[i]
        data = model.data_preprocessor(data, False)
        batch_inputs, batch_data_samples = data["inputs"], data["data_samples"]
        if isinstance(batch_inputs, torch.Tensor):
            batch_inputs = batch_inputs.unsqueeze(0).to(device)
        batch_data_samples = [batch_data_samples]
        with torch.no_grad():
            out = model.test_step(batch_inputs)
        results.extend(out)

    # Build evaluator with classwise enabled
    evaluator = METRICS.build(
        dict(
            type="CocoMetric",
            ann_file=ann_file,
            metric="bbox",
            classwise=True,
            format_only=False,
        )
    )
    evaluator.dataset_meta = dataset.metainfo

    for data_sample in results:
        evaluator.process({}, [data_sample])

    metrics = evaluator.evaluate(len(results))
    return metrics


def _extract_per_class(metrics: dict) -> dict[str, float]:
    """Pull per-class AP keys from the flat metric dict."""
    per_class: dict[str, float] = {}
    for cls_name in CLASS_NAMES:
        key = f"coco/{cls_name}_precision"
        if key in metrics:
            per_class[cls_name] = round(float(metrics[key]), 4)
        else:
            per_class[cls_name] = float("nan")
    return per_class


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 90)
    print("Per-Class AP Analysis: Coupling Strategy Comparison")
    print("=" * 90)
    print(f"Device: {DEVICE}")
    print(f"Annotation file: {ANN_FILE}")
    print(f"Models to evaluate: {len(MODELS)}")
    print()

    all_results: dict[str, dict] = {}
    group_labels = [
        "A (1-3)", "A (1-3)", "A (1-3)",
        "B (4-5)", "B (4-5)",
        "C (6-12)", "C (6-12)", "C (6-12)", "C (6-12)", "C (6-12)", "C (6-12)", "C (6-12)",
        "D (13-15)", "D (13-15)", "D (13-15)",
        "E (16-18)", "E (16-18)", "E (16-18)",
        "F (19-20)", "F (19-20)",
        "G (21-22)", "G (21-22)",
        "X", "Y",
    ]

    for label, config_rel, ckpt_rel in MODELS:
        config_path = _resolve(config_rel)
        ckpt_path = _resolve(ckpt_rel)

        print(f"--- {label} ---")
        print(f"  Config: {config_path}")
        print(f"  Checkpoint: {ckpt_path}")

        if not config_path.exists():
            print(f"  [SKIP] Config not found: {config_path}")
            continue
        if not ckpt_path.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        model, cfg = _load_model(config_path, ckpt_path, DEVICE)
        metrics = _evaluate_model(model, config_path, ANN_FILE, DEVICE)
        per_class = _extract_per_class(metrics)
        all_results[label] = {
            "per_class": per_class,
            "mAP": round(float(metrics.get("coco/bbox_mAP", float("nan"))), 4),
            "AP50": round(float(metrics.get("coco/bbox_mAP_50", float("nan"))), 4),
            "AP75": round(float(metrics.get("coco/bbox_mAP_75", float("nan"))), 4),
        }
        print(f"  mAP={all_results[label]['mAP']}  AP50={all_results[label]['AP50']}  AP75={all_results[label]['AP75']}")
        print()

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Identify baseline
    baseline_label = MODELS[0][0]
    if baseline_label not in all_results:
        print("ERROR: Baseline model not evaluated.")
        sys.exit(1)
    baseline_pc = all_results[baseline_label]["per_class"]

    # ---- Comparison Table ----
    print()
    print("=" * 120)
    print("Per-Class AP Comparison (delta from baseline in parentheses)")
    print("=" * 120)

    # Header
    header = f"{'Class':>5} {'Group':<14}"
    for label in [m[0] for m in MODELS]:
        if label in all_results:
            header += f" {label:>26}"
    print(header)
    print("-" * 120)

    for i, cls_name in enumerate(CLASS_NAMES):
        row = f"{cls_name:>5} {group_labels[i]:<14}"
        for label in [m[0] for m in MODELS]:
            if label not in all_results:
                row += f" {'N/A':>26}"
                continue
            val = all_results[label]["per_class"].get(cls_name, float("nan"))
            delta = val - baseline_pc.get(cls_name, 0)
            row += f" {val:>7.4f} ({delta:+.4f})"
        print(row)

    # Summary row
    print("-" * 120)
    summary = f"{'mAP':>5} {'':<14}"
    for label in [m[0] for m in MODELS]:
        if label not in all_results:
            summary += f" {'N/A':>26}"
            continue
        val = all_results[label]["mAP"]
        delta = val - all_results[baseline_label]["mAP"]
        summary += f" {val:>7.4f} ({delta:+.4f})"
    print(summary)
    print()

    # ---- Group Analysis ----
    print("=" * 120)
    print("Per-Group Mean AP")
    print("=" * 120)
    groups = sorted(set(group_labels))
    for grp in groups:
        cls_indices = [i for i, g in enumerate(group_labels) if g == grp]
        row = f"  {grp:<14}"
        for label in [m[0] for m in MODELS]:
            if label not in all_results:
                row += f" {'N/A':>26}"
                continue
            vals = [all_results[label]["per_class"][CLASS_NAMES[i]] for i in cls_indices]
            mean_val = np.mean(vals)
            base_vals = [baseline_pc[CLASS_NAMES[i]] for i in cls_indices]
            delta = mean_val - np.mean(base_vals)
            row += f" {mean_val:>7.4f} ({delta:+.4f})"
        print(row)
    print()

    # ---- Worst-hit Classes Under Hard OT ----
    print("=" * 120)
    print("Classes Most Affected by Hard OT (vs Random baseline)")
    print("=" * 120)
    hard_ot_label = None
    for m in MODELS:
        if "Hard OT" in m[0] and m[0] in all_results:
            hard_ot_label = m[0]
            break
    if hard_ot_label:
        deltas = []
        for cls_name in CLASS_NAMES:
            delta = all_results[hard_ot_label]["per_class"].get(cls_name, 0) - baseline_pc.get(cls_name, 0)
            deltas.append((cls_name, delta, group_labels[CLASS_NAMES.index(cls_name)]))
        deltas.sort(key=lambda x: x[1])  # most negative first
        print(f"{'Class':>5} {'Group':<14} {'Delta':>8} {'Baseline AP':>12} {'Hard OT AP':>12}")
        print("-" * 60)
        for cls_name, delta, grp in deltas:
            base_ap = baseline_pc.get(cls_name, float("nan"))
            hard_ap = all_results[hard_ot_label]["per_class"].get(cls_name, float("nan"))
            marker = " *** LARGEST DROP" if delta < -0.03 else ""
            print(f"{cls_name:>5} {grp:<14} {delta:>+8.4f} {base_ap:>12.4f} {hard_ap:>12.4f}{marker}")

    # ---- Save JSON ----
    output_path = REPO / "projects/LDMDet/per_class_ap_results.json"
    output_data = {
        "baseline": baseline_label,
        "class_names": CLASS_NAMES,
        "group_labels": group_labels,
        "results": {
            label: {
                "mAP": data["mAP"],
                "AP50": data["AP50"],
                "AP75": data["AP75"],
                "per_class": data["per_class"],
            }
            for label, data in all_results.items()
        },
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
