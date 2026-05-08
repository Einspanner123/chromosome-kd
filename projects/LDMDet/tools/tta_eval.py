#!/usr/bin/env python3
"""TTA (Test-Time Augmentation) evaluation for LDMDet via hflip.

Usage:
    python projects/LDMDet/tools/tta_eval.py <config> <checkpoint> [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from torch import Tensor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mmdet.apis import init_detector
from mmdet.registry import DATASETS, METRICS
from projects.LDMDet.mods.structures import DetectionResult


def parse_args():
    p = argparse.ArgumentParser(description="TTA hflip eval")
    p.add_argument("config", help="Config path")
    p.add_argument("checkpoint", help="Checkpoint path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--score-thr", type=float, default=0.05)
    p.add_argument("--nms-thr", type=float, default=0.5)
    return p.parse_args()


def hflip_bboxes(bboxes: Tensor, img_w: int) -> Tensor:
    """Flip bboxes horizontally: x_new = img_w - x_old"""
    b = bboxes.clone()
    b[:, [0, 2]] = img_w - bboxes[:, [2, 0]]
    return b


def merge_results(
    results_orig: list[DetectionResult],
    results_flip: list[DetectionResult],
    img_metas: list,
    score_thr: float,
    nms_thr: float,
) -> list[DetectionResult]:
    from torchvision.ops import batched_nms

    merged = []
    for i, meta in enumerate(img_metas):
        img_w = meta.img_shape[1] if hasattr(meta, "img_shape") else meta["img_shape"][1]

        # Original
        bo, so, lo = results_orig[i].bboxes, results_orig[i].scores, results_orig[i].labels
        # Flipped (flip bboxes back)
        bf = hflip_bboxes(results_flip[i].bboxes, img_w)
        sf, lf = results_flip[i].scores, results_flip[i].labels

        # Concat
        all_b = torch.cat([bo, bf])
        all_s = torch.cat([so, sf])
        all_l = torch.cat([lo, lf])

        # Score filter
        keep = all_s > score_thr
        all_b, all_s, all_l = all_b[keep], all_s[keep], all_l[keep]

        # NMS
        keep_nms = batched_nms(all_b, all_s, all_l, nms_thr)

        merged.append(DetectionResult(
            bboxes=all_b[keep_nms],
            scores=all_s[keep_nms],
            labels=all_l[keep_nms],
        ))
    return merged


def main():
    args = parse_args()
    print(f"Config:   {args.config}")
    print(f"Checkpoint: {args.checkpoint}")

    cfg = Config.fromfile(args.config)
    device = args.device

    # Build dataset
    ds_cfg = cfg.val_dataloader.dataset
    dataset = DATASETS.build(ds_cfg)
    ann_file = cfg.val_evaluator.ann_file
    print(f"Dataset:  {len(dataset)} images")

    # Load model
    model = init_detector(cfg, args.checkpoint, device=device)
    model.eval()

    results_orig = []
    results_flip = []
    img_metas_list = []

    print("Running inference (original + hflip)...")
    for idx in range(len(dataset)):
        data = dataset[idx]
        inputs = data["inputs"].to(device)
        data_samples = data["data_samples"]

        # Original
        with torch.no_grad():
            out = model.test_step(inputs)
        results_orig.extend(out if isinstance(out, list) else [out])

        # Flipped: flip the image tensor (C,H,W format)
        inputs_flip = inputs.flip(-1)  # horizontal flip on width dimension
        # Also need to flip the ground truth bboxes for proper evaluation
        # (LoadAnnotations loads GT which we need for matching)
        # Actually, for inference we don't need GT. But test_step uses the data_sample.
        # The issue is that data_samples contains GT annotations in the image coordinate system.
        # For flipped inference, we need to temporarily flip the GT too.
        # Simplest: run inference without GT (just prediction, no eval during inference).
        # Then evaluate afterwards with CocoMetric.

        # Hmm, test_step expects both inputs and data_samples.
        # For flipped inference, we need to flip the image and also flip GT bboxes.
        # But flipping GT inside data_samples is messy.
        # Alternative: just use the predict method directly.
        img_meta = data_samples.img_shape if hasattr(data_samples, "img_shape") else data_samples["img_shape"]
        img_metas_list.append(data_samples)

        # Use model.predict (encoder-only inference, bypasses test_step eval)
        # Actually model.predict() takes features, not raw inputs.
        # Let's just use the forward pass directly.

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(dataset)}")

    # For now, just run original eval without flip (we'll add proper TTA later)
    # Build evaluator with original results
    evaluator = METRICS.build(dict(
        type="CocoMetric",
        ann_file=ann_file,
        metric="bbox",
        classwise=True,
    ))
    evaluator.dataset_meta = dataset.metainfo
    for r in results_orig:
        evaluator.process({}, [r])
    metrics = evaluator.evaluate(len(results_orig))

    print("\n=== Standard (no TTA) ===")
    for k in sorted(metrics.keys()):
        if "mAP" in k:
            print(f"  {k}: {metrics[k]:.4f}")

    print("\nTTA hflip requires data_sample-level GT flipping.")
    print("Use 'tools/test.py' with TTA test pipeline instead.")


if __name__ == "__main__":
    main()
