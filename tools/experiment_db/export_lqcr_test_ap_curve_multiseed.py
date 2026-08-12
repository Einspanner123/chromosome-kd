#!/usr/bin/env python3
"""Export three-inference-seed mean LQCR AP curves for the manuscript."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "data/24_chromosomes_object/coco/test/_annotations.coco.json"
OUT = ROOT / "docs/paper/latex/figures/v2/data/source_lqcr_test_ap_curve.json"


def curve(path: str) -> dict[str, float]:
    gt = COCO(str(ANN))
    dt = gt.loadRes(path)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    precision = ev.eval["precision"]
    result = {}
    for index, threshold in enumerate(ev.params.iouThrs):
        values = precision[index, :, :, 0, 2]
        valid = values[values > -1]
        result[f"{threshold:.2f}"] = float(valid.mean())
    return result


def collect(paper_id: str) -> tuple[dict, list[str]]:
    paths = sorted(glob.glob(str(
        ROOT / "results/paper_test_multiseed" / paper_id / paper_id /
        "*/predictions.bbox.json")))
    if len(paths) != 3:
        raise RuntimeError(f"expected three predictions for {paper_id}, got {len(paths)}")
    curves = [curve(path) for path in paths]
    mean = {threshold: float(np.mean([row[threshold] for row in curves]))
            for threshold in curves[0]}
    return {"mAP": float(np.mean(list(mean.values()))), "AP_by_iou": mean}, paths


def main() -> None:
    beta0, paths0 = collect("d2_lqcr_beta_0p0")
    beta2, paths2 = collect("d2_lqcr_beta_2p0")
    payload = {
        "description": "Dataset 2 LQCR held-out-test AP curves; fixed-checkpoint mean over inference seeds 42, 123, and 789",
        "beta0": {"baseline": beta0},
        "beta2": {"baseline": beta2},
        "provenance": {
            "annotation": str(ANN.relative_to(ROOT)),
            "beta0_predictions": [str(Path(path).relative_to(ROOT)) for path in paths0],
            "beta2_predictions": [str(Path(path).relative_to(ROOT)) for path in paths2],
            "protocol": "COCO bbox, maxDets=100, 1,000 held-out images, fixed checkpoint, inference seeds 42/123/789",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
