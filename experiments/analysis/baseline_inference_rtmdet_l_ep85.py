#!/usr/bin/env python3
"""RTMDet-L ep85 (actual best) inference on Dataset 2 (24obj) val.

Background:
  - baseline_inference_24obj.py used epoch_86.pth (mAP=0.8610, last epoch),
    with a comment "ep85 best lost".
  - VERIFICATION (2026-07-20): epoch_85.pth ACTUALLY EXISTS (667 MB),
    and train.log confirms ep85 is the best epoch (mAP=0.8630).
  - The paper-reported RTMDet-L mAP=0.869 in Table 6 is WRONG — training
    never reached 0.869; the actual best is 0.8630 @ ep85.
  - This script re-runs RTMDet-L inference using ep85 (true best) to get
    fair per-image predictions for paired Wilcoxon test against RF.

Input:
  - RTMDet-L checkpoint: work_dirs/baselines/rtmdet_l_24obj/epoch_85.pth
    (best epoch, mAP=0.8630 per train.log)
  - Config: experiments/configs/baselines/benchmark_24obj/rtmdet_l.py

Output:
  - experiments/analysis/baseline_inference_24obj_cache/
      RTMDet_L_ep85_seed42_preds.json  (per-image COCO DT predictions)
  - experiments/analysis/baseline_inference_rtmdet_l_ep85_run.log
"""

from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import torch

# Ensure REPO root is on sys.path so custom_imports resolves
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Trigger mmdet's registry registration (same workaround as
# baseline_inference_24obj.py — mmdet fork's transform registry scope issue)
import mmdet.datasets  # noqa: F401
import mmdet.datasets.transforms  # noqa: F401
from mmengine.registry import TRANSFORMS as _MMENGINE_TRANSFORMS
from mmdet.registry import TRANSFORMS as _MMDET_TRANSFORMS
for _name, _module in list(_MMDET_TRANSFORMS._module_dict.items()):
    if _name not in _MMENGINE_TRANSFORMS._module_dict:
        _MMENGINE_TRANSFORMS.register_module(
            name=_name, module=_module, force=True
        )

from mmengine.config import Config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANN_FILE = str(
    REPO / 'data/24_chromosomes_object/coco/valid/_annotations.coco.json'
)

CONFIG_PATH = str(
    REPO / 'experiments/configs/baselines/benchmark_24obj/rtmdet_l.py'
)

# Use ep85 (actual best, mAP=0.8630 per train.log) — NOT ep86 (last, 0.8610)
CKPT_PATH = str(
    REPO / 'work_dirs/baselines/rtmdet_l_24obj/epoch_85.pth'
)

OUTPUT_PATH = (
    REPO
    / 'experiments/analysis/baseline_inference_24obj_cache'
    / 'RTMDet_L_ep85_seed42_preds.json'
)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
for i, arg in enumerate(sys.argv):
    if arg == '--device' and i + 1 < len(sys.argv):
        DEVICE = sys.argv[i + 1]
        break


def _run_inference(cfg: Config, ckpt_path: str, device: str) -> list[dict]:
    """Run inference on val set, return COCO DT-format predictions."""
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS

    val_dataset_cfg = cfg.val_dataloader.dataset
    dataset = DATASETS.build(val_dataset_cfg)

    model = init_detector(cfg, ckpt_path, device=device)
    model.eval()

    predictions: list[dict] = []
    n_imgs = len(dataset)
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_imgs):
            data = dataset[i]
            data['inputs'] = data['inputs'].unsqueeze(0).to(device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]

            out = model.test_step(data)
            for r in (out if isinstance(out, list) else [out]):
                pi = r.pred_instances
                if pi is None or len(pi.bboxes) == 0:
                    continue
                img_id = getattr(r, 'img_id', i)
                scores = pi.scores.cpu().tolist()
                labels = pi.labels.cpu().tolist()
                bboxes = pi.bboxes.cpu().tolist()
                for j in range(len(bboxes)):
                    bbox = bboxes[j]
                    bbox_xywh = [
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                    ]
                    predictions.append({
                        'image_id': int(img_id),
                        'category_id': int(labels[j]) + 1,  # COCO 1-indexed
                        'bbox': [float(x) for x in bbox_xywh],
                        'score': float(scores[j]),
                    })

            if (i + 1) % 50 == 0 or i == n_imgs - 1:
                elapsed = time.time() - t0
                print(f'    inference: {i + 1}/{n_imgs} '
                      f'({elapsed:.1f}s, {len(predictions)} preds so far)')

    return predictions


def _compute_aggregate_metrics(
    ann_file: str, predictions: list[dict]
) -> dict:
    """Compute COCO aggregate metrics (mAP, AP50, AP75, AP_S, AP_M, AP_L)."""
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, 'bbox')
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    return {
        'mAP': float(s[0]),
        'AP50': float(s[1]),
        'AP75': float(s[2]),
        'AP_s': float(s[3]),
        'AP_m': float(s[4]),
        'AP_l': float(s[5]),
    }


def main() -> None:
    print('=' * 70)
    print('RTMDet-L ep85 (actual best) inference on Dataset 2 (24obj) val')
    print('=' * 70)
    print(f'Config: {CONFIG_PATH}')
    print(f'Ckpt:   {CKPT_PATH}')
    print(f'Output: {OUTPUT_PATH}')
    print(f'Device: {DEVICE}')
    print(f'AnnFile: {ANN_FILE}')
    print()

    # Verify checkpoint exists
    if not Path(CKPT_PATH).exists():
        raise FileNotFoundError(f'Checkpoint not found: {CKPT_PATH}')
    print(f'[OK] Checkpoint exists ({Path(CKPT_PATH).stat().st_size / 1e6:.1f} MB)')

    # Load config
    cfg = Config.fromfile(CONFIG_PATH)
    print(f'[OK] Config loaded')

    # Run inference
    print(f'\n[Running inference on {DEVICE}]...')
    predictions = _run_inference(cfg, CKPT_PATH, DEVICE)
    print(f'\n[OK] {len(predictions)} predictions')

    # Compute aggregate metrics
    print(f'\n[Computing aggregate metrics]...')
    metrics = _compute_aggregate_metrics(ANN_FILE, predictions)
    print(f'\nAggregate: mAP={metrics["mAP"]:.4f} AP50={metrics["AP50"]:.4f} '
          f'AP75={metrics["AP75"]:.4f} AP_S={metrics["AP_s"]:.4f} '
          f'AP_M={metrics["AP_m"]:.4f} AP_L={metrics["AP_l"]:.4f}')
    print(f'(train.log reported: ep85 mAP=0.8630)')

    # Save predictions
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(predictions, f)
    print(f'\n[OK] Saved to {OUTPUT_PATH}')

    # Save summary
    summary = {
        'experiment': 'rtmdet_l_ep85_inference_24obj',
        'description': (
            'RTMDet-L ep85 (actual best per train.log) inference on '
            'Dataset 2 val. Corrects baseline_inference_24obj.py which '
            'used ep86 (last epoch, mAP=0.8610) under the false assumption '
            'that ep85 was lost. ep85.pth actually exists (667 MB).'
        ),
        'ann_file': ANN_FILE,
        'dataset': '24_chromosomes_object (valid, 500 imgs)',
        'config': CONFIG_PATH,
        'checkpoint': CKPT_PATH,
        'checkpoint_epoch': 85,
        'train_log_reported_mAP': 0.8630,
        'metrics': {'aggregate': metrics},
        'num_predictions': len(predictions),
        'output': str(OUTPUT_PATH),
    }
    summary_path = OUTPUT_PATH.parent / 'rtmdet_l_ep85_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[OK] Summary saved to {summary_path}')


if __name__ == '__main__':
    main()
