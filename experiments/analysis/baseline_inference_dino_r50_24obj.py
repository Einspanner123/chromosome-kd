#!/usr/bin/env python3
"""DINO R50 inference on Dataset 2 (24obj) val — produces per-image COCO DT
predictions for paired Wilcoxon tests against the RF (LDMDet StochOT) baseline.

Background:
  - The paper-reported DINO R50 mAP=0.868 (actual 0.869 @ ep102 per
    metrics.json on workstation). The .pth checkpoint was originally thought
    lost; on 2026-07-20 it was discovered on the workstation server at
    `work_dirs/dino_r50/best_coco_bbox_mAP_epoch_102.pth` (340 MB) and
    rsynced to ross.
  - This script runs inference ONCE and writes a flat predictions JSON cache
    matching the format of `baseline_inference_24obj_cache/RTMDet_L_seed42_preds.json`
    so the existing `baseline_inference_24obj_perimage.py` can pick it up by
    adding a single entry to its CACHES dict.

Inputs:
  - Config: work_dirs/baselines/dino_r50_24obj/dino_r50.py (full training-time
    config dump from workstation, no _base_ dependency)
  - Checkpoint: work_dirs/baselines/dino_r50_24obj/best_coco_bbox_mAP_epoch_102.pth
  - Annotation: data/24_chromosomes_object/coco/valid/_annotations.coco.json

Output:
  - experiments/analysis/baseline_inference_24obj_cache/DINO_R50_seed42_preds.json
"""

from __future__ import annotations
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

# Ensure REPO root is on sys.path so custom_imports resolves
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Trigger mmdet's full registry registration (transforms etc.)
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANN_FILE = str(
    REPO / 'data/24_chromosomes_object/coco/valid/_annotations.coco.json'
)

CONFIG_PATH = REPO / 'work_dirs/baselines/dino_r50_24obj/dino_r50.py'
CKPT_PATH = REPO / 'work_dirs/baselines/dino_r50_24obj/best_coco_bbox_mAP_epoch_102.pth'

OUTPUT_PATH = (
    REPO / 'experiments/analysis/baseline_inference_24obj_cache/DINO_R50_seed42_preds.json'
)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
for i, arg in enumerate(sys.argv):
    if arg == '--device' and i + 1 < len(sys.argv):
        DEVICE = sys.argv[i + 1]
        break


def _load_config_safe(config_path: Path) -> Config:
    """Load config and strip custom_imports to avoid swanlab dependency."""
    with open(config_path, 'r') as f:
        config_text = f.read()

    # Remove custom_imports block to avoid swanlab import failure at inference
    # (we don't need swanlab for inference)
    if 'custom_imports' in config_text:
        # Naive but effective: replace the custom_imports = dict(...) line
        # with an empty dict. The full dump format is single-line.
        import re
        config_text = re.sub(
            r"custom_imports\s*=\s*dict\([^)]*\)",
            "custom_imports = dict(imports=[], allow_failed_imports=True)",
            config_text,
            count=1,
            flags=re.DOTALL,
        )
        print(f'  [FIX] Stripped custom_imports (avoid swanlab dep)')

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', dir=str(REPO), delete=False
    ) as tmp:
        tmp.write(config_text)
        tmp_path = tmp.name

    try:
        cfg = Config.fromfile(tmp_path)
    finally:
        os.unlink(tmp_path)
    return cfg


def _run_inference(cfg: Config, ckpt_path: Path, device: str) -> list[dict]:
    """Run inference on val set, return COCO DT-format predictions."""
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS

    val_dataset_cfg = cfg.val_dataloader.dataset
    dataset = DATASETS.build(val_dataset_cfg)

    model = init_detector(cfg, str(ckpt_path), device=device)
    model.eval()

    predictions: list[dict] = []
    n_imgs = len(dataset)
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
                for j in range(len(pi.bboxes)):
                    bbox = pi.bboxes[j].cpu().tolist()
                    bbox_xywh = [
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                    ]
                    predictions.append({
                        'image_id': img_id,
                        'category_id': int(pi.labels[j].cpu().item()) + 1,
                        'bbox': bbox_xywh,
                        'score': float(pi.scores[j].cpu().item()),
                    })

            if (i + 1) % 50 == 0:
                print(f'    inference: {i + 1}/{n_imgs}')

    del model
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass
    return predictions


def main():
    print('=' * 100)
    print('DINO R50 Inference: Dataset 2 (24obj) val')
    print('=' * 100)
    print(f'Device: {DEVICE}')
    print(f'Annotation: {ANN_FILE}')
    print(f'Config:    {CONFIG_PATH}')
    print(f'Checkpoint: {CKPT_PATH}')
    print(f'Output:    {OUTPUT_PATH}')
    print()

    if not CKPT_PATH.exists():
        print(f'[ERROR] Checkpoint not found: {CKPT_PATH}')
        sys.exit(1)
    if not CONFIG_PATH.exists():
        print(f'[ERROR] Config not found: {CONFIG_PATH}')
        sys.exit(1)

    cfg = _load_config_safe(CONFIG_PATH)
    print(f'[OK] Config loaded')
    print()

    print('Running inference...')
    predictions = _run_inference(cfg, CKPT_PATH, DEVICE)
    print(f'[OK] {len(predictions)} predictions')

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(predictions, f)
    print(f'[OK] Saved to {OUTPUT_PATH}')

    # Quick sanity: compute aggregate mAP
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(ANN_FILE)
    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, 'bbox')
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    print(f'\nAggregate: mAP={s[0]:.4f} AP50={s[1]:.4f} AP75={s[2]:.4f} '
          f'AP_S={s[3]:.4f} AP_M={s[4]:.4f} AP_L={s[5]:.4f}')
    print(f'(paper-reported: mAP=0.868/0.869 @ ep102)')


if __name__ == '__main__':
    main()
