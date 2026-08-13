#!/usr/bin/env python3
"""Check that the new D1 KaryoFlow recipe preserves running semantics."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def normalize_model(value):
    value = deepcopy(value.to_dict())
    head = value['bbox_head']
    head.setdefault('box_renewal', True)
    head.setdefault('use_ensemble', True)
    head.setdefault('ddim_sampling_eta', 1.0)
    return value


def strip_non_scientific_loader(value):
    value = deepcopy(value.to_dict())
    dataset = value.get('dataset', {})
    dataset.get('metainfo', {}).pop('palette', None)
    dataset.pop('backend_args', None)
    return value


def main() -> int:
    old = Config.fromfile(ROOT / 'experiments/configs/self1700/karyoflow.py')
    new = Config.fromfile(ROOT / 'experiments/configs/v2/recipes/d1/karyoflow.py')
    checks = {
        'model': normalize_model(old.model) == normalize_model(new.model),
        'train_cfg': old.train_cfg == new.train_cfg,
        'optimizer': old.optim_wrapper == new.optim_wrapper,
        'scheduler': old.param_scheduler == new.param_scheduler,
        'train_loader': strip_non_scientific_loader(old.train_dataloader)
                        == strip_non_scientific_loader(new.train_dataloader),
        'val_loader': strip_non_scientific_loader(old.val_dataloader)
                      == strip_non_scientific_loader(new.val_dataloader),
        'test_loader': strip_non_scientific_loader(old.test_dataloader)
                       == strip_non_scientific_loader(new.test_dataloader),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print('D1 V2 EQUIVALENCE: FAIL', ', '.join(failed))
        return 1
    print('D1 V2 EQUIVALENCE: PASS')
    print('model/train/optimizer/scheduler/pipelines preserved')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
