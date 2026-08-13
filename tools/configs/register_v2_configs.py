#!/usr/bin/env python3
"""Register canonical v2 recipes without rewriting historical run records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import sys

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V2 = ROOT / 'experiments/configs/v2'
DB = ROOT / 'tools/experiment_db/experiments.db'
CATALOG = V2 / 'manifests/recipe_catalog.json'


def pipeline_metadata(pipeline):
    raw = json.dumps(pipeline, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()
    step_types = []

    def visit(value):
        if isinstance(value, dict):
            if 'type' in value:
                step_types.append(str(value['type']))
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(pipeline)
    return digest, step_types


def canonical_hash(cfg: Config) -> str:
    keys = ('dataset_id', 'dataset_manifest_sha256',
            'train_annotation_sha256', 'val_annotation_sha256',
            'test_annotation_sha256', 'model', 'train_dataloader',
            'val_dataloader', 'test_dataloader', 'optim_wrapper',
            'param_scheduler', 'train_cfg')
    payload = {key: cfg.get(key) for key in keys}
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def main() -> int:
    records = []
    conn = sqlite3.connect(DB)
    for path in sorted((V2 / 'recipes').rglob('*.py')):
        cfg = Config.fromfile(path)
        meta = cfg.experiment
        head = cfg.model.bbox_head
        single = head.single_head
        rel = str(path.relative_to(ROOT))
        digest = canonical_hash(cfg)
        pipeline_hash, step_types = pipeline_metadata(
            cfg.train_dataloader.dataset.pipeline)
        record = dict(
            config_id=meta.config_id, path=rel, scientific_config_sha256=digest,
            dataset_id=cfg.dataset_id, method_id=meta.method_id,
            role=meta.role, parent_method_id=meta.get('parent_method_id'),
            coupling=head.coupling.type, solver=head.solver_type,
            steps=head.sampling_timesteps,
            time_conditioning=single.time_conditioning,
            renewal=head.box_renewal,
        )
        records.append(record)
        conn.execute(
            """INSERT OR IGNORE INTO aug_pipeline(
              pipeline_hash,pipeline_name,has_random_choice_resize,
              has_random_crop,has_random_flip,has_fixed_resize,
              step_types_json,description) VALUES(?,?,?,?,?,?,?,?)""",
            (pipeline_hash, f"v2:{cfg.dataset_id}",
             'RandomChoiceResize' in step_types,
             'RandomCrop' in step_types, 'RandomFlip' in step_types,
             'Resize' in step_types,
             json.dumps(step_types),
             'Canonical v2 training augmentation pipeline'),
        )
        conn.execute(
            """INSERT OR REPLACE INTO config(
              config_path,source_config_path,dataset,data_root,aug_pipeline_hash,
              coupling_type,ot_epsilon,ot_matcher,ot_sample,coupling_mode,
              time_conditioning,solver_type,rf_schedule,rf_shift,batch_size,
              max_epochs,num_classes,num_proposals,sampling_timesteps,
              has_early_stopping) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rel, f"v2:{meta.config_id}:{digest}", cfg.dataset_id, cfg.data_root,
             pipeline_hash,
             head.coupling.type, None, None, False, None,
             single.time_conditioning, head.solver_type, head.get('rf_schedule'),
             head.get('rf_shift'), cfg.train_dataloader.batch_size,
             cfg.train_cfg.max_epochs, head.num_classes, head.num_proposals,
             head.sampling_timesteps,
             any(h.get('type') == 'EarlyStoppingHook'
                 for h in cfg.get('custom_hooks', []))),
        )
    conn.commit()
    conn.close()
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    CATALOG.write_text(json.dumps(dict(version=1, recipes=records),
                                  indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status='PASS', recipes=len(records),
                          catalog=str(CATALOG.relative_to(ROOT))), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
