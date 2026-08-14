#!/usr/bin/env python3
"""Register canonical matrix-method combinations without rewriting run records."""

from __future__ import annotations
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments.matrix import (
    load_matrix,
    resolve_config,
    scientific_hash,
)

CONFIG_ROOT = ROOT / 'experiments/configs'
DB = ROOT / 'tools/experiment_db/experiments.db'
CATALOG = ROOT / 'experiments/manifests/experiment_catalog.json'


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


def unwrap_dataset(dataset):
    """Return the innermost concrete dataset of MMDet dataset wrappers."""
    while isinstance(dataset, dict) and isinstance(
        dataset.get('dataset'), dict
    ):
        dataset = dataset['dataset']
    return dataset


def find_num_classes(value):
    """Find the first declared class count across heterogeneous heads."""
    if isinstance(value, dict):
        if isinstance(value.get('num_classes'), int):
            return value['num_classes']
        for child in value.values():
            found = find_num_classes(child)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = find_num_classes(child)
            if found is not None:
                return found
    return None


def main() -> int:
    records = []
    conn = sqlite3.connect(DB)
    conn.execute('PRAGMA foreign_keys=ON')
    referenced = conn.execute(
        """SELECT COUNT(*) FROM experiment
           WHERE config_path LIKE 'experiments/configs/recipes/%'"""
    ).fetchone()[0]
    if referenced:
        raise RuntimeError(
            'legacy v2 recipe rows are already referenced by runs'
        )
    conn.execute(
        "DELETE FROM config WHERE config_path LIKE 'experiments/configs/recipes/%'"
    )

    for matrix_path in sorted((CONFIG_ROOT / 'matrices').glob('*.yaml')):
        _, matrix = load_matrix(matrix_path)
        for method_name in matrix['methods']:
            cfg, sources = resolve_config(
                matrix_path, method_name, matrix['training_seeds'][0]
            )
            meta = cfg.experiment
            digest = scientific_hash(cfg)
            identity = f'{matrix_path.relative_to(ROOT)}#method={method_name}'
            wrapped_dataset = cfg.train_dataloader.dataset
            concrete_dataset = unwrap_dataset(wrapped_dataset)
            pipeline = wrapped_dataset.get(
                'pipeline', concrete_dataset.get('pipeline', [])
            )
            pipeline_hash, step_types = pipeline_metadata(pipeline)
            head = cfg.model.get('bbox_head', {})
            coupling = head.get('coupling', {})
            coupling_type = coupling.get('type') if coupling else None
            single_head = head.get('single_head', {})
            record = dict(
                config_id=meta.config_id,
                matrix_path=str(matrix_path.relative_to(ROOT)),
                dataset_config=str(sources['dataset_path'].relative_to(ROOT)),
                method_config=str(sources['method_path'].relative_to(ROOT)),
                method_name=method_name,
                scientific_config_sha256=digest,
                dataset_id=cfg.dataset_id,
                method_id=meta.method_id,
                role=meta.role,
                parent_method_id=meta.get('parent_method_id'),
                training_seeds=matrix['training_seeds'],
                coupling=coupling_type,
                solver=head.get('solver_type'),
                steps=head.get('sampling_timesteps'),
                time_conditioning=single_head.get('time_conditioning'),
                renewal=head.get('box_renewal'),
            )
            records.append(record)
            conn.execute(
                """INSERT OR IGNORE INTO aug_pipeline(
                  pipeline_hash,pipeline_name,has_random_choice_resize,
                  has_random_crop,has_random_flip,has_fixed_resize,
                  step_types_json,description) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    pipeline_hash,
                    f'v2:{cfg.dataset_id}',
                    'RandomChoiceResize' in step_types,
                    'RandomCrop' in step_types,
                    'RandomFlip' in step_types,
                    'Resize' in step_types,
                    json.dumps(step_types),
                    'Canonical v2 training augmentation pipeline',
                ),
            )
            conn.execute(
                """INSERT OR REPLACE INTO config(
                  config_path,source_config_path,dataset,data_root,
                  aug_pipeline_hash,coupling_type,ot_epsilon,ot_matcher,
                  ot_sample,coupling_mode,time_conditioning,solver_type,
                  rf_schedule,rf_shift,batch_size,max_epochs,num_classes,
                  num_proposals,sampling_timesteps,has_early_stopping)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    identity,
                    f'v2:{meta.config_id}:{digest}',
                    cfg.dataset_id,
                    cfg.get('data_root', concrete_dataset.get('data_root')),
                    pipeline_hash,
                    coupling_type,
                    None,
                    None,
                    False,
                    None,
                    single_head.get('time_conditioning'),
                    head.get('solver_type'),
                    head.get('rf_schedule'),
                    head.get('rf_shift'),
                    cfg.train_dataloader.batch_size,
                    cfg.train_cfg.max_epochs,
                    find_num_classes(cfg.model),
                    head.get('num_proposals', cfg.model.get('num_queries')),
                    head.get('sampling_timesteps'),
                    any(
                        h.get('type') == 'EarlyStoppingHook'
                        for h in cfg.get('custom_hooks', [])
                    ),
                ),
            )
    conn.commit()
    conn.close()
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    CATALOG.write_text(
        json.dumps(
            dict(schema_version=2, experiment_definitions=records),
            indent=2,
            sort_keys=True,
        )
        + '\n'
    )
    print(
        json.dumps(
            dict(
                status='PASS',
                combinations=len(records),
                catalog=str(CATALOG.relative_to(ROOT)),
            ),
            indent=2,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
