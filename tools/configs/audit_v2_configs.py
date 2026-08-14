#!/usr/bin/env python3
"""Audit v2 sources, matrices, resolved configs, and model invariants."""

from __future__ import annotations
import argparse
import ast
import gc
import re
import sys
import tempfile
from pathlib import Path

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / 'experiments/configs/v2'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments.v2_matrix import (
    load_matrix,
    resolve_config,
    scientific_hash,
)

FORBIDDEN = re.compile(
    r'(/home/|/media/|/data/linkst|linkst@|\bross\b|\bworkstation\b|api_key)',
    re.IGNORECASE,
)


def bases(path: Path) -> list[Path]:
    tree = ast.parse(path.read_text())
    output = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == '_base_' for t in node.targets):
            continue
        value = ast.literal_eval(node.value)
        values = [value] if isinstance(value, str) else value
        output.extend((path.parent / item).resolve() for item in values)
    return output


def depth(path: Path, seen: tuple[Path, ...] = ()) -> int:
    if path in seen:
        raise ValueError(f'cycle: {path}')
    parents = bases(path)
    return 0 if not parents else 1 + max(depth(p, (*seen, path)) for p in parents)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-models', action='store_true')
    parser.add_argument('--roundtrip', action='store_true')
    args = parser.parse_args()
    errors = []
    files = sorted(V2.rglob('*.py'))
    sources = files + sorted((V2 / 'matrices').glob('*.yaml'))
    for path in sources:
        if FORBIDDEN.search(path.read_text()):
            errors.append(f'{path.relative_to(ROOT)}: forbidden path/host/credential')
    for path in files:
        try:
            inherited = depth(path)
            if inherited > 3:
                errors.append(f'{path.relative_to(ROOT)}: inheritance depth {inherited} > 3')
            for parent in bases(path):
                if not parent.exists():
                    errors.append(f'{path.relative_to(ROOT)}: missing parent {parent}')
                elif V2 not in parent.parents and parent != V2:
                    errors.append(f'{path.relative_to(ROOT)}: inherits outside v2')
        except Exception as exc:
            errors.append(f'{path.relative_to(ROOT)}: {exc}')

    if (V2 / 'recipes').exists():
        errors.append('legacy v2/recipes directory must not exist')
    methods = sorted((V2 / 'methods').glob('*.py'))
    matrices = sorted((V2 / 'matrices').glob('*.yaml'))
    for method_path in methods:
        try:
            method_cfg = Config.fromfile(method_path)
            forbidden_keys = {
                'dataset_id', 'data_root', 'train_dataloader',
                'val_dataloader', 'test_dataloader', 'val_evaluator',
                'test_evaluator',
            }
            leaked = sorted(forbidden_keys & set(method_cfg.keys()))
            if leaked:
                errors.append(
                    f'{method_path.relative_to(ROOT)}: dataset fields {leaked}')
        except Exception as exc:
            errors.append(f'{method_path.relative_to(ROOT)}: method load failed: {exc}')
    config_ids = set()
    resolved = []
    for matrix_path in matrices:
        try:
            _, matrix = load_matrix(matrix_path)
            enabled = set(matrix['methods'])
            unknown = enabled - {p.stem for p in methods}
            if unknown:
                errors.append(f'{matrix_path.relative_to(ROOT)}: unknown methods {unknown}')
            for method_name in matrix['methods']:
                cfg, _ = resolve_config(
                    matrix_path, method_name, matrix['training_seeds'][0])
                meta = cfg.experiment
                cid = meta.config_id
                if cid in config_ids:
                    errors.append(f'duplicate config_id {cid}')
                config_ids.add(cid)
                if meta.dataset_id != cfg.dataset_id:
                    errors.append(f'{cid}: metadata/dataset mismatch')
                model_type = cfg.model.type
                allowed_models = {
                    'LDMDet', 'DINO', 'RTMDet', 'CascadeRCNN', 'YOLOX',
                }
                if model_type not in allowed_models:
                    errors.append(f'{cid}: unsupported model type {model_type}')
                if cfg.test_evaluator.get('format_only', False):
                    errors.append(f'{cid}: test evaluator must compute metrics')
                head = cfg.model.get('bbox_head', {})
                if model_type == 'LDMDet' \
                        and meta.method_id == 'karyoflow_r50':
                    expected = ('rectified_flow', 'dpm_solver_pp', 4, 'random')
                    actual = (head.diffusion_type, head.solver_type,
                              head.sampling_timesteps, head.coupling.type)
                    if actual != expected:
                        errors.append(f'{cid}: KaryoFlow invariant {actual}')
                if 'lqcr' in meta.method_id:
                    if model_type != 'LDMDet':
                        errors.append(f'{cid}: LQCR requires LDMDet')
                    elif meta.parent_method_id not in [
                            resolve_config(matrix_path, name,
                                           matrix['training_seeds'][0])[0]
                            .experiment.method_id
                            for name in matrix['methods']]:
                        errors.append(f'{cid}: parent method absent from matrix')
                    if meta.pairing != 'same_training_seed':
                        errors.append(f'{cid}: invalid parent pairing')
                    if head.quality_calibration_mode != 'final_only' \
                            or not head.quality_only_training:
                        errors.append(f'{cid}: LQCR is not final-only')
                if meta.role == 'paired_head_distillation':
                    parent_ids = [
                        resolve_config(matrix_path, name,
                                       matrix['training_seeds'][0])[0]
                        .experiment.method_id
                        for name in matrix['methods']
                    ]
                    if meta.parent_method_id not in parent_ids:
                        errors.append(f'{cid}: distillation parent absent')
                    if meta.get('pairing') != 'same_training_seed':
                        errors.append(f'{cid}: invalid distillation pairing')
                    if meta.get('parent_checkpoint_binding') \
                            != 'teacher_checkpoint':
                        errors.append(
                            f'{cid}: parent must bind as teacher checkpoint')
                    if not head.get('use_distillation', False):
                        errors.append(f'{cid}: distillation is disabled')
                    if head.get('num_heads') != 3:
                        errors.append(f'{cid}: student must have three heads')
                    if head.get('distill_head_map') != {0: 0, 1: 2, 2: 5}:
                        errors.append(f'{cid}: invalid head map')
                resolved.append((cid, cfg))
        except Exception as exc:
            errors.append(f'{matrix_path.relative_to(ROOT)}: resolve failed: {exc}')

    if args.roundtrip and not errors:
        for cid, cfg in resolved:
            with tempfile.NamedTemporaryFile(suffix='.py') as stream:
                cfg.dump(stream.name)
                loaded = Config.fromfile(stream.name)
                if scientific_hash(loaded) != scientific_hash(cfg):
                    errors.append(f'{cid}: resolved config round-trip mismatch')
                del loaded
                gc.collect()

    if args.build_models and not errors:
        from mmengine.utils import import_modules_from_strings

        from mmdet.registry import MODELS
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=True)
        for cid, cfg in resolved:
            import_modules_from_strings(**cfg.custom_imports)
            try:
                model = MODELS.build(cfg.model)
                del model
                gc.collect()
            except Exception as exc:
                errors.append(f'{cid}: model build failed: {exc}')

    if errors:
        print('V2 CONFIG AUDIT: FAIL')
        for error in errors:
            print('-', error)
        return 1
    print('V2 CONFIG AUDIT: PASS')
    print(f'sources={len(sources)} methods={len(methods)} '
          f'matrices={len(matrices)} combinations={len(resolved)} '
          f'max_depth={max(map(depth, files))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
