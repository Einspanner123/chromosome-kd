#!/usr/bin/env python3
"""Validate v2 configuration structure and resolved scientific invariants."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import sys

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / 'experiments/configs/v2'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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
    return 0 if not parents else 1 + max(depth(p, seen + (path,)) for p in parents)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-models', action='store_true')
    args = parser.parse_args()
    errors = []
    files = sorted(V2.rglob('*.py'))
    config_ids = set()
    for path in files:
        text = path.read_text()
        if FORBIDDEN.search(text):
            errors.append(f'{path.relative_to(ROOT)}: forbidden path/host/credential')
        try:
            d = depth(path)
            if d > 3:
                errors.append(f'{path.relative_to(ROOT)}: inheritance depth {d} > 3')
            for parent in bases(path):
                if not parent.exists():
                    errors.append(f'{path.relative_to(ROOT)}: missing parent {parent}')
                elif V2 not in parent.parents and parent != V2:
                    errors.append(f'{path.relative_to(ROOT)}: inherits outside v2')
        except Exception as exc:
            errors.append(f'{path.relative_to(ROOT)}: {exc}')

    recipes = sorted((V2 / 'recipes').rglob('*.py'))
    resolved = []
    for path in recipes:
        try:
            cfg = Config.fromfile(path)
            meta = cfg.get('experiment', {})
            cid = meta.get('config_id')
            if not cid or cid in config_ids:
                errors.append(f'{path.relative_to(ROOT)}: missing/duplicate config_id {cid}')
            config_ids.add(cid)
            if cfg.model.type != 'LDMDet':
                errors.append(f'{path.relative_to(ROOT)}: expected LDMDet')
            if meta.get('dataset_id') != cfg.dataset_id:
                errors.append(f'{path.relative_to(ROOT)}: metadata/dataset mismatch')
            if cfg.test_evaluator.get('format_only', False):
                errors.append(f'{path.relative_to(ROOT)}: test evaluator must compute metrics')
            h = cfg.model.bbox_head
            if meta.get('method_id') == 'karyoflow_r50':
                expected = ('rectified_flow', 'dpm_solver_pp', 4, 'random')
                actual = (h.diffusion_type, h.solver_type,
                          h.sampling_timesteps, h.coupling.type)
                if actual != expected:
                    errors.append(f'{path.relative_to(ROOT)}: KaryoFlow invariant {actual}')
            if 'lqcr' in meta.get('method_id', ''):
                if h.quality_calibration_mode != 'final_only' or not h.quality_only_training:
                    errors.append(f'{path.relative_to(ROOT)}: LQCR is not final-only')
            resolved.append((path, cfg))
        except Exception as exc:
            errors.append(f'{path.relative_to(ROOT)}: resolve failed: {exc}')

    if args.build_models and not errors:
        from mmengine.utils import import_modules_from_strings
        from mmdet.registry import MODELS
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=True)
        for path, cfg in resolved:
            import_modules_from_strings(**cfg.custom_imports)
            try:
                MODELS.build(cfg.model)
            except Exception as exc:
                errors.append(f'{path.relative_to(ROOT)}: model build failed: {exc}')

    if errors:
        print('V2 CONFIG AUDIT: FAIL')
        for error in errors:
            print('-', error)
        return 1
    print('V2 CONFIG AUDIT: PASS')
    print(f'files={len(files)} recipes={len(recipes)} max_depth={max(map(depth, files))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
