#!/usr/bin/env python3
"""Classify a legacy experiment against a canonical v2 matrix-method target."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import gzip
import io
import json
from pathlib import Path
import re
import sys

from mmengine.config import Config, ConfigDict
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments.v2_matrix import (  # noqa: E402
    resolve_config, scientific_hash)

METRIC_KEYS = ('mAP', 'AP50', 'AP75', 'AP_S', 'AP_M', 'AP_L')
SEED_PATTERN = re.compile(r'(?:seed|trainseed)[_-]?(\d+)', re.IGNORECASE)
LEGACY_DIFFUSION_SCHEDULE_BUFFERS = {
    'bbox_head.alphas_cumprod_prev',
    'bbox_head.posterior_log_variance_clipped',
    'bbox_head.posterior_variance',
    'bbox_head.sqrt_alphas_cumprod',
    'bbox_head.sqrt_one_minus_alphas_cumprod',
    'bbox_head.sqrt_recip_alphas_cumprod',
    'bbox_head.sqrt_recipm1_alphas_cumprod',
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def shown(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def plain(value):
    if isinstance(value, (Config, ConfigDict)):
        value = value.to_dict()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def strip_nonsemantic(value):
    value = plain(value)
    if isinstance(value, dict):
        output = {}
        for key, child in value.items():
            if key == 'palette' or (key == 'backend_args' and child in (None, [], {})):
                continue
            output[key] = strip_nonsemantic(child)
        return output
    if isinstance(value, list):
        return [strip_nonsemantic(v) for v in value]
    return value


def normalized_model(cfg: Config):
    model = deepcopy(plain(cfg.model))
    # These defaults belong to the LDMDet diffusion head only. Injecting them
    # into generic MMDetection heads corrupts comparisons, and Cascade R-CNN
    # has no top-level bbox_head at all.
    head = model.get('bbox_head')
    detector_type = str(model.get('type', ''))
    head_type = str(head.get('type', '')) if isinstance(head, dict) else ''
    is_ldmdet = (
        detector_type in {'LDMDet', 'LDMDetDetector'}
        or 'DiffusionDetHead' in head_type
    )
    if is_ldmdet and isinstance(head, dict):
        head.setdefault('box_renewal', True)
        head.setdefault('use_ensemble', True)
        head.setdefault('ddim_sampling_eta', 1.0)
    return strip_nonsemantic(model)


def scientific_sections(cfg: Config) -> dict:
    hooks = [hook for hook in cfg.get('custom_hooks', [])
             if hook.get('type') != 'CopyProjectHook']
    return {
        'model': normalized_model(cfg),
        'optimization': strip_nonsemantic({
            'optim_wrapper': cfg.get('optim_wrapper'),
            'param_scheduler': cfg.get('param_scheduler'),
            'train_cfg': cfg.get('train_cfg'),
            'custom_hooks': hooks,
        }),
        'data': strip_nonsemantic({
            'train_dataloader': cfg.get('train_dataloader'),
            'val_dataloader': cfg.get('val_dataloader'),
            'test_dataloader': cfg.get('test_dataloader'),
            'val_evaluator': cfg.get('val_evaluator'),
            'test_evaluator': cfg.get('test_evaluator'),
        }),
    }


def recursive_diff(left, right, prefix='') -> list[dict]:
    if isinstance(left, dict) and isinstance(right, dict):
        output = []
        for key in sorted(set(left) | set(right)):
            path = f'{prefix}.{key}' if prefix else key
            if key not in left:
                output.append(dict(path=path, legacy='__MISSING__',
                                   target=right[key]))
            elif key not in right:
                output.append(dict(path=path, legacy=left[key],
                                   target='__MISSING__'))
            else:
                output.extend(recursive_diff(left[key], right[key], path))
        return output
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [dict(path=prefix, legacy=left, target=right)]
        output = []
        for index, (a, b) in enumerate(zip(left, right)):
            output.extend(recursive_diff(a, b, f'{prefix}[{index}]'))
        return output
    return [] if left == right else [dict(path=prefix, legacy=left, target=right)]


def leaf_dataset(cfg: Config, split: str):
    loader = cfg[f'{split}_dataloader']
    dataset = loader.dataset
    # MultiImageMixDataset and similar wrappers keep the annotation fields on
    # their nested dataset.
    while isinstance(dataset, (dict, ConfigDict)) and 'dataset' in dataset \
            and ('ann_file' not in dataset or 'data_root' not in dataset
                 or 'metainfo' not in dataset):
        dataset = dataset['dataset']
    return dataset


def annotation_path(cfg: Config, split: str) -> Path:
    dataset = leaf_dataset(cfg, split)
    ann_file = Path(dataset.get('ann_file', ''))
    data_root = Path(dataset.get('data_root', ''))
    candidates = []
    if ann_file.is_absolute():
        candidates.append(ann_file)
    else:
        # Some legacy configs already prefix ann_file with data_root.
        candidates.extend([ROOT / ann_file, ROOT / data_root / ann_file])
        if data_root.is_absolute():
            candidates.append(data_root / ann_file)
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_file():
            return resolved
    return candidates[-1].resolve()


def dataset_record(cfg: Config, split: str,
                   annotation_override: Path | None = None) -> dict:
    configured_path = annotation_path(cfg, split)
    path = annotation_override or configured_path
    data = json.loads(path.read_text())
    categories = [(int(x['id']), str(x['name'])) for x in data['categories']]
    configured = list(leaf_dataset(cfg, split).metainfo.classes)
    # pycocotools getCatIds(catNms=...) filters categories by membership while
    # retaining annotation order. Record the effective label mapping rather
    # than treating a display-order mismatch as label corruption.
    effective = [(cat_id, name) for cat_id, name in categories
                 if name in set(configured)]
    return dict(
        path=shown(path), configured_path=shown(configured_path),
        annotation_overridden=(path != configured_path),
        sha256=sha256_file(path),
        images=len(data['images']), annotations=len(data['annotations']),
        categories=categories, configured_classes=configured,
        configured_matches_annotation=(configured == [x[1] for x in categories]),
        effective_categories=effective,
    )


def seed_candidates(value, prefix='') -> list[dict]:
    output = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f'{prefix}.{key}' if prefix else str(key)
            if 'seed' in str(key).lower() and isinstance(child, int):
                output.append(dict(source=path, value=child))
            output.extend(seed_candidates(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            output.extend(seed_candidates(child, f'{prefix}[{index}]'))
    return output


def log_seed_candidates(path: Path | None) -> list[int]:
    if path is None:
        return []
    if path.suffix == '.gz':
        with gzip.open(path, mode='rt', errors='replace') as stream:
            text = stream.read()
    else:
        text = path.read_text(errors='replace')
    patterns = (
        r'numpy_random_seed\s*[:=]\s*(\d+)',
        r'Randomness\([^\n]*seed\s*=\s*(\d+)',
        r'training_seed\s*[:=]\s*(\d+)',
    )
    return sorted({int(value) for pattern in patterns
                   for value in re.findall(pattern, text, re.IGNORECASE)})


def recompute_coco(annotation: Path, prediction: Path) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation))
        dt = gt.loadRes(str(prediction))
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.imgIds = sorted(gt.getImgIds())
        evaluator.params.catIds = sorted(gt.getCatIds())
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    return dict(mAP=float(stats[0]), AP50=float(stats[1]),
                AP75=float(stats[2]), AP_S=float(stats[3]),
                AP_M=float(stats[4]), AP_L=float(stats[5]))


def verify_summary(summary_path: Path, legacy_config: Path,
                   checkpoint: Path, test_annotation: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    base = summary_path.parent
    prediction = base / 'predictions.bbox.json'
    issues = []
    required = [prediction, base / 'evaluation.log',
                base / 'resolved_test_config.py']
    for path in required:
        if not path.is_file():
            issues.append(f'missing result artifact: {shown(path)}')
    expected_hashes = summary.get('artifact_sha256', {})
    artifact_hashes = {}
    for path in required:
        if path.is_file():
            digest = sha256_file(path)
            artifact_hashes[path.name] = digest
            if expected_hashes.get(path.name) != digest:
                issues.append(f'artifact hash mismatch: {path.name}')
    protocol = summary.get('protocol', {})
    checks = {
        'config_sha256': (protocol.get('config_source_sha256'),
                          sha256_file(legacy_config)),
        'checkpoint_sha256': (protocol.get('checkpoint_sha256'),
                              sha256_file(checkpoint)),
        'annotation_sha256': (protocol.get('annotation_sha256'),
                              sha256_file(test_annotation)),
        'split': (protocol.get('split'), 'test'),
        'max_dets': (protocol.get('max_dets'), 100),
    }
    for name, (recorded, actual) in checks.items():
        if recorded != actual:
            issues.append(f'{name} mismatch: {recorded!r} != {actual!r}')
    recomputed = recompute_coco(test_annotation, prediction) if prediction.is_file() else {}
    recorded_metrics = summary.get('metrics_exact', {})
    metric_deltas = {
        key: recomputed.get(key, float('nan')) - recorded_metrics.get(key, float('nan'))
        for key in METRIC_KEYS
    }
    if any(abs(delta) > 1e-12 for delta in metric_deltas.values()):
        issues.append('independent COCO metric recomputation mismatch')
    return dict(
        path=shown(summary_path), sha256=sha256_file(summary_path),
        protocol=protocol, artifacts=artifact_hashes,
        recorded_metrics=recorded_metrics, recomputed_metrics=recomputed,
        metric_deltas=metric_deltas, issues=issues,
    )


def checkpoint_record(path: Path, target: Config) -> dict:
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    meta = checkpoint.get('meta', {})
    from mmengine.utils import import_modules_from_strings
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**target.custom_imports)
    model = MODELS.build(target.model)
    target_state = model.state_dict()
    missing = sorted(set(target_state) - set(state))
    unexpected = sorted(set(state) - set(target_state))
    shape_mismatch = sorted(
        key for key in set(state) & set(target_state)
        if tuple(state[key].shape) != tuple(target_state[key].shape))
    strict_load = not missing and not unexpected and not shape_mismatch
    legacy_schedule_buffers_only = (
        not missing and not shape_mismatch and bool(unexpected)
        and set(unexpected) <= LEGACY_DIFFUSION_SCHEDULE_BUFFERS
    )
    if strict_load:
        model.load_state_dict(state, strict=True)
    return dict(
        path=shown(path), sha256=sha256_file(path),
        tensors=len(state), target_tensors=len(target_state),
        strict_load=strict_load, missing=missing,
        unexpected=unexpected, shape_mismatch=shape_mismatch,
        legacy_schedule_buffers_only=legacy_schedule_buffers_only,
        meta_seed_candidates=seed_candidates(meta),
    )


def checkpoint_lineage(parent_path: Path, child_path: Path) -> dict:
    parent_raw = torch.load(parent_path, map_location='cpu')
    child_raw = torch.load(child_path, map_location='cpu')
    parent = parent_raw.get('state_dict', parent_raw)
    child = child_raw.get('state_dict', child_raw)
    shared = sorted(set(parent) & set(child))
    changed = [name for name in shared
               if tuple(parent[name].shape) != tuple(child[name].shape)
               or not torch.equal(parent[name], child[name])]
    parent_only = sorted(set(parent) - set(child))
    child_only = sorted(set(child) - set(parent))
    return dict(
        parent_path=shown(parent_path), parent_sha256=sha256_file(parent_path),
        shared_tensor_count=len(shared), changed_shared_tensors=changed,
        parent_only_tensors=parent_only, child_only_tensors=child_only,
        final_quality_only=(
            not changed and not parent_only and len(child_only) == 5
            and all('.head_series.5.quality_head.' in f'.{name}'
                    for name in child_only)),
        parent_meta_seed_candidates=sorted({
            item['value'] for item in seed_candidates(parent_raw.get('meta', {}))}),
    )


def directory_seed(paths: list[Path]) -> list[int]:
    return sorted({int(value) for path in paths
                   for value in SEED_PATTERN.findall(str(path))})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--parent-checkpoint')
    parser.add_argument('--training-log', required=True)
    parser.add_argument('--result-summary', required=True)
    parser.add_argument('--target-matrix', required=True)
    parser.add_argument('--target-method', required=True)
    parser.add_argument('--target-seed', type=int, default=42)
    parser.add_argument('--output')
    args = parser.parse_args()

    legacy_path = resolve_path(args.legacy_config)
    checkpoint_path = resolve_path(args.checkpoint)
    parent_path = resolve_path(args.parent_checkpoint) \
        if args.parent_checkpoint else None
    log_path = resolve_path(args.training_log)
    summary_path = resolve_path(args.result_summary)
    required = [legacy_path, checkpoint_path, log_path, summary_path]
    if parent_path is not None:
        required.append(parent_path)
    absent = [shown(path) for path in required if not path.is_file()]
    if absent:
        report = dict(
            schema_version=1,
            generated_at=datetime.now(timezone.utc).isoformat(),
            report_id='legacy-compat-missing-' + hashlib.sha256(
                json.dumps(absent, sort_keys=True).encode()).hexdigest()[:12],
            status='MISSING_EVIDENCE', canonical_import_allowed=False,
            legacy_preservation_allowed=False,
            requested=dict(
                legacy_config=args.legacy_config, checkpoint=args.checkpoint,
                parent_checkpoint=args.parent_checkpoint,
                training_log=args.training_log,
                result_summary=args.result_summary,
                target_matrix=args.target_matrix,
                target_method=args.target_method,
                target_seed=args.target_seed),
            missing=absent,
        )
        output = (resolve_path(args.output) if args.output else
                  ROOT / 'tools/experiment_db/compatibility_reports' /
                  f'{report["report_id"]}.json')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(json.dumps(dict(status='MISSING_EVIDENCE',
                              report=shown(output), missing=absent), indent=2))
        return 2

    legacy = Config.fromfile(legacy_path)
    target, sources = resolve_config(
        args.target_matrix, args.target_method, args.target_seed)
    differences = recursive_diff(
        scientific_sections(legacy), scientific_sections(target))
    for difference in differences:
        difference['severity'] = 'scientific'

    summary_preview = json.loads(summary_path.read_text())
    verified_test_annotation = resolve_path(
        summary_preview['protocol']['annotation_path'])
    legacy_data = {
        'train': dataset_record(legacy, 'train'),
        'val': dataset_record(legacy, 'val'),
        # Historical dumped configs sometimes contain a stale test path. The
        # persisted result protocol is authoritative and hash-checked below.
        'test': dataset_record(legacy, 'test', verified_test_annotation),
    }
    target_data = {split: dataset_record(target, split)
                   for split in ('train', 'val', 'test')}
    dataset_issues = []
    dataset_warnings = []
    for split in ('train', 'val', 'test'):
        if legacy_data[split]['sha256'] != target_data[split]['sha256']:
            dataset_issues.append(f'{split} annotation SHA mismatch')
        if not legacy_data[split]['configured_matches_annotation']:
            dataset_warnings.append(
                f'legacy {split} configured class display order differs')
        if not target_data[split]['configured_matches_annotation']:
            dataset_issues.append(f'target {split} class order mismatch')
        if legacy_data[split]['effective_categories'] != \
                target_data[split]['effective_categories']:
            dataset_issues.append(f'{split} effective category mapping mismatch')

    checkpoint = checkpoint_record(checkpoint_path, target)
    parent_required = target.experiment.get('parent_checkpoint_required', False)
    lineage = (checkpoint_lineage(parent_path, checkpoint_path)
               if parent_path is not None else None)
    result = verify_summary(
        summary_path, legacy_path, checkpoint_path,
        Path(target_data['test']['path']) if Path(target_data['test']['path']).is_absolute()
        else ROOT / target_data['test']['path'])
    summary_seed = result['protocol'].get('training_seed')
    config_seeds = seed_candidates(plain(legacy.get('randomness', {})))
    checkpoint_seeds = sorted({x['value'] for x in checkpoint['meta_seed_candidates']})
    log_seeds = log_seed_candidates(log_path)
    if parent_required:
        parent_seeds = (lineage or {}).get('parent_meta_seed_candidates', [])
        parent_evidence = [value for value in [summary_seed] + parent_seeds
                           if isinstance(value, int)]
        child_evidence = checkpoint_seeds + log_seeds
        parent_consistent = bool(parent_evidence) and len(set(parent_evidence)) == 1
        child_consistent = bool(child_evidence) and len(set(child_evidence)) == 1
        seed_consistent = parent_consistent and child_consistent
        authoritative_seed = parent_evidence[0] if parent_consistent else None
        child_seed = child_evidence[0] if child_consistent else None
    else:
        parent_seeds = []
        authoritative = [value for value in [summary_seed] + checkpoint_seeds + log_seeds
                         if isinstance(value, int)]
        seed_consistent = bool(authoritative) and len(set(authoritative)) == 1
        authoritative_seed = authoritative[0] if seed_consistent else None
        child_seed = None
    seeds = dict(
        directory_labels=directory_seed([legacy_path, checkpoint_path, log_path]),
        config_candidates=config_seeds,
        checkpoint_candidates=checkpoint_seeds,
        log_candidates=log_seeds,
        result_summary_training_seed=summary_seed,
        parent_checkpoint_candidates=parent_seeds,
        authoritative_training_seed=authoritative_seed,
        child_training_seed=child_seed,
        authoritative_consistent=seed_consistent,
    )

    hard_issues = list(dataset_issues) + list(result['issues'])
    structural_issues = []
    if not checkpoint['strict_load']:
        if checkpoint['legacy_schedule_buffers_only']:
            structural_issues.append(
                'checkpoint retains legacy derived diffusion-schedule buffers')
        else:
            hard_issues.append('checkpoint is not structurally compatible')
    if parent_required and parent_path is None:
        hard_issues.append('paired child is missing parent checkpoint evidence')
    if lineage is not None and not lineage['final_quality_only']:
        hard_issues.append('parent-child tensor isolation failed')
    if not seed_consistent:
        hard_issues.append('authoritative training seed evidence conflicts')
    for difference in differences:
        path = difference['path']
        if '.metainfo.classes' in path:
            difference['severity'] = 'metadata_only'
        elif path.startswith('data.test_evaluator.') or \
                path.startswith('data.test_dataloader.dataset.ann_file'):
            difference['severity'] = 'overridden_by_verified_test_protocol'
        elif path.endswith('.pin_memory'):
            difference['severity'] = 'runtime_only'
        else:
            difference['severity'] = 'scientific'
    scientific_differences = [d for d in differences
                              if d['severity'] == 'scientific']

    if hard_issues:
        status = 'INCOMPATIBLE'
    elif scientific_differences or structural_issues:
        status = 'STRUCTURAL_ONLY'
    else:
        status = 'EXACT'

    identity_payload = dict(
        target_config_id=target.experiment.config_id,
        target_scientific_config_sha256=scientific_hash(target),
        legacy_config_sha256=sha256_file(legacy_path),
        checkpoint_sha256=checkpoint['sha256'],
        training_log_sha256=sha256_file(log_path),
        result_summary_sha256=sha256_file(summary_path),
        parent_checkpoint_sha256=(lineage or {}).get('parent_sha256'),
        status=status,
        scientific_differences=scientific_differences,
        structural_issues=structural_issues,
        hard_issues=hard_issues,
    )
    identity = hashlib.sha256(json.dumps(
        identity_payload, sort_keys=True, separators=(',', ':'), default=str
    ).encode()).hexdigest()
    report = dict(
        schema_version=1,
        generated_at=result['protocol'].get('completed_at') or
                     json.loads(summary_path.read_text()).get('completed_at'),
        status=status,
        canonical_import_allowed=(status == 'EXACT'),
        legacy_preservation_allowed=(status in {'EXACT', 'STRUCTURAL_ONLY'}),
        target=dict(
            matrix_id=target.experiment.matrix_id,
            config_id=target.experiment.config_id,
            method_id=target.experiment.method_id,
            scientific_config_sha256=scientific_hash(target),
            matrix_path=shown(sources['matrix_path']),
            method_path=shown(sources['method_path']),
        ),
        legacy=dict(
            config_path=shown(legacy_path),
            config_sha256=sha256_file(legacy_path),
            checkpoint=checkpoint,
            parent_child_lineage=lineage,
            training_log=dict(path=shown(log_path), sha256=sha256_file(log_path)),
        ),
        seeds=seeds,
        dataset=dict(legacy=legacy_data, target=target_data,
                     issues=dataset_issues, warnings=dataset_warnings),
        result_evidence=result,
        configuration_differences=differences,
        scientific_differences=scientific_differences,
        structural_issues=structural_issues,
        hard_issues=hard_issues,
        identity_payload_sha256=identity,
    )
    report['report_id'] = f'legacy-compat-{identity[:12]}'
    output = (resolve_path(args.output) if args.output else
              ROOT / 'tools/experiment_db/compatibility_reports' /
              f'{report["report_id"]}.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    temporary.replace(output)
    print(json.dumps(dict(status=status, report=shown(output),
                          differences=len(differences),
                          scientific_differences=len(scientific_differences),
                          structural_issues=structural_issues,
                          hard_issues=hard_issues,
                          canonical_import_allowed=report['canonical_import_allowed']),
                     indent=2))
    return 0 if status in {'EXACT', 'STRUCTURAL_ONLY'} else 2


if __name__ == '__main__':
    raise SystemExit(main())
