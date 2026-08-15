#!/usr/bin/env python3
"""Evaluate one preregistered V2 model and emit validated run evidence."""

from __future__ import annotations
import argparse
import contextlib
import hashlib
import io
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from mmengine.config import Config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiment_db.import_run_evidence import import_record
from tools.experiment_db.validate_run_evidence import validate
from tools.experiments.registry import (
    DB,
    register_eval_plan,
    set_train_run_status,
    sha256_file,
)

METRIC_DEFINITION = 'COCO bbox mAP@[.50:.95]'
MAX_DETS = [1, 10, 100]
METRIC_NAMES = ('mAP', 'AP50', 'AP75', 'AP_S', 'AP_M', 'AP_L')
FRAMEWORK_KEYS = {
    'mAP': 'bbox_mAP',
    'AP50': 'bbox_mAP_50',
    'AP75': 'bbox_mAP_75',
    'AP_S': 'bbox_mAP_s',
    'AP_M': 'bbox_mAP_m',
    'AP_L': 'bbox_mAP_l',
}


def stable_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'), default=str
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def project_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'expected project-relative path: {relative}')
    return ROOT / path


def artifact(path: Path, relative: str | None = None) -> dict:
    if relative is None:
        relative = path.relative_to(ROOT).as_posix()
    return {'path': relative, 'sha256': sha256_file(path)}


def git_state() -> dict:
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ['git', 'status', '--porcelain=v1'],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not status:
        return {'git_commit': commit, 'dirty': False}
    diff = (
        subprocess.run(
            ['git', 'diff', '--binary', 'HEAD'],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        + b'\n--STATUS--\n'
        + status.encode()
    )
    return {
        'git_commit': commit,
        'dirty': True,
        'diff_sha256': hashlib.sha256(diff).hexdigest(),
    }


def load_overrides(raw: str | None) -> dict:
    if raw is None:
        return {}
    payload = (
        Path(raw[1:]).read_text(encoding='utf-8')
        if raw.startswith('@')
        else raw
    )
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError('--override-json must decode to an object')
    return value


def set_dotted(config, dotted_key: str, value) -> None:
    current = config
    parts = dotted_key.split('.')
    for part in parts[:-1]:
        if part not in current:
            raise KeyError(f'override path does not exist: {dotted_key}')
        current = current[part]
    if parts[-1] not in current:
        raise KeyError(f'override target does not exist: {dotted_key}')
    current[parts[-1]] = value


def bbox_head(config: Config):
    model = config.get('model', {})
    return model.get('bbox_head', {}) if isinstance(model, dict) else {}


def protocol_fields(config: Config) -> dict:
    model = config.get('model', {})
    head = bbox_head(config)
    candidate_count = int(
        head.get('num_proposals', model.get('num_queries', 100))
    )
    solver = str(head.get('solver_type', 'not_applicable'))
    steps = int(head.get('sampling_timesteps', 1))
    if solver == 'heun':
        nfe = max(1, 2 * steps - 1)
    else:
        nfe = max(1, steps)
    return {
        'metric_definition': METRIC_DEFINITION,
        'max_dets': MAX_DETS,
        'eval_code_version': sha256_file(Path(__file__)),
        'candidate_count': candidate_count,
        'solver': solver,
        'steps': max(1, steps),
        'nfe': nfe,
    }


def annotation_info(config: Config, split: str) -> tuple[Path, str, int]:
    evaluator = (
        config.val_evaluator if split == 'val' else config.test_evaluator
    )
    if isinstance(evaluator, list):
        evaluator = next(
            item for item in evaluator if item.get('type') == 'CocoMetric'
        )
    relative = evaluator['ann_file']
    path = project_path(relative)
    document = json.loads(path.read_text(encoding='utf-8'))
    return path, relative, len(document['images'])


def independent_coco(annotation: Path, prediction: Path) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(annotation))
        predictions = json.loads(prediction.read_text(encoding='utf-8'))
        if not predictions:
            raise RuntimeError('prediction JSON is empty')
        coco_dt = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_dt, 'bbox')
        evaluator.params.maxDets = MAX_DETS
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        name: float(evaluator.stats[index])
        for index, name in enumerate(METRIC_NAMES)
    }


def framework_metrics(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding='utf-8'))['metrics']
    output = {}
    for name, suffix in FRAMEWORK_KEYS.items():
        matches = [value for key, value in raw.items() if key.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(
                f'framework metric {suffix}: found {len(matches)}'
            )
        output[name] = float(matches[0])
    return output


def fetch_run(train_run_id: str) -> dict:
    connection = sqlite3.connect(DB)
    connection.row_factory = sqlite3.Row
    try:
        train = connection.execute(
            'SELECT * FROM train_run_registry WHERE train_run_id=?',
            (train_run_id,),
        ).fetchone()
        if train is None:
            raise ValueError(f'unknown train_run_id: {train_run_id}')
        selected = connection.execute(
            'SELECT * FROM selected_checkpoint WHERE train_run_id=?',
            (train_run_id,),
        ).fetchone()
        if selected is None:
            raise ValueError(f'no selected checkpoint: {train_run_id}')
        parent_sha = None
        if train['parent_train_run_id']:
            parent = connection.execute(
                'SELECT checkpoint_sha256 FROM selected_checkpoint '
                'WHERE train_run_id=?',
                (train['parent_train_run_id'],),
            ).fetchone()
            if parent is None:
                raise ValueError(
                    'paired child has no registered parent checkpoint'
                )
            parent_sha = parent[0]
        return {
            'train': dict(train),
            'selected': dict(selected),
            'parent_checkpoint_sha256': parent_sha,
        }
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-run-id', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--inference-seed', type=int, default=42)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--override-json')
    parser.add_argument(
        '--output-root', type=Path, default=ROOT / 'results/v2/evaluations'
    )
    parser.add_argument(
        '--status',
        choices=('test_evaluated', 'verified', 'paper_eligible'),
        default='test_evaluated',
    )
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--no-import', action='store_true')
    args = parser.parse_args()

    run = fetch_run(args.train_run_id)
    train, selected = run['train'], run['selected']
    config_path = project_path(train['config_path'])
    checkpoint_path = project_path(selected['checkpoint_path'])
    if sha256_file(config_path) != train['config_sha256']:
        raise RuntimeError('registered config SHA mismatch')
    if sha256_file(checkpoint_path) != selected['checkpoint_sha256']:
        raise RuntimeError('registered checkpoint SHA mismatch')
    config = Config.fromfile(config_path)
    overrides = load_overrides(args.override_json)
    for key, value in sorted(overrides.items()):
        set_dotted(config, key, value)
    annotation_path, annotation_relative, num_images = annotation_info(
        config, args.split
    )
    expected_annotation_sha = config.get(f'{args.split}_annotation_sha256')
    annotation_sha = sha256_file(annotation_path)
    if annotation_sha != expected_annotation_sha:
        raise RuntimeError('resolved config annotation SHA mismatch')
    code = git_state()
    if args.status == 'paper_eligible' and code['dirty']:
        raise RuntimeError(
            'paper-eligible evaluation requires a clean worktree'
        )
    protocol = protocol_fields(config)
    protocol_payload = {
        'train_run_id': args.train_run_id,
        'config_sha256': train['config_sha256'],
        'checkpoint_sha256': selected['checkpoint_sha256'],
        'annotation_sha256': annotation_sha,
        'split': args.split,
        'inference_seed': args.inference_seed,
        'overrides': overrides,
        'evaluation_protocol': protocol,
        'git_commit': code['git_commit'],
    }
    protocol_sha = sha256_text(stable_json(protocol_payload))
    eval_run_id = (
        f'{args.train_run_id}__{args.split}__inferseed-{args.inference_seed}'
        f'__{protocol_sha[:12]}'
    )
    output_dir = (
        args.output_root
        / args.train_run_id
        / args.split
        / f'inferseed_{args.inference_seed}'
        / protocol_sha[:12]
    )
    plan = {
        'eval_run_id': eval_run_id,
        'train_run_id': args.train_run_id,
        'dataset_id': train['dataset_id'],
        'config': train['config_path'],
        'checkpoint': selected['checkpoint_path'],
        'split': args.split,
        'num_images': num_images,
        'inference_seed': args.inference_seed,
        'overrides': overrides,
        'protocol_sha256': protocol_sha,
        'output_dir': output_dir.relative_to(ROOT).as_posix(),
    }
    if args.plan:
        print(json.dumps({'status': 'PLANNED', **plan}, indent=2))
        return 0

    evidence_path = output_dir / 'run_evidence.json'
    if evidence_path.is_file():
        record = json.loads(evidence_path.read_text(encoding='utf-8'))
        errors = validate(record, ROOT, verify_files=True)
        if errors:
            raise RuntimeError(
                'existing evidence is invalid: ' + '; '.join(errors)
            )
        if not args.no_import:
            connection = sqlite3.connect(DB)
            connection.execute('PRAGMA foreign_keys=ON')
            try:
                with connection:
                    import_record(connection, record, evidence_path, ROOT)
            finally:
                connection.close()
        print(json.dumps({'status': 'REUSED', **plan}, indent=2))
        return 0
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f'partial evaluation directory requires review: {output_dir}'
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    register_eval_plan(
        eval_run_id=eval_run_id,
        train_run_id=args.train_run_id,
        split=args.split,
        inference_seed=args.inference_seed,
        annotation_sha256=annotation_sha,
        protocol_name=METRIC_DEFINITION,
        protocol_sha256=protocol_sha,
        selection_source=selected['selection_source'],
        test_tuned=bool(config.experiment.get('test_tuned', False)),
        status='running',
    )
    metrics_path = output_dir / 'framework_metrics.json'
    prediction_prefix = output_dir / 'predictions'
    log_path = output_dir / 'evaluation.log'
    command = [
        sys.executable,
        str(ROOT / 'experiments/runners/test.py'),
        str(config_path),
        '--checkpoint',
        str(checkpoint_path),
        '--dataset',
        args.split,
        '--gpu-id',
        str(args.gpu_id),
        '--seed',
        str(args.inference_seed),
        '--prediction-prefix',
        str(prediction_prefix),
        '--output-json',
        str(metrics_path),
        '--exp-name',
        eval_run_id,
    ]
    if overrides:
        command.extend(['--override-json', stable_json(overrides)])
    with log_path.open('w', encoding='utf-8') as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        register_eval_plan(
            eval_run_id=eval_run_id,
            train_run_id=args.train_run_id,
            split=args.split,
            inference_seed=args.inference_seed,
            annotation_sha256=annotation_sha,
            protocol_name=METRIC_DEFINITION,
            protocol_sha256=protocol_sha,
            selection_source=selected['selection_source'],
            test_tuned=bool(config.experiment.get('test_tuned', False)),
            status='failed',
        )
        raise RuntimeError(f'evaluation failed; inspect {log_path}')

    prediction_path = Path(str(prediction_prefix) + '.bbox.json')
    independent = independent_coco(annotation_path, prediction_path)
    framework = framework_metrics(metrics_path)
    differences = {
        key: abs(independent[key] - framework[key]) for key in METRIC_NAMES
    }
    if max(differences.values()) > 5.1e-4:
        raise RuntimeError(
            f'framework/independent COCO mismatch: {differences}'
        )
    record = {
        'schema_version': '1.0',
        'record_id': eval_run_id,
        'record_type': 'evaluation',
        'dataset_id': train['dataset_id'],
        'split': args.split,
        'train_run_id': args.train_run_id,
        'annotation': artifact(annotation_path, annotation_relative),
        'num_images': num_images,
        'training_seed': train['training_seed'],
        'inference_seed': args.inference_seed,
        'replication_unit': train['replication_unit'],
        'parent_train_run_id': train['parent_train_run_id'],
        'parent_checkpoint_sha256': run['parent_checkpoint_sha256'],
        'config': artifact(config_path, train['config_path']),
        'checkpoint': artifact(checkpoint_path, selected['checkpoint_path']),
        'source_log': artifact(log_path),
        'code': code,
        'evaluation_protocol': {
            **protocol,
            'overrides': overrides,
            'prediction_sha256': sha256_file(prediction_path),
            'framework_metrics_sha256': sha256_file(metrics_path),
            'framework_independent_max_abs_error': max(differences.values()),
        },
        'protocol_sha256': protocol_sha,
        'selection_source': selected['selection_source'],
        'test_tuned': bool(config.experiment.get('test_tuned', False)),
        'status': args.status,
        'metrics': independent,
    }
    errors = validate(record, ROOT, verify_files=True)
    if errors:
        raise RuntimeError(
            'generated evidence failed validation: ' + '; '.join(errors)
        )
    temporary = evidence_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    temporary.replace(evidence_path)
    if not args.no_import:
        connection = sqlite3.connect(DB)
        connection.execute('PRAGMA foreign_keys=ON')
        try:
            with connection:
                import_record(connection, record, evidence_path, ROOT)
        finally:
            connection.close()
    set_train_run_status(args.train_run_id, 'test_evaluated')
    print(
        json.dumps(
            {'status': 'PASS', 'metrics': independent, **plan}, indent=2
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
