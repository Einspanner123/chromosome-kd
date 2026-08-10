"""Wait for one clean baseline process and launch its paired LQCR training.

This queue runner makes the hand-off auditable: it refuses crashed baselines,
selects the retained best-mAP checkpoint, records its SHA256, and launches the
strict final-only quality-head config on the same physical GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / 'experiments' / 'runners' / 'train.py'
CONFIG = (
    REPO / 'experiments' / 'configs' / 'ldmdet' / 'directions' / 'capr'
    / 'capr_quality_final_only_paired_clean_chr2024.py'
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--baseline-work-dir', type=Path, required=True)
    parser.add_argument('--lqcr-work-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True, choices=(42, 123, 789))
    parser.add_argument('--gpu-id', type=int, required=True)
    parser.add_argument('--poll-seconds', type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dir = (REPO / args.baseline_work_dir).resolve()
    lqcr_dir = (REPO / args.lqcr_work_dir).resolve()
    lqcr_dir.mkdir(parents=True, exist_ok=True)
    queue_log = lqcr_dir / 'queue.log'

    with queue_log.open('a') as stream:
        stream.write(f'waiting for pid={args.wait_pid} seed={args.seed} gpu={args.gpu_id}\n')
        stream.flush()
        while pid_alive(args.wait_pid):
            time.sleep(args.poll_seconds)

        # Allow dataloader workers and CUDA context to terminate.
        time.sleep(30)
        baseline_log = baseline_dir / 'launch.log'
        if not baseline_log.exists():
            raise RuntimeError(f'missing baseline log: {baseline_log}')
        tail = baseline_log.read_text(errors='replace')[-200_000:]
        if 'Traceback (most recent call last)' in tail:
            raise RuntimeError('baseline ended with a traceback; refusing automatic LQCR launch')

        checkpoints = sorted(
            baseline_dir.glob('best_coco_bbox_mAP_epoch_*.pth'),
            key=lambda path: path.stat().st_mtime,
        )
        if not checkpoints:
            raise RuntimeError(f'no best-mAP checkpoint found under {baseline_dir}')
        checkpoint = checkpoints[-1].resolve()
        lineage = {
            'status': 'launching',
            'seed': args.seed,
            'validation_seed': 42,
            'gpu_id': args.gpu_id,
            'baseline_work_dir': str(baseline_dir),
            'baseline_checkpoint': str(checkpoint),
            'baseline_checkpoint_sha256': sha256(checkpoint),
            'intervention': 'quality_head_training_final_only_ranking',
            'same_checkpoint_required_for_baseline_eval': True,
        }
        lineage_path = lqcr_dir / 'lineage.json'
        atomic_json(lineage_path, lineage)
        stream.write(f'launching from {checkpoint}\n')
        stream.flush()

    env = os.environ.copy()
    env['LQCR_BASE_CHECKPOINT'] = str(checkpoint)
    command = [
        sys.executable, '-u', str(TRAIN), str(CONFIG),
        '--work-dir', str(lqcr_dir), '--seed', str(args.seed),
        '--val-seed', '42', '--gpu-id', str(args.gpu_id),
    ]
    with (lqcr_dir / 'launch.log').open('w') as output:
        completed = subprocess.run(
            command, cwd=REPO, env=env, stdout=output,
            stderr=subprocess.STDOUT, check=False,
        )
    lineage['status'] = 'complete' if completed.returncode == 0 else 'failed'
    lineage['returncode'] = completed.returncode
    produced = sorted(lqcr_dir.glob('best_coco_bbox_mAP_epoch_*.pth'))
    if produced:
        best = max(produced, key=lambda path: path.stat().st_mtime)
        lineage['lqcr_checkpoint'] = str(best.resolve())
        lineage['lqcr_checkpoint_sha256'] = sha256(best)
    atomic_json(lqcr_dir / 'lineage.json', lineage)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
