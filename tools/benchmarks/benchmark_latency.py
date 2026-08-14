#!/usr/bin/env python3
"""Benchmark one detector checkpoint under an explicit, reproducible protocol.

The timed boundary is ``model.test_step(batch)``. Data loading is excluded;
the same held-out batch is reused after warm-up. Accuracy is never inferred
from this benchmark and must be linked separately through run evidence.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
    ).strip()


def measure(model, batch, warmup: int, iterations: int, repeats: int):
    with torch.inference_mode():
        for _ in range(warmup):
            model.test_step(batch)
        torch.cuda.synchronize()
        repeat_ms = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                model.test_step(batch)
            end.record()
            torch.cuda.synchronize()
            repeat_ms.append(start.elapsed_time(end) / iterations)
    return repeat_ms


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the latency protocol')
    if min(args.warmup, args.iterations, args.repeats) < 1:
        raise ValueError('warmup, iterations, and repeats must be positive')

    config = Config.fromfile(args.config)
    config.work_dir = str(ROOT / '.build_tmp' / 'latency_benchmark')
    runner = Runner.from_cfg(config)
    model = runner.model.to(args.device).eval()
    load_checkpoint(
        model, str(args.checkpoint), map_location='cpu', strict=True
    )
    batch = next(iter(runner.test_dataloader))

    repeat_ms = measure(
        model, batch, args.warmup, args.iterations, args.repeats
    )
    mean_ms = statistics.mean(repeat_ms)
    result = {
        'schema_version': 1,
        'kind': 'hardware_efficiency_repeat',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'config': {
            'path': str(args.config),
            'sha256': sha256_file(args.config),
        },
        'checkpoint': {
            'path': str(args.checkpoint),
            'sha256': sha256_file(args.checkpoint),
        },
        'code': {'git_commit': git_commit()},
        'hardware': {
            'device': args.device,
            'gpu_model': torch.cuda.get_device_name(torch.device(args.device)),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
        },
        'protocol': {
            'timing_boundary': 'model.test_step on one preloaded held-out batch',
            'data_loading_included': False,
            'precision': 'configuration-defined',
            'warmup': args.warmup,
            'iterations_per_repeat': args.iterations,
            'repeats': args.repeats,
            'batch_size': len(batch.get('data_samples', [])),
        },
        'latency_ms_per_batch_repeats': repeat_ms,
        'latency_ms_per_batch_mean': mean_ms,
        'latency_ms_per_batch_sample_std': (
            statistics.stdev(repeat_ms) if len(repeat_ms) > 1 else 0.0
        ),
        'batches_per_second': 1000.0 / mean_ms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
