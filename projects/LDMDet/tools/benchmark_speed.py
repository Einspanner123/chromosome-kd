#!/usr/bin/env python3
"""Inference speed benchmark for LDMDet vs standard detectors.

Usage:
    python projects/LDMDet/tools/benchmark_speed.py \
        --config projects/LDMDet/tools/benchmark_speed_config.json \
        --output projects/LDMDet/results/benchmark_speed_results.json

    # Force rerun all models
    python projects/LDMDet/tools/benchmark_speed.py \
        --config ... --output ... --force-rerun
"""

from __future__ import annotations
import argparse
import glob as glob_mod
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mmengine.analysis import get_model_complexity_info
from mmengine.config import Config

from mmdet.apis import init_detector
from mmdet.registry import DATASETS


def parse_args():
    p = argparse.ArgumentParser(description='Inference speed benchmark')
    p.add_argument(
        '--config',
        type=str,
        default='projects/LDMDet/tools/benchmark_speed_config.json',
        help='Path to benchmark config JSON',
    )
    p.add_argument(
        '--output',
        type=str,
        default='projects/LDMDet/results/benchmark_speed_results.json',
        help='Path to output results JSON',
    )
    p.add_argument(
        '--force-rerun',
        action='store_true',
        help='Force rerun all models even if results exist',
    )
    p.add_argument(
        '--device',
        type=str,
        default=None,
        help='Override device from config (e.g. cuda:0)',
    )
    return p.parse_args()


def resolve_checkpoint(ckpt_pattern: str) -> str:
    if os.path.isfile(ckpt_pattern):
        return ckpt_pattern
    candidates = sorted(glob_mod.glob(ckpt_pattern))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f'Checkpoint not found: {ckpt_pattern}'
    )


def load_results(output_path: str) -> Dict[str, Any]:
    if os.path.isfile(output_path):
        with open(output_path) as f:
            return json.load(f)
    return {'meta': {}, 'results': []}


def save_results(data: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_completed_names(results_data: Dict[str, Any]) -> set:
    return {r['name'] for r in results_data.get('results', [])}


def build_dataset(cfg: Config) -> Tuple[Any, List]:
    ds_cfg = cfg.val_dataloader.dataset
    dataset = DATASETS.build(ds_cfg)
    samples = []
    for idx in range(len(dataset)):
        data = dataset[idx]
        samples.append(data)
    return dataset, samples


def measure_params(model: nn.Module) -> float:
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def measure_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> float:
    try:
        flops, _ = get_model_complexity_info(
            model,
            input_shape,
            show_table=False,
            show_arch=False,
        )
        if isinstance(flops, str):
            flops_val = float(flops.replace(' GFLOPs', '').replace(' MACs', ''))
        else:
            flops_val = float(flops) / 1e9
        return flops_val
    except Exception:
        return -1.0


def benchmark_e2e(
    model: nn.Module,
    samples: List,
    device: str,
    warmup_iters: int,
    bench_iters: int,
) -> Dict[str, float]:
    model.eval()
    latencies = []

    with torch.no_grad():
        for i in range(warmup_iters):
            data = samples[i % len(samples)]
            inputs = data['inputs'].to(device) if isinstance(data, dict) else data['inputs'].to(device)
            data_samples = data['data_samples'] if isinstance(data, dict) else data.data_samples
            if not isinstance(data_samples, list):
                data_samples = [data_samples]
            _ = model.test_step({'inputs': inputs.unsqueeze(0), 'data_samples': data_samples})

        torch.cuda.synchronize()
        for i in range(bench_iters):
            data = samples[i % len(samples)]
            inputs = data['inputs'].to(device) if isinstance(data, dict) else data['inputs'].to(device)
            data_samples = data['data_samples'] if isinstance(data, dict) else data.data_samples
            if not isinstance(data_samples, list):
                data_samples = [data_samples]

            torch.cuda.synchronize()
            t0 = time.time()
            _ = model.test_step({'inputs': inputs.unsqueeze(0), 'data_samples': data_samples})
            torch.cuda.synchronize()
            t1 = time.time()
            latencies.append((t1 - t0) * 1000)

    lat = torch.tensor(latencies)
    mean_ms = lat.mean().item()
    std_ms = lat.std().item()
    min_ms = lat.min().item()
    max_ms = lat.max().item()
    fps = 1000.0 / mean_ms

    return {
        'fps': round(fps, 2),
        'latency_mean_ms': round(mean_ms, 2),
        'latency_std_ms': round(std_ms, 2),
        'latency_min_ms': round(min_ms, 2),
        'latency_max_ms': round(max_ms, 2),
    }


def benchmark_forward(
    model: nn.Module,
    samples: List,
    device: str,
    warmup_iters: int,
    bench_iters: int,
    is_ldmdet: bool = False,
) -> Dict[str, float]:
    model.eval()
    latencies = []

    with torch.no_grad():
        for i in range(warmup_iters):
            data = samples[i % len(samples)]
            inputs = data['inputs'].to(device) if isinstance(data, dict) else data['inputs'].to(device)
            batch_inputs = inputs.unsqueeze(0)
            features = model.extract_feat(batch_inputs)
            if is_ldmdet:
                from projects.LDMDet.mods.structures import ImageMeta
                img_metas = [
                    ImageMeta(
                        img_shape=(inputs.shape[-2], inputs.shape[-1]),
                        ori_shape=(inputs.shape[-2], inputs.shape[-1]),
                        scale_factor=(1.0, 1.0, 1.0, 1.0),
                    )
                ]
                _ = model.bbox_head.predict(features, img_metas, rescale=True)
            else:
                data_samples = data['data_samples'] if isinstance(data, dict) else data.data_samples
                if not isinstance(data_samples, list):
                    data_samples = [data_samples]
                img_metas = [ds.metainfo for ds in data_samples]
                _ = model.bbox_head.predict(features, img_metas, rescale=True)

        torch.cuda.synchronize()
        for i in range(bench_iters):
            data = samples[i % len(samples)]
            inputs = data['inputs'].to(device) if isinstance(data, dict) else data['inputs'].to(device)
            batch_inputs = inputs.unsqueeze(0)

            torch.cuda.synchronize()
            t0 = time.time()
            features = model.extract_feat(batch_inputs)
            if is_ldmdet:
                from projects.LDMDet.mods.structures import ImageMeta
                img_metas = [
                    ImageMeta(
                        img_shape=(inputs.shape[-2], inputs.shape[-1]),
                        ori_shape=(inputs.shape[-2], inputs.shape[-1]),
                        scale_factor=(1.0, 1.0, 1.0, 1.0),
                    )
                ]
                _ = model.bbox_head.predict(features, img_metas, rescale=True)
            else:
                data_samples = data['data_samples'] if isinstance(data, dict) else data.data_samples
                if not isinstance(data_samples, list):
                    data_samples = [data_samples]
                img_metas = [ds.metainfo for ds in data_samples]
                _ = model.bbox_head.predict(features, img_metas, rescale=True)
            torch.cuda.synchronize()
            t1 = time.time()
            latencies.append((t1 - t0) * 1000)

    lat = torch.tensor(latencies)
    mean_ms = lat.mean().item()
    std_ms = lat.std().item()
    min_ms = lat.min().item()
    max_ms = lat.max().item()
    fps = 1000.0 / mean_ms

    return {
        'fps': round(fps, 2),
        'latency_mean_ms': round(mean_ms, 2),
        'latency_std_ms': round(std_ms, 2),
        'latency_min_ms': round(min_ms, 2),
        'latency_max_ms': round(max_ms, 2),
    }


def measure_peak_memory(
    model: nn.Module,
    samples: List,
    device: str,
    warmup_iters: int,
    is_ldmdet: bool = False,
) -> float:
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    with torch.no_grad():
        data = samples[0]
        inputs = data['inputs'].to(device) if isinstance(data, dict) else data['inputs'].to(device)
        data_samples = data['data_samples'] if isinstance(data, dict) else data.data_samples
        if not isinstance(data_samples, list):
            data_samples = [data_samples]
        for _ in range(warmup_iters):
            _ = model.test_step({'inputs': inputs.unsqueeze(0), 'data_samples': data_samples})

    peak_bytes = torch.cuda.max_memory_allocated(device)
    return round(peak_bytes / 1024 / 1024, 2)


def run_single_model(
    model_cfg: Dict[str, Any],
    samples: List,
    dataset_len: int,
    device: str,
    warmup_iters: int,
    bench_iters: int,
    force_rerun: bool,
    results_data: Dict[str, Any],
    output_path: str,
) -> List[Dict[str, Any]]:
    model_type = model_cfg['type']
    is_ldmdet = model_type == 'ldmdet'
    config_path = model_cfg['config']
    ckpt_path = resolve_checkpoint(model_cfg['checkpoint'])

    cfg = Config.fromfile(config_path)

    print(f'\n{"="*60}')
    print(f'Loading: {model_cfg["name"]}')
    print(f'  Config:     {config_path}')
    print(f'  Checkpoint: {ckpt_path}')
    print(f'  Type:       {model_type}')

    model = init_detector(cfg, ckpt_path, device=device)
    model.eval()

    params_m = measure_params(model)

    sample_data = samples[0]
    sample_input = sample_data['inputs']
    if isinstance(sample_input, torch.Tensor):
        c, h, w = sample_input.shape
    else:
        c, h, w = 3, 800, 1333
    flops_g = measure_flops(model, (1, c, h, w))

    sampling_steps_list = model_cfg.get('sampling_steps', [None])
    if is_ldmdet and sampling_steps_list == [None]:
        sampling_steps_list = [1, 2, 4, 8]

    original_sampling_timesteps = None
    if is_ldmdet and hasattr(model, 'bbox_head'):
        original_sampling_timesteps = model.bbox_head.sampling_timesteps

    new_results = []

    for steps in sampling_steps_list:
        if is_ldmdet and steps is not None:
            name = f'{model_cfg["name"]}-{steps}step'
            model.bbox_head.sampling_timesteps = steps
        else:
            name = model_cfg['name']

        completed = get_completed_names(results_data)
        if not force_rerun and name in completed:
            print(f'  [{name}] Already completed, skipping.')
            continue

        print(f'\n  Benchmarking: {name}')
        print(f'    Params: {params_m:.2f}M | FLOPs: {flops_g:.2f}G')

        peak_mem = measure_peak_memory(model, samples, device, warmup_iters=min(5, len(samples)), is_ldmdet=is_ldmdet)
        print(f'    Peak Memory: {peak_mem:.1f} MB')

        print(f'    Running E2E benchmark (warmup={warmup_iters}, bench={bench_iters})...')
        e2e_result = benchmark_e2e(model, samples, device, warmup_iters, bench_iters)
        print(f'    E2E: {e2e_result["fps"]:.1f} FPS | {e2e_result["latency_mean_ms"]:.1f} ± {e2e_result["latency_std_ms"]:.1f} ms')

        print('    Running Forward-only benchmark...')
        fwd_result = benchmark_forward(model, samples, device, warmup_iters, bench_iters, is_ldmdet=is_ldmdet)
        print(f'    Forward: {fwd_result["fps"]:.1f} FPS | {fwd_result["latency_mean_ms"]:.1f} ± {fwd_result["latency_std_ms"]:.1f} ms')

        result_entry = {
            'name': name,
            'type': model_type,
            'config': config_path,
            'checkpoint': ckpt_path,
            'e2e': e2e_result,
            'forward': fwd_result,
            'params_M': round(params_m, 2),
            'flops_G': round(flops_g, 2),
            'peak_mem_MB': peak_mem,
        }
        if is_ldmdet and steps is not None:
            result_entry['sampling_steps'] = steps

        new_results.append(result_entry)

        results_data['results'].append(result_entry)
        save_results(results_data, output_path)
        print('    Result saved.')

    if is_ldmdet and original_sampling_timesteps is not None:
        model.bbox_head.sampling_timesteps = original_sampling_timesteps

    del model
    torch.cuda.empty_cache()

    return new_results


def print_summary(results_data: Dict[str, Any]):
    results = results_data.get('results', [])
    if not results:
        print('No results to summarize.')
        return

    header = f'{"Model":<35} {"E2E FPS":>10} {"E2E ms":>10} {"Fwd FPS":>10} {"Fwd ms":>10} {"Params(M)":>10} {"FLOPs(G)":>10} {"Mem(MB)":>10}'
    sep = '-' * len(header)
    print(f'\n{"="*60}')
    print('BENCHMARK SUMMARY')
    print(sep)
    print(header)
    print(sep)
    for r in results:
        name = r['name'][:34]
        e2e_fps = r['e2e']['fps']
        e2e_ms = r['e2e']['latency_mean_ms']
        fwd_fps = r['forward']['fps']
        fwd_ms = r['forward']['latency_mean_ms']
        params = r['params_M']
        flops = r['flops_G']
        mem = r['peak_mem_MB']
        print(f'{name:<35} {e2e_fps:>10.1f} {e2e_ms:>10.1f} {fwd_fps:>10.1f} {fwd_ms:>10.1f} {params:>10.2f} {flops:>10.2f} {mem:>10.1f}')
    print(sep)


def main():
    args = parse_args()

    with open(args.config) as f:
        bench_cfg = json.load(f)

    device = args.device or bench_cfg.get('device', 'cuda:0')
    warmup_iters = bench_cfg.get('warmup_iters', 50)
    bench_iters = bench_cfg.get('bench_iters', 300)

    results_data = load_results(args.output)

    results_data['meta'] = {
        'date': datetime.now().isoformat(),
        'device': device,
        'gpu_name': torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'N/A',
        'dataset': bench_cfg.get('dataset', 'unknown'),
        'warmup_iters': warmup_iters,
        'bench_iters': bench_iters,
    }
    save_results(results_data, args.output)

    first_cfg = Config.fromfile(bench_cfg['models'][0]['config'])
    dataset, samples = build_dataset(first_cfg)
    print(f'Dataset: {len(dataset)} images')
    print(f'Device:  {device} ({results_data["meta"]["gpu_name"]})')
    print(f'Warmup:  {warmup_iters} iters | Bench: {bench_iters} iters')

    for model_cfg in bench_cfg['models']:
        run_single_model(
            model_cfg=model_cfg,
            samples=samples,
            dataset_len=len(dataset),
            device=device,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            force_rerun=args.force_rerun,
            results_data=results_data,
            output_path=args.output,
        )

    print_summary(results_data)
    print(f'\nResults saved to: {args.output}')


if __name__ == '__main__':
    main()
