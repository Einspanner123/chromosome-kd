"""
LDMDet 推理速度 Benchmark

测量不同模型、不同采样步数下的 FPS 和 mAP。

用法:
    python projects/LDMDet/benchmark_speed.py \
        --config work_dirs/ldmdet_flowdet_adaln/ldmdet_flowdet_adaln.py \
        --checkpoint work_dirs/ldmdet_flowdet_adaln/best_coco_bbox_mAP_epoch_79.pth \
        --sampling-steps 1 2 4 8 \
        --warmup 10 --runs 100

    # 批量测试多个模型
    python projects/LDMDet/benchmark_speed.py --batch-all
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import Runner


def build_model(config_path, checkpoint_path, device="cuda:0"):
    from mmdet.apis import init_detector

    cfg = Config.fromfile(config_path)
    model = init_detector(cfg, checkpoint_path, device=device)
    model.eval()
    return model


def benchmark_fps(model, input_shape=(3, 1024, 1024), warmup=10, runs=100):
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    dummy_img = torch.randn(1, *input_shape, device=device)

    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData

    data_sample = DetDataSample()
    data_sample.set_metainfo(
        {
            "img_shape": input_shape[1:],
            "ori_shape": input_shape[1:],
            "scale_factor": (1.0, 1.0, 1.0, 1.0),
        }
    )
    data_sample.gt_instances = InstanceData()
    data_sample.gt_instances.bboxes = torch.zeros(0, 4, device=device)
    data_sample.gt_instances.labels = torch.zeros(0, dtype=torch.long, device=device)

    for _ in range(warmup):
        with torch.no_grad():
            model.predict(dummy_img, [data_sample])
    if use_cuda:
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        with torch.no_grad():
            model.predict(dummy_img, [data_sample])
    if use_cuda:
        torch.cuda.synchronize()

    elapsed = time.time() - start
    fps = runs / elapsed
    latency_ms = elapsed / runs * 1000

    return fps, latency_ms


def benchmark_with_steps(config_path, checkpoint_path, steps_list, device="cuda:0",
                         input_shape=(3, 1024, 1024), warmup=10, runs=100):
    results = []
    for steps in steps_list:
        cfg = Config.fromfile(config_path)
        cfg.model.bbox_head.sampling_timesteps = steps

        from mmdet.apis import init_detector

        model = init_detector(cfg, checkpoint_path, device=device)
        model.eval()

        fps, latency = benchmark_fps(model, input_shape, warmup, runs)
        result = {
            "config": os.path.basename(config_path),
            "sampling_steps": steps,
            "fps": round(fps, 1),
            "latency_ms": round(latency, 1),
            "device": str(device),
        }
        results.append(result)
        print(f"  Steps={steps}: FPS={fps:.1f}, Latency={latency:.1f}ms")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def evaluate_map(config_path, checkpoint_path, sampling_steps=None, device="cuda:0"):
    cfg = Config.fromfile(config_path)
    if sampling_steps is not None:
        cfg.model.bbox_head.sampling_timesteps = sampling_steps

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    return metrics


BATCH_CONFIGS = [
    {
        "name": "ldmdet_baseline (DDPM)",
        "config": "work_dirs/ldmdet_baseline/ldmdet_baseline.py",
        "checkpoint": None,
        "steps": [1],
    },
    {
        "name": "ldmdet_rf (Rectified Flow)",
        "config": "work_dirs/ldmdet_rf/ldmdet_rf.py",
        "checkpoint": None,
        "steps": [1, 2, 4, 8],
    },
    {
        "name": "ldmdet_rf_shifted_schdule",
        "config": "work_dirs/ldmdet_rf_shifted_schdule/ldmdet_rf_shifted_schdule.py",
        "checkpoint": None,
        "steps": [1, 2, 4, 8],
    },
    {
        "name": "ldmdet_flowdet_adaln (Best)",
        "config": "work_dirs/ldmdet_flowdet_adaln/ldmdet_flowdet_adaln.py",
        "checkpoint": None,
        "steps": [1, 2, 4, 8],
    },
    {
        "name": "ldmdet_flowdet_adaln_ot",
        "config": "work_dirs/ldmdet_flowdet_adaln_ot/ldmdet_flowdet_adaln_ot.py",
        "checkpoint": None,
        "steps": [1, 2, 4, 8],
    },
    {
        "name": "ldmdet_flowdet_sinkhorn",
        "config": "work_dirs/ldmdet_flowdet_sinkhorn/ldmdet_flowdet_sinkhorn.py",
        "checkpoint": None,
        "steps": [1, 2, 4, 8],
    },
]


def find_best_checkpoint(work_dir):
    import glob

    best_ckpts = sorted(glob.glob(os.path.join(work_dir, "best_coco_bbox_mAP_epoch_*.pth")))
    if best_ckpts:
        return best_ckpts[-1]
    last_ckpt_path = os.path.join(work_dir, "last_checkpoint")
    if os.path.exists(last_ckpt_path):
        with open(last_ckpt_path) as f:
            return f.read().strip()
    return None


def main():
    parser = argparse.ArgumentParser(description="LDMDet Speed Benchmark")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument(
        "--sampling-steps", type=int, nargs="+", default=[1, 2, 4, 8],
        help="Sampling steps to benchmark",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Benchmark iterations")
    parser.add_argument(
        "--input-shape", type=int, nargs=3, default=[3, 1024, 1024],
        help="Input shape (C, H, W)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--eval-map", action="store_true", help="Also evaluate mAP for each step count",
    )
    parser.add_argument(
        "--batch-all", action="store_true",
        help="Benchmark all models defined in BATCH_CONFIGS",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.json",
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[INFO] CUDA not available, falling back to CPU")
        args.device = "cpu"

    all_results = []

    if args.batch_all:
        for cfg_info in BATCH_CONFIGS:
            config_path = cfg_info["config"]
            work_dir = os.path.dirname(config_path)

            if not os.path.exists(config_path):
                print(f"[SKIP] Config not found: {config_path}")
                continue

            checkpoint = cfg_info.get("checkpoint") or find_best_checkpoint(work_dir)
            if not checkpoint or not os.path.exists(checkpoint):
                print(f"[SKIP] Checkpoint not found for: {config_path}")
                continue

            print(f"\n{'='*60}")
            print(f"Benchmarking: {cfg_info['name']}")
            print(f"  Config: {config_path}")
            print(f"  Checkpoint: {checkpoint}")
            print(f"{'='*60}")

            results = benchmark_with_steps(
                config_path, checkpoint, cfg_info["steps"],
                device=args.device, input_shape=tuple(args.input_shape),
                warmup=args.warmup, runs=args.runs,
            )
            for r in results:
                r["model_name"] = cfg_info["name"]
            all_results.extend(results)

    elif args.config:
        if not args.checkpoint:
            work_dir = os.path.dirname(args.config)
            args.checkpoint = find_best_checkpoint(work_dir)
        if not args.checkpoint:
            raise FileNotFoundError(f"No checkpoint found for {args.config}")

        print(f"\nBenchmarking: {args.config}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Steps: {args.sampling_steps}")

        results = benchmark_with_steps(
            args.config, args.checkpoint, args.sampling_steps,
            device=args.device, input_shape=tuple(args.input_shape),
            warmup=args.warmup, runs=args.runs,
        )
        all_results.extend(results)

        if args.eval_map:
            print("\nEvaluating mAP for each step count...")
            for steps in args.sampling_steps:
                print(f"  Evaluating steps={steps}...")
                metrics = evaluate_map(args.config, args.checkpoint, steps, args.device)
                print(f"  Steps={steps}: {metrics}")

    else:
        parser.print_help()
        return

    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"{'Model':<40} {'Steps':>5} {'FPS':>8} {'Latency(ms)':>12}")
    print("-" * 70)
    for r in all_results:
        name = r.get("model_name", r["config"])
        print(f"{name:<40} {r['sampling_steps']:>5} {r['fps']:>8.1f} {r['latency_ms']:>12.1f}")

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output,
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
