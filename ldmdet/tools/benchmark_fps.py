"""LDMDet 端到端 FPS Benchmark

测量完整推理链路 (backbone + neck + head) 的延迟和 FPS, 支持横向对比:
  - LDMDet 变体: A0/A1/A3/A4/IO3-K200/IO3-K300
  - mmdet baselines: RTMDet-L, DINO 等

测试方式:
  - 使用 mmengine Config 构建完整模型
  - 加载 checkpoint
  - 合成 512x512 输入 (batch_size=1)
  - CUDA Event 精确计时
  - warmup 10 + iters 100

用法:
    # 测量所有 LDMDet 变体
    python ldmdet/tools/benchmark_fps.py --models a0 a1 a3 a4 io3_k200 io3_k300

    # 测量 baselines
    python ldmdet/tools/benchmark_fps.py --models rtmdet_l

    # 测量全部
    python ldmdet/tools/benchmark_fps.py --models all

    # 自定义参数
    python ldmdet/tools/benchmark_fps.py --models a3 a4 --img-size 512 --iters 100 --gpu 0

输出:
    - 终端打印对比表
    - results/benchmark_fps_{timestamp}.md 持久化记录
"""

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData


# ============================================================
# 模型注册表: (name, config_path, checkpoint_path, description)
# ============================================================
MODEL_REGISTRY = {
    # LDMDet 主路线消融 (A0 checkpoint 不存在,已移除)
    'a1': {
        'config': 'experiments/configs/ldmdet/directions/mainline_ablation_24obj/a1_rf_heun_24obj.py',
        'checkpoint': 'work_dirs/ldmdet_rf_heun_shifted_bs8_aug_v2/epoch_77.pth',
        'desc': 'A1: RF + Heun (4 steps)',
        'type': 'ldmdet',
    },
    'a3': {
        'config': 'experiments/configs/ldmdet/directions/mainline_ablation_24obj/a3_full_sota_24obj.py',
        'checkpoint': 'work_dirs/ablation/adaln_stochot_eps5/epoch_120.pth',
        'desc': 'A3: RF+Heun+AdaLN+StochOT (SOTA, Heun 4 steps)',
        'type': 'ldmdet',
    },
    'a4': {
        'config': 'experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py',
        'checkpoint': 'work_dirs/a4_dpm_pp_24obj/epoch_147.pth',
        'desc': 'A4: DPM-Solver++ (4 steps)',
        'type': 'ldmdet',
    },
    'io3_k300': {
        'config': 'experiments/configs/ldmdet/directions/inference_opt/io3_eval_k300_24obj.py',
        'checkpoint': 'work_dirs/ablation/adaln_stochot_eps5/epoch_120.pth',
        'desc': 'IO3: A3 + Top-K pruning K=300',
        'type': 'ldmdet',
    },
    'io3_k200': {
        'config': 'experiments/configs/ldmdet/directions/inference_opt/io3_eval_k200_24obj.py',
        'checkpoint': 'work_dirs/ablation/adaln_stochot_eps5/epoch_120.pth',
        'desc': 'IO3: A3 + Top-K pruning K=200',
        'type': 'ldmdet',
    },
    # A4 + IO3 组合 (DPM-Solver++ + Top-K 剪枝, 使用 A4 checkpoint)
    'a4_io3_k300': {
        'config': 'experiments/configs/ldmdet/directions/inference_opt/a4_io3_eval_k300_24obj.py',
        'checkpoint': 'work_dirs/a4_dpm_pp_24obj/epoch_147.pth',
        'desc': 'A4+IO3: DPM-Solver++ + Top-K pruning K=300',
        'type': 'ldmdet',
    },
    'a4_io3_k200': {
        'config': 'experiments/configs/ldmdet/directions/inference_opt/a4_io3_eval_k200_24obj.py',
        'checkpoint': 'work_dirs/a4_dpm_pp_24obj/epoch_147.pth',
        'desc': 'A4+IO3: DPM-Solver++ + Top-K pruning K=200',
        'type': 'ldmdet',
    },
    'a4_io3_k100': {
        'config': 'experiments/configs/ldmdet/directions/inference_opt/a4_io3_eval_k100_24obj.py',
        'checkpoint': 'work_dirs/a4_dpm_pp_24obj/epoch_147.pth',
        'desc': 'A4+IO3: DPM-Solver++ + Top-K pruning K=100 (aggressive)',
        'type': 'ldmdet',
    },
    # Baselines (mAP < A4=0.863, 用于论文 SOTA 对比)
    'cascade_rcnn': {
        'config': 'experiments/configs/baselines/benchmark_24obj/cascade_rcnn_r50.py',
        'checkpoint': 'work_dirs/baselines/cascade_rcnn_r50/best_coco_bbox_mAP_epoch_72.pth',
        'desc': 'Cascade R-CNN R50 (two-stage, mmdet)',
        'type': 'mmdet',
    },
    'yolox_s': {
        'config': 'experiments/configs/baselines/benchmark_24obj/yolox_s.py',
        'checkpoint': 'work_dirs/baselines/yolox_s/best_coco_bbox_mAP_epoch_200.pth',
        'desc': 'YOLOX-S (single-stage, mmdet)',
        'type': 'mmdet',
    },
    'diffusiondet': {
        'config': 'experiments/configs/baselines/benchmark_24obj/diffusiondet_ddpm.py',
        'checkpoint': 'work_dirs/baselines/diffusiondet_24obj/best_coco_bbox_mAP_epoch_26.pth',
        'desc': 'DiffusionDet (DDPM, euler 1-step, ldmdet framework)',
        'type': 'ldmdet',
    },
    # 以下 baseline mAP > A4=0.863, 论文中不纳入 SOTA 对比, 仅供参考
    'rtmdet_l': {
        'config': 'experiments/configs/baselines/benchmark_24obj/rtmdet_l.py',
        'checkpoint': 'work_dirs/baselines/rtmdet_l_24obj/epoch_86.pth',
        'desc': 'RTMDet-L (single-stage, mmdet) [mAP=0.869 > A4, excluded from paper]',
        'type': 'mmdet',
    },
}

# 已知 mAP (val set, 来自 SwanLab 验证 2026-07-15)
KNOWN_MAP = {
    # A0 (0.774) checkpoint 不存在,无法 benchmark
    'a1': 0.856,
    'a3': 0.858,
    'a4': 0.863,
    'io3_k300': 0.857,
    'io3_k200': 0.856,
    'a4_io3_k300': 0.861,  # 2026-07-14 评估
    'a4_io3_k200': 0.860,  # 2026-07-14 评估
    'a4_io3_k100': 0.850,  # 2026-07-14 评估
    # Baselines (SwanLab verified)
    'cascade_rcnn': 0.854,
    'yolox_s': 0.796,
    'diffusiondet': 0.787,
    'rtmdet_l': 0.869,  # mAP > A4, 论文中排除
}


def create_dummy_data_samples(bs: int, img_size: int) -> List[DetDataSample]:
    """创建 dummy DetDataSample (仅含 metainfo, 无需 GT)"""
    samples = []
    for _ in range(bs):
        ds = DetDataSample()
        ds.set_metainfo({
            'img_shape': (img_size, img_size, 3),
            'pad_shape': (img_size, img_size, 3),
            'ori_shape': (img_size, img_size, 3),
            'scale_factor': (1.0, 1.0),
            'img_id': 0,
        })
        samples.append(ds)
    return samples


def build_and_load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """从 mmengine Config 构建模型并加载 checkpoint

    注意: 这里使用 mmdet 的 MODELS.build 构建 detector, 但 detector 的
    data_preprocessor 字段会由 mmengine.BaseModel.__init__ 使用 mmengine 的
    MODELS registry 构建 (无法找到 mmdet 注册的 DetDataPreprocessor, 因为父
    registry 不会向子 registry 搜索)。解决方法是将 data_preprocessor 设为
    None, BaseModel 会自动用 BaseDataPreprocessor (mmengine 原生, 已注册)。
    benchmark 使用合成 tensor 输入, predict() 不会调用 data_preprocessor,
    因此不影响延迟测量。
    """
    cfg = Config.fromfile(os.path.join(PROJECT_ROOT, config_path))

    # 处理 custom_imports (注册 LDMDetDetector, PurePyTorch* 模块等)
    from mmengine.utils import import_modules_from_strings
    if 'custom_imports' in cfg:
        import_modules_from_strings(**cfg['custom_imports'])

    # 额外确保 mmdet 标准组件已注册
    import mmdet.models  # noqa: F401

    # 构建 model
    model_cfg = copy.deepcopy(cfg.model)
    # 移除 init_cfg (避免尝试加载预训练权重)
    if 'init_cfg' in model_cfg:
        model_cfg['init_cfg'] = None
    # 关闭 data_preprocessor: 避免 mmengine MODELS 找不到 DetDataPreprocessor
    # (DetDataPreprocessor 注册在 mmdet 子 registry, 父 registry 无法访问)。
    # benchmark 使用合成 tensor 输入, predict() 不调用 data_preprocessor,
    # 因此 BaseDataPreprocessor (mmengine 默认) 已足够。
    model_cfg['data_preprocessor'] = None

    model = MODELS.build(model_cfg)
    model = model.to(device)
    model.eval()

    # 加载 checkpoint
    if checkpoint_path:
        ckpt_full_path = os.path.join(PROJECT_ROOT, checkpoint_path)
    else:
        ckpt_full_path = ''
    if ckpt_full_path and os.path.isfile(ckpt_full_path):
        load_checkpoint(model, ckpt_full_path, map_location=device)
        print(f'  Checkpoint loaded: {checkpoint_path}')
    else:
        if checkpoint_path:
            print(f'  WARNING: Checkpoint not found: {ckpt_full_path}')
        else:
            print(f'  No checkpoint specified (random weights, latency still valid)')

    return model, cfg


def cuda_time(fn, warmup: int = 10, iters: int = 100) -> Tuple[float, float]:
    """测量 CUDA 函数平均执行时间, 返回 (mean_ms, std_ms)"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = np.array(times)
    return float(np.mean(times)), float(np.std(times))


def benchmark_ldmdet(model, batch_inputs: torch.Tensor, data_samples: List[DetDataSample],
                     warmup: int, iters: int) -> Dict:
    """测量 LDMDet 端到端延迟 (backbone + neck + head)"""
    def fn():
        with torch.no_grad():
            model.predict(batch_inputs, copy.deepcopy(data_samples), rescale=False)

    mean_ms, std_ms = cuda_time(fn, warmup, iters)

    # 额外测量 backbone+neck vs head 的分解
    def fn_backbone():
        with torch.no_grad():
            model.extract_feat(batch_inputs)

    backbone_ms, _ = cuda_time(fn_backbone, warmup, iters)
    head_ms = mean_ms - backbone_ms

    return {
        'total_ms': mean_ms,
        'std_ms': std_ms,
        'fps': 1000.0 / mean_ms if mean_ms > 0 else 0,
        'backbone_neck_ms': backbone_ms,
        'head_ms': head_ms,
    }


def benchmark_mmdet(model, batch_inputs: torch.Tensor, data_samples: List[DetDataSample],
                    warmup: int, iters: int) -> Dict:
    """测量 mmdet 标准 detector 端到端延迟"""
    def fn():
        with torch.no_grad():
            model.predict(batch_inputs, copy.deepcopy(data_samples))

    mean_ms, std_ms = cuda_time(fn, warmup, iters)

    def fn_backbone():
        with torch.no_grad():
            model.extract_feat(batch_inputs)

    backbone_ms, _ = cuda_time(fn_backbone, warmup, iters)
    head_ms = mean_ms - backbone_ms

    return {
        'total_ms': mean_ms,
        'std_ms': std_ms,
        'fps': 1000.0 / mean_ms if mean_ms > 0 else 0,
        'backbone_neck_ms': backbone_ms,
        'head_ms': head_ms,
    }


def run_benchmark(model_name: str, img_size: int, warmup: int, iters: int,
                  gpu: int) -> Optional[Dict]:
    """运行单个模型的 benchmark"""
    if model_name not in MODEL_REGISTRY:
        print(f'  Unknown model: {model_name}')
        return None

    info = MODEL_REGISTRY[model_name]
    config_path = info['config']
    checkpoint_path = info['checkpoint']
    model_type = info['type']
    desc = info['desc']

    print(f'\n{"=" * 80}')
    print(f'Benchmarking: {model_name} — {desc}')
    print(f'  Config: {config_path}')
    print(f'  Checkpoint: {checkpoint_path}')
    print(f'{"=" * 80}')

    device = torch.device(f'cuda:{gpu}')

    try:
        model, cfg = build_and_load_model(config_path, checkpoint_path, device)
    except Exception as e:
        print(f'  ERROR building model: {e}')
        import traceback
        traceback.print_exc()
        return None

    # 创建合成输入
    batch_inputs = torch.randn(1, 3, img_size, img_size, device=device)
    data_samples = create_dummy_data_samples(1, img_size)

    # 运行 benchmark
    try:
        if model_type == 'ldmdet':
            results = benchmark_ldmdet(model, batch_inputs, data_samples, warmup, iters)
        else:
            results = benchmark_mmdet(model, batch_inputs, data_samples, warmup, iters)
    except Exception as e:
        print(f'  ERROR during benchmark: {e}')
        import traceback
        traceback.print_exc()
        return None

    # 补充元信息
    results['name'] = model_name
    results['desc'] = desc
    results['config'] = config_path
    results['checkpoint'] = checkpoint_path
    results['img_size'] = img_size
    results['warmup'] = warmup
    results['iters'] = iters
    results['gpu'] = torch.cuda.get_device_name(device)
    results['known_mAP'] = KNOWN_MAP.get(model_name, None)

    # 获取采样器信息 (仅 LDMDet)
    if model_type == 'ldmdet' and hasattr(model, 'bbox_head'):
        head = model.bbox_head
        results['solver_type'] = getattr(head, 'solver_type', 'unknown')
        results['sampling_timesteps'] = getattr(head, 'sampling_timesteps', 'unknown')
        results['num_proposals'] = getattr(head, 'num_proposals', 'unknown')
        results['topk_pruning'] = getattr(head, 'topk_pruning_enabled', False)
        results['topk_k'] = getattr(head, 'topk_k', None)

    print(f'\n  Results:')
    print(f'    Total latency: {results["total_ms"]:.2f} ± {results["std_ms"]:.2f} ms')
    print(f'    FPS:           {results["fps"]:.1f}')
    print(f'    Backbone+Neck: {results["backbone_neck_ms"]:.2f} ms')
    print(f'    Head:          {results["head_ms"]:.2f} ms')
    if results.get('known_mAP') is not None:
        print(f'    Known mAP:     {results["known_mAP"]:.3f}')

    # 释放显存
    del model
    torch.cuda.empty_cache()

    return results


def print_summary_table(all_results: List[Dict]):
    """打印汇总对比表"""
    if not all_results:
        print('\nNo results to summarize.')
        return

    print('\n' + '=' * 100)
    print('FPS BENCHMARK SUMMARY')
    print('=' * 100)

    # 获取 GPU 信息
    gpu_name = all_results[0].get('gpu', 'Unknown')
    img_size = all_results[0].get('img_size', '?')
    print(f'GPU: {gpu_name} | Image: {img_size}x{img_size} | Batch: 1')
    print()

    # 表头
    header = (
        f'{"Model":<14} {"Desc":<48} '
        f'{"Latency(ms)":>12} {"FPS":>8} '
        f'{"B+Neck(ms)":>11} {"Head(ms)":>9} '
        f'{"mAP":>7}'
    )
    print(header)
    print('-' * len(header))

    # 基准 (a1 作为 LDMDet 基线, 或第一个结果)
    base_latency = None
    for r in all_results:
        if r['name'] == 'a1':
            base_latency = r['total_ms']
            break
    if base_latency is None:
        base_latency = all_results[0]['total_ms']

    for r in all_results:
        name = r['name']
        desc = r['desc'][:46]
        lat = r['total_ms']
        fps = r['fps']
        bn = r['backbone_neck_ms']
        hd = r['head_ms']
        mAP = r.get('known_mAP')
        mAP_str = f'{mAP:.3f}' if mAP is not None else '—'

        # 加速比
        speedup = base_latency / lat if lat > 0 else 0
        speedup_str = f'({speedup:.2f}x)' if name != 'a1' and speedup > 0 else ''

        print(
            f'{name:<14} {desc:<48} '
            f'{lat:>10.2f}  {fps:>8.1f} '
            f'{bn:>11.2f} {hd:>9.2f} '
            f'{mAP_str:>7}'
        )

    # LDMDet 采样器详情
    ldmdet_results = [r for r in all_results if 'solver_type' in r]
    if ldmdet_results:
        print(f'\nLDMDet Sampler Details:')
        print(f'{"Model":<14} {"Solver":<18} {"Steps":>6} {"Proposals":>10} {"TopK":>8}')
        print('-' * 60)
        for r in ldmdet_results:
            solver = r.get('solver_type', '?')
            steps = r.get('sampling_timesteps', '?')
            props = r.get('num_proposals', '?')
            topk = f'K={r["topk_k"]}' if r.get('topk_pruning') else 'off'
            print(f'{r["name"]:<14} {str(solver):<18} {str(steps):>6} {str(props):>10} {topk:>8}')


def save_results(all_results: List[Dict], output_dir: str = 'results'):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # JSON (完整数据)
    json_path = os.path.join(output_dir, f'benchmark_fps_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to: {json_path}')

    # Markdown (人类可读)
    md_path = os.path.join(output_dir, f'benchmark_fps_{timestamp}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f'# FPS Benchmark Results\n\n')
        f.write(f'**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'**GPU**: {all_results[0].get("gpu", "Unknown")}\n')
        f.write(f'**Image Size**: {all_results[0].get("img_size", "?")}x{all_results[0].get("img_size", "?")}\n')
        f.write(f'**Batch Size**: 1\n')
        f.write(f'**Warmup**: {all_results[0].get("warmup", "?")} | **Iters**: {all_results[0].get("iters", "?")}\n\n')

        f.write(f'## End-to-End Latency\n\n')
        f.write(f'| Model | Description | Latency (ms) | FPS | Backbone+Neck (ms) | Head (ms) | mAP |\n')
        f.write(f'|-------|-------------|-------------:|----:|-------------------:|----------:|----:|\n')

        base_latency = all_results[0]['total_ms']
        for r in all_results:
            mAP = r.get('known_mAP')
            mAP_str = f'{mAP:.3f}' if mAP is not None else '—'
            speedup = base_latency / r['total_ms'] if r['total_ms'] > 0 else 0
            f.write(
                f'| {r["name"]} | {r["desc"]} | '
                f'{r["total_ms"]:.2f} ± {r["std_ms"]:.2f} | '
                f'{r["fps"]:.1f} | '
                f'{r["backbone_neck_ms"]:.2f} | '
                f'{r["head_ms"]:.2f} | '
                f'{mAP_str} |\n'
            )

        # 采样器详情
        ldmdet_results = [r for r in all_results if 'solver_type' in r]
        if ldmdet_results:
            f.write(f'\n## LDMDet Sampler Configuration\n\n')
            f.write(f'| Model | Solver | Steps | Proposals | Top-K Pruning |\n')
            f.write(f'|-------|--------|------:|----------:|--------------|\n')
            for r in ldmdet_results:
                solver = r.get('solver_type', '?')
                steps = r.get('sampling_timesteps', '?')
                props = r.get('num_proposals', '?')
                topk = f'K={r["topk_k"]}' if r.get('topk_pruning') else 'off'
                f.write(f'| {r["name"]} | {solver} | {steps} | {props} | {topk} |\n')

    print(f'Results saved to: {md_path}')


def main():
    parser = argparse.ArgumentParser(
        description='LDMDet End-to-End FPS Benchmark'
    )
    parser.add_argument('--models', type=str, nargs='+', default=['all'],
                        help='Models to benchmark (space-separated). Use "all" for all models.')
    parser.add_argument('--img-size', type=int, default=512,
                        help='Input image size (square)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=100,
                        help='Measurement iterations')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    args = parser.parse_args()

    # 解析模型列表
    if 'all' in args.models:
        models_to_test = list(MODEL_REGISTRY.keys())
    else:
        models_to_test = args.models

    print('=' * 80)
    print('LDMDet End-to-End FPS Benchmark')
    print('=' * 80)
    print(f'PyTorch: {torch.__version__}')

    if not torch.cuda.is_available():
        print(f'ERROR: CUDA not available. GPU required for benchmark.')
        print(f'Run this script in a terminal with GPU access:')
        print(f'  conda activate chromo')
        print(f'  python ldmdet/tools/benchmark_fps.py --models all --gpu 0')
        sys.exit(1)

    print(f'CUDA device: {torch.cuda.get_device_name(args.gpu)}')
    print(f'Image size: {args.img_size}x{args.img_size}, Batch: 1')
    print(f'Warmup: {args.warmup}, Iters: {args.iters}')
    print(f'Models: {models_to_test}')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # 逐个测试
    all_results = []
    for model_name in models_to_test:
        result = run_benchmark(model_name, args.img_size, args.warmup, args.iters, 0)
        if result is not None:
            all_results.append(result)

    # 汇总
    print_summary_table(all_results)

    # 保存
    if all_results:
        save_results(all_results, args.output_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
