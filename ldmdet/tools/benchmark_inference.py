"""LDMDet 推理性能 Benchmark

测量不同采样器 (Euler / Heun / DPM-Solver++ 2nd / DPM-Solver++ 3rd) 在不同步数下的:
  - 端到端延迟与 FPS
  - CUDA kernel 级瓶颈排名 (PyTorch Profiler)
  - 按功能模块聚合的耗时占比 (roi_align / attention / dynamic_conv / ffn / cls_head / ...)

用于评估 commit 0da80238 (Triton 优化: #1 SDPA + #3 Sinkhorn fused logsumexp) 在
新结构下的推理瓶颈, 并对比 DPM-Solver++ 与 Euler/Heun 的采样器开销。

用法:
    # 默认 (A3 SOTA 配置: RF + Heun + AdaLN + StochOT, 24 类, 500 proposals)
    python ldmdet/tools/benchmark_inference.py

    # 自定义步数与采样器对比
    python ldmdet/tools/benchmark_inference.py \\
        --solvers euler heun dpm_solver_pp dpm_solver_pp_3 \\
        --steps 1 2 4 8 \\
        --bs 1 --num-proposals 500

    # 仅测延迟, 不做 kernel profile
    python ldmdet/tools/benchmark_inference.py --no-profile

    # 指定 GPU
    python ldmdet/tools/benchmark_inference.py --gpu 0
"""

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from ldmdet.tools.profile_training import build_model, generate_synthetic_batch


# ============================================================
# 采样器配置
# ============================================================
# 每个 solver 的描述 + 每步模型调用次数 (用于估算理论开销)
SOLVER_INFO = {
    'euler': {
        'desc': 'Euler 1阶 (RF 直线路径)',
        'calls_per_step': 1,
    },
    'heun': {
        'desc': 'Heun 2阶 (梯形校正, 每步 2 次模型调用)',
        'calls_per_step': 2,
    },
    'dpm_solver_pp': {
        'desc': 'DPM-Solver++ 2阶 (RF 多步法, history-based)',
        'calls_per_step': 1,
    },
    'dpm_solver_pp_3': {
        'desc': 'DPM-Solver++ 3阶 (RF 多步法, 3 阶 history)',
        'calls_per_step': 1,
    },
}


def set_solver(head, solver_type: str, sampling_timesteps: int):
    """运行时切换 head 的采样器 (避免重建模型)"""
    head.solver_type = solver_type
    head._sampler.solver_type = solver_type
    head._sampler.sampling_timesteps = sampling_timesteps


# ============================================================
# 延迟测量
# ============================================================
def cuda_time(fn, warmup=10, iters=50):
    """测量 CUDA 函数平均执行时间 (ms)"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def benchmark_latency(head, features, img_metas, solvers, steps_list,
                      warmup=10, iters=50):
    """对每个 (solver, steps) 组合测量端到端延迟"""
    print('\n' + '=' * 80)
    print('LATENCY BENCHMARK')
    print('=' * 80)
    print(f'{"Solver":<22} {"Steps":>6} {"Model Calls":>13} '
          f'{"Latency (ms)":>15} {"FPS":>8}')
    print('-' * 80)

    results = []
    for solver in solvers:
        for steps in steps_list:
            set_solver(head, solver, steps)

            def fn():
                head.predict(features, img_metas, rescale=False)

            try:
                t = cuda_time(fn, warmup=warmup, iters=iters)
            except Exception as e:
                print(f'{solver:<22} {steps:>6d}   ERROR: {e}')
                continue

            fps = 1000.0 / t if t > 0 else 0
            calls = SOLVER_INFO[solver]['calls_per_step'] * steps
            print(f'{solver:<22} {steps:>6d} {calls:>13d} '
                  f'{t:>15.2f} {fps:>8.1f}')
            results.append((solver, steps, t, fps))

    return results


# ============================================================
# Kernel 级 Profile
# ============================================================
def profile_solver(head, features, img_metas, solver: str, steps: int,
                   profile_iters: int = 5):
    """对指定 solver 做 PyTorch Profiler kernel 级分析"""
    from torch.profiler import profile, ProfilerActivity

    set_solver(head, solver, steps)
    # Warmup
    for _ in range(3):
        head.predict(features, img_metas, rescale=False)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(profile_iters):
            head.predict(features, img_metas, rescale=False)

    return prof


def print_kernel_ranking(prof, solver: str, steps: int, top_n: int = 25):
    """打印 CUDA kernel 排名"""
    print('\n' + '=' * 80)
    print(f'TOP-{top_n} CUDA KERNELS BY TOTAL CUDA TIME  '
          f'[{solver}, {steps} steps]')
    print('=' * 80)
    print(prof.key_averages().table(
        sort_by='cuda_time_total', row_limit=top_n
    ))


def print_category_breakdown(prof, solver: str, steps: int):
    """按功能模块聚合 kernel 耗时

    注意: PyTorch Profiler 的 key_averages() 会同时报告 op 级 (aten::addmm) 和
    raw CUDA kernel 级 (ampere_sgemm_*) 条目, 二者的 Self CUDA time 是同一段
    GPU 时间的不同视角。为避免 double-counting, 这里只统计 op 级条目
    (aten::*, torchvision::*, triton_*), 跳过 raw CUDA kernels。
    """
    # 推理路径模块映射 (按优先级排序: 越具体的越先匹配)
    # 注意: keyword 匹配是 substring 匹配, 要避免歧义
    categories = [
        ('roi_align',    ['roi_align', 'RoIAlign', 'roi_pool']),
        ('attention',    ['scaled_dot_product', 'multihead_attention',
                          '_efficient_attention', '_flash_attention_forward',
                          'flash_attn']),
        # bmm 优先归 dynamic_conv (推理路径中 bmm 主要在 DynamicConv)
        ('dynamic_conv', ['::bmm', 'bmm']),
        ('ffn',          ['addmm', '::linear', 'mm']),
        ('norm',         ['layer_norm', 'batch_norm', 'LayerNorm',
                          'group_norm']),
        ('activation',   ['relu', '::silu', 'gelu', 'clamp_min']),
        ('box_ops',      ['box_iou', 'nms', 'giou']),
        ('sinkhorn',     ['logsumexp', 'cdist']),
        ('indexing',     ['::index', '::gather', '::scatter', '::select',
                          'nonzero', 'index_put']),
        ('concat',       ['::cat', '::stack']),
        ('elementwise',  ['::mul', '::add', '::sub', '::div', '::exp',
                          '::log', '::where', '::copy_', '::fill_',
                          '::zero_', 'vectorized_elementwise']),
        ('reduction',    ['::sum', '::mean', '::max', '::min', '::prod',
                          'reduction']),
        ('solver',       ['dpm_solver', 'heun', 'rectified_flow']),
        ('other',        []),
    ]
    cat_keywords = {c: kws for c, kws in categories}

    key_avgs = prof.key_averages()
    cat_times = {}
    uncategorized = []

    for item in key_avgs:
        name = item.key
        cuda_time = item.cuda_time_total
        if cuda_time == 0:
            continue

        # 跳过 raw CUDA kernels (避免 double-counting)
        # raw kernels 通常不以 aten:: / torchvision:: / triton 开头
        is_op_level = (
            name.startswith('aten::')
            or name.startswith('torchvision::')
            or name.startswith('triton')
            or name.startswith('custom::')
        )
        if not is_op_level:
            continue

        categorized = False
        for cat_name, keywords in categories:
            if cat_name == 'other':
                continue
            if any(kw.lower() in name.lower() for kw in keywords):
                cat_times[cat_name] = cat_times.get(cat_name, 0) + cuda_time
                categorized = True
                break
        if not categorized:
            cat_times['other'] = cat_times.get('other', 0) + cuda_time
            uncategorized.append((name, cuda_time))

    total_cuda_time = sum(cat_times.values())

    print('\n' + '=' * 80)
    print(f'INFERENCE KERNEL CATEGORY BREAKDOWN (op-level)  '
          f'[{solver}, {steps} steps]')
    print('=' * 80)
    print(f'\n{"Category":<18} {"Time (ms)":>12} {"Percentage":>12}')
    print('-' * 44)
    for cat, t in sorted(cat_times.items(), key=lambda x: -x[1]):
        if t == 0:
            continue
        pct = t / max(total_cuda_time, 1) * 100
        print(f'{cat:<18} {t/1000:>12.2f} {pct:>11.1f}%')
    print('-' * 44)
    print(f'{"TOTAL":<18} {total_cuda_time/1000:>12.2f}')

    if uncategorized:
        print(f'\nTop uncategorized op-level kernels:')
        for name, t in sorted(uncategorized, key=lambda x: -x[1])[:10]:
            print(f'  {name:<60} {t/1000:.2f} ms')

    # 额外: 列出 Top raw CUDA kernels (供参考, 不计入 category)
    print(f'\nTop raw CUDA kernels (not counted in categories above):')
    raw_kernels = []
    for item in key_avgs:
        name = item.key
        cuda_time = item.cuda_time_total
        if cuda_time == 0:
            continue
        if not (name.startswith('aten::')
                or name.startswith('torchvision::')
                or name.startswith('triton')
                or name.startswith('custom::')):
            raw_kernels.append((name, cuda_time))
    for name, t in sorted(raw_kernels, key=lambda x: -x[1])[:10]:
        print(f'  {name:<60} {t/1000:.2f} ms')

    return cat_times, total_cuda_time


# ============================================================
# Solver 间瓶颈排名对比
# ============================================================
def compare_category_breakdown(per_solver_cats: dict, total_dict: dict,
                               steps: int):
    """打印各 solver 在同一 step 下的 category 排名对比"""
    print('\n' + '=' * 80)
    print(f'CATEGORY RANKING COMPARISON @ {steps} steps')
    print('=' * 80)

    # 收集所有 category
    all_cats = set()
    for s, cats in per_solver_cats.items():
        all_cats.update(cats.keys())
    all_cats = sorted(all_cats)

    # 表头
    solvers = list(per_solver_cats.keys())
    header = f'{"Category":<18}'
    for s in solvers:
        header += f' {s:>16}'
    print(header)
    print('-' * (18 + 17 * len(solvers)))

    for cat in all_cats:
        line = f'{cat:<18}'
        for s in solvers:
            t_us = per_solver_cats[s].get(cat, 0)  # μs
            total_us = total_dict.get(s, 1)  # μs
            t_ms = t_us / 1000
            pct = t_us / max(total_us, 1) * 100
            line += f' {t_ms:>8.2f}({pct:>4.1f}%)'
        print(line)


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='LDMDet Inference Benchmark (multi-solver, kernel-level)'
    )
    # 模型参数 (与 profile_kernel_level.py 对齐, A3 SOTA 默认)
    parser.add_argument("--num-proposals", type=int, default=500)
    parser.add_argument("--num-heads", type=int, default=6,
                        help='级联 head 数量')
    parser.add_argument("--num-classes", type=int, default=24)
    parser.add_argument("--feat-channels", type=int, default=256)
    parser.add_argument("--pooler-resolution", type=int, default=7)
    parser.add_argument("--bs", type=int, default=1,
                        help='推理 batch size')
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--snr-scale", type=float, default=2.0)
    parser.add_argument("--diffusion-type", type=str, default='rectified_flow')
    parser.add_argument("--rf-schedule", type=str, default='shifted')
    parser.add_argument("--rf-shift", type=float, default=2.0)
    parser.add_argument("--coupling", type=str, default='ot_flow',
                        choices=['random', 'ot_flow'],
                        help='耦合策略 (ldmdet 纯 PyTorch 库支持的名称)')
    parser.add_argument("--ot-epsilon", type=float, default=5.0)
    parser.add_argument("--ot-num-iters", type=int, default=20)
    parser.add_argument("--time-conditioning", type=str, default='scale_shift',
                        choices=['scale_shift', 'adaln_zero'])
    parser.add_argument("--use-flash-attn", action="store_true",
                        help='(compat, no-op; SDPA 默认启用)')
    parser.add_argument("--scale-aware", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default=None,
                        choices=['bfloat16', 'float16', 'bf16', 'fp16'],
                        help='推理 AMP 精度 (启用 predict() 路径的 autocast)')
    parser.add_argument("--torch-compile", action='store_true',
                        help='用 torch.compile 编译 forward (减少 kernel launch)')

    # Benchmark 参数
    parser.add_argument("--solvers", type=str, nargs='+',
                        default=['euler', 'heun', 'dpm_solver_pp',
                                 'dpm_solver_pp_3'],
                        choices=list(SOLVER_INFO.keys()),
                        help='待测采样器列表')
    parser.add_argument("--steps", type=int, nargs='+',
                        default=[1, 2, 4, 8],
                        help='采样步数列表')
    parser.add_argument("--no-profile", action='store_true',
                        help='只测延迟, 不做 kernel profile')
    parser.add_argument("--profile-solver", type=str, default='dpm_solver_pp',
                        choices=list(SOLVER_INFO.keys()),
                        help='做 kernel profile 的 solver')
    parser.add_argument("--profile-steps", type=int, default=4,
                        help='做 kernel profile 的步数')
    parser.add_argument("--profile-iters", type=int, default=5)
    parser.add_argument("--top-kernels", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    args.use_sdpa = not args.use_flash_attn  # profile_training.build_model 用 use_sdpa
    # 单独的 attn_half 开关 (默认 False, 与 A3 一致)
    args.attn_half = False
    # 解析 amp_dtype 字符串到 torch.dtype
    amp_dtype_map = {
        None: None,
        'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
        'float16': torch.float16, 'fp16': torch.float16,
    }
    amp_dtype = amp_dtype_map.get(args.amp_dtype, None)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============================================================
    # 构建模型 (A3 SOTA 配置)
    # ============================================================
    head = build_model(args).to(device)
    if amp_dtype is not None:
        head.amp_dtype = amp_dtype
    if args.torch_compile and hasattr(torch, 'compile'):
        print('[torch.compile] Compiling forward (may take a while...)')
        head.forward = torch.compile(head.forward, dynamic=True)
    head.eval()  # 推理模式

    # 构造合成输入
    features, img_metas, gt_bboxes, gt_labels = generate_synthetic_batch(args, device)

    print('=' * 80)
    print('LDMDet Inference Benchmark')
    print('=' * 80)
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    try:
        import triton
        print(f'Triton: {triton.__version__}')
    except ImportError:
        print('Triton: not available')
    print(f'\nModel: bs={args.bs}, num_proposals={args.num_proposals}, '
          f'num_heads={args.num_heads}, num_classes={args.num_classes}')
    print(f'       img_size={args.img_size}, coupling={args.coupling} '
          f'(eps={args.ot_epsilon}, iters={args.ot_num_iters})')
    print(f'       diffusion={args.diffusion_type}, schedule={args.rf_schedule}, '
          f'shift={args.rf_shift}')
    print(f'       time_conditioning={args.time_conditioning}, '
          f'use_sdpa={args.use_sdpa}, attn_half={args.attn_half}, '
          f'amp_dtype={amp_dtype}, torch_compile={args.torch_compile}')

    # ============================================================
    # 1. 延迟基准 (所有 solver × 所有 steps)
    # ============================================================
    latency_results = benchmark_latency(
        head, features, img_metas, args.solvers, args.steps,
        warmup=args.warmup, iters=args.iters,
    )

    # ============================================================
    # 2. Kernel 级 Profile (单一 solver, 指定步数)
    # ============================================================
    if not args.no_profile:
        prof = profile_solver(
            head, features, img_metas,
            solver=args.profile_solver, steps=args.profile_steps,
            profile_iters=args.profile_iters,
        )

        # 2a. Top-N kernel 排名
        print_kernel_ranking(
            prof, args.profile_solver, args.profile_steps,
            top_n=args.top_kernels,
        )

        # 2b. 按模块聚合的瓶颈排名
        cat_times, total_cuda = print_category_breakdown(
            prof, args.profile_solver, args.profile_steps,
        )

        # ============================================================
        # 3. 对比所有 solver 在 profile_steps 下的瓶颈分布
        # ============================================================
        if len(args.solvers) > 1:
            per_solver_cats = {}
            total_dict = {}
            for s in args.solvers:
                if s == args.profile_solver:
                    per_solver_cats[s] = cat_times
                    total_dict[s] = total_cuda
                    continue
                prof_s = profile_solver(
                    head, features, img_metas,
                    solver=s, steps=args.profile_steps,
                    profile_iters=args.profile_iters,
                )
                cats_s, total_s = print_category_breakdown(
                    prof_s, s, args.profile_steps,
                )
                per_solver_cats[s] = cats_s
                total_dict[s] = total_s

            compare_category_breakdown(
                per_solver_cats, total_dict, args.profile_steps,
            )

    # ============================================================
    # Summary
    # ============================================================
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print('\n[Latency @ 4 steps] (typical inference setting):')
    print(f'{"Solver":<22} {"Latency (ms)":>15} {"FPS":>8} '
          f'{"Calls":>8} {"Calls/Step":>12}')
    print('-' * 70)
    for solver, steps, t, fps in latency_results:
        if steps != 4:
            continue
        calls = SOLVER_INFO[solver]['calls_per_step'] * steps
        cps = SOLVER_INFO[solver]['calls_per_step']
        print(f'{solver:<22} {t:>15.2f} {fps:>8.1f} '
              f'{calls:>8d} {cps:>12d}')

    print('\n[Sampler theoretical cost]')
    for s in args.solvers:
        info = SOLVER_INFO[s]
        print(f'  {s:<20} {info["desc"]}')

    print('\nDone.')


if __name__ == '__main__':
    main()
