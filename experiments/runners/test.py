"""Checkpoint evaluation for a resolved canonical configuration.

Usage:
    python experiments/runners/test.py artifacts/runs/<run-id>/resolved_config.py \\
        --checkpoint work_dirs/xxx/best_coco_bbox_mAP_epoch_59.pth \\
        --dataset val

采样器/步数覆盖:
    python experiments/runners/test.py artifacts/runs/<run-id>/resolved_config.py \\
        --checkpoint work_dirs/xxx/best.pth --dataset test \\
        --sampling-steps 3 --solver-type euler --seed 42

DDPM baseline:
    python experiments/runners/test.py artifacts/runs/<run-id>/resolved_config.py \\
        --checkpoint work_dirs/multi_seed_aug/ddpm/seed_42/best_coco_bbox_mAP_epoch_79.pth \\
        --dataset test --sampling-steps 4 --seed 42

SwanLab: 推理测试独立到 'ldmdet-inference' 项目, 实验名默认 '{solver}_{steps}step'
         (可用 --exp-name 覆盖)

输出: mAP, AP50, AP75, per-class AP

Per-image AP dump (用于 C3 统计显著性检验 Wilcoxon signed-rank + paired t-test):
    python experiments/runners/test.py artifacts/runs/<run-id>/resolved_config.py \\
        --checkpoint work_dirs/xxx/best.pth --dataset val \\
        --dump-per-image work_dirs/xxx/per_image_ap.json
"""

import argparse
import contextlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# 确保项目根目录在 Python path 中
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mmengine.config import Config
from mmengine.runner import Runner

# 推理测试独立 SwanLab 项目名 (与训练 'ldmdet-ablation' 分离)
_INFERENCE_SWANLAB_PROJECT = 'ldmdet-inference'


def set_seed(seed: int):
    """固定随机种子, 确保推理可复现 (初始噪声 + box_renewal 均受 seed 控制)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _find_coco_metric(runner):
    """从 runner.test_evaluator.metrics 中查找 CocoMetric 实例。

    返回 CocoMetric 或 None (当不存在时)。
    """
    for metric in runner.test_evaluator.metrics:
        # 兼容 CocoMetric 及其子类 (按类名匹配, 避免硬依赖 import 路径)
        cls_name = type(metric).__name__
        if cls_name == 'CocoMetric':
            return metric
    return None


def _set_evaluator_outfile_prefix(evaluator_cfg, prefix):
    """向 test_evaluator config (dict 或 list) 中所有 CocoMetric 注入 outfile_prefix。

    用于在 dump-per-image 模式下让 CocoMetric 把预测结果写到稳定路径 (而非临时目录)。
    """
    if isinstance(evaluator_cfg, dict):
        evaluator_cfg['outfile_prefix'] = prefix
    elif isinstance(evaluator_cfg, list):
        for ev in evaluator_cfg:
            if isinstance(ev, dict):
                ev['outfile_prefix'] = prefix
    # 其它形态 (已实例化的 Evaluator 对象等) 不处理, 后续 dump 会明确报错


def _set_dotted(config, dotted_key, value):
    """Apply an explicit JSON override to a resolved configuration."""
    current = config
    parts = dotted_key.split('.')
    if not parts or any(not part for part in parts):
        raise ValueError(f'invalid override key: {dotted_key!r}')
    for part in parts[:-1]:
        if part not in current:
            raise KeyError(f'override path does not exist: {dotted_key}')
        current = current[part]
    current[parts[-1]] = value


def _load_overrides(raw):
    if raw is None:
        return {}
    if raw.startswith('@'):
        payload = Path(raw[1:]).read_text(encoding='utf-8')
    else:
        payload = raw
    overrides = json.loads(payload)
    if not isinstance(overrides, dict):
        raise ValueError('--override-json must decode to an object')
    return overrides


def dump_per_image_ap(runner, dump_path):
    """dump per-image AP (COCO 标准 6 指标) 到 JSON, 用于 C3 统计显著性检验。

    依赖 runner.test_evaluator.metrics 中的 CocoMetric 已完成 evaluate(),
    并且其 outfile_prefix 指向稳定路径 (即预测 .bbox.json 仍然存在)。

    JSON 格式 (list[dict], 每张图一条):
        [
            {
                "image_id": 1,
                "image_path": "valid/foo.png",
                "AP":   0.812,   # COCO mAP, IoU=0.5:0.95
                "AP50": 0.935,   # IoU=0.50
                "AP75": 0.851,   # IoU=0.75
                "AP_S": 0.612,   # small  objects
                "AP_M": 0.783,   # medium objects
                "AP_L": 0.911    # large  objects
            },
            ...
        ]

    说明:
      - 若一张图无 GT 或无预测, 对应 AP 值可能为 -1 (pycocotools 约定, 表示未定义)。
        下游统计检验可自行决定是否过滤。
      - maxDets / iouThrs / catIds 与 CocoMetric 自身设置保持一致,
        因此 per-image AP 与 runner.test() 返回的聚合 mAP 在数值上可对齐。
    """
    # 延迟 import: 仅在 dump 模式下需要
    from mmdet.datasets.api_wrappers import COCOeval
    from mmengine.fileio import load

    coco_metric = _find_coco_metric(runner)
    if coco_metric is None:
        raise RuntimeError(
            '[dump-per-image] test_evaluator 中未找到 CocoMetric, 无法计算 per-image AP。'
        )

    coco_gt = coco_metric._coco_api
    if coco_gt is None:
        raise RuntimeError(
            '[dump-per-image] CocoMetric._coco_api 为 None。'
            ' 请确认 runner.test() 已执行, 且 ann_file 配置正确。'
        )

    outfile_prefix = coco_metric.outfile_prefix
    if outfile_prefix is None:
        raise RuntimeError(
            '[dump-per-image] CocoMetric.outfile_prefix 为 None。'
            ' 内部逻辑应在 runner.test() 前注入 outfile_prefix, 请检查代码。'
        )
    pred_file = f'{outfile_prefix}.bbox.json'
    if not os.path.exists(pred_file):
        raise RuntimeError(
            f'[dump-per-image] 预测结果文件不存在: {pred_file}。'
            ' 可能是 runner.test() 未写入, 或文件已被清理。'
        )

    # 复用 mmdet CocoMetric 的预测结果 -> 构造 coco_dt
    predictions = load(pred_file)
    if len(predictions) == 0:
        raise RuntimeError(f'[dump-per-image] 预测结果为空: {pred_file}')
    coco_dt = coco_gt.loadRes(predictions)

    # 与 CocoMetric 自身使用的评估参数保持一致 (保证与聚合 mAP 对齐)
    cat_ids = coco_metric.cat_ids
    img_ids = coco_metric.img_ids
    iou_thrs = coco_metric.iou_thrs
    proposal_nums = coco_metric.proposal_nums

    # image_id -> file_name (若 COCO GT 中有 images 字段)
    img_id_to_path = {}
    for img_info in coco_gt.dataset.get('images', []):
        img_id_to_path[img_info['id']] = img_info.get('file_name', '')

    # 静默 pycocotools 的 verbose 输出 (evaluate/accumulate/summarize 都会 print)
    per_image_records = []
    n_images = len(img_ids)
    for idx, img_id in enumerate(img_ids):
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.catIds = cat_ids
        coco_eval.params.imgIds = [img_id]
        # summarize() 需 maxDets 至少 3 个元素 (访问 maxDets[2]), 这里沿用 CocoMetric 设置
        coco_eval.params.maxDets = list(proposal_nums)
        coco_eval.params.iouThrs = iou_thrs

        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        # stats[0:6] = [AP, AP50, AP75, AP_S, AP_M, AP_L]
        stats = coco_eval.stats
        record = {
            'image_id': int(img_id),
            'image_path': img_id_to_path.get(int(img_id), ''),
            'AP': float(stats[0]),
            'AP50': float(stats[1]),
            'AP75': float(stats[2]),
            'AP_S': float(stats[3]),
            'AP_M': float(stats[4]),
            'AP_L': float(stats[5]),
        }
        per_image_records.append(record)

        if (idx + 1) % 100 == 0 or (idx + 1) == n_images:
            print(f'[dump-per-image] progress: {idx + 1}/{n_images}')

    # 写出 JSON
    dump_dir = os.path.dirname(os.path.abspath(dump_path))
    os.makedirs(dump_dir, exist_ok=True)
    with open(dump_path, 'w') as f:
        json.dump(per_image_records, f, indent=2)

    print(
        f'[dump-per-image] 已写出 {len(per_image_records)} 条记录 -> {dump_path}'
    )


def main():
    parser = argparse.ArgumentParser(description='LDMDet Test (Inference)')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument(
        '--dataset',
        default='val',
        choices=['val', 'test'],
        help='Dataset split',
    )
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible inference',
    )
    parser.add_argument(
        '--sampling-steps',
        type=int,
        default=None,
        help='Override sampling steps',
    )
    parser.add_argument(
        '--solver-type',
        type=str,
        default=None,
        choices=['euler', 'heun', 'ddim', 'dpm_solver_pp', 'dpm_solver_pp_3'],
        help='Override solver type',
    )
    parser.add_argument(
        '--exp-name',
        type=str,
        default=None,
        help='SwanLab experiment name (default: auto {solver}_{steps}step)',
    )
    parser.add_argument(
        '--dump-per-image',
        type=str,
        default=None,
        metavar='PATH',
        help='若指定, 将 per-image AP (AP/AP50/AP75/AP_S/AP_M/AP_L) '
        'dump 到该 JSON 文件, 用于 C3 统计显著性检验。'
        '不加此参数时行为与原先完全一致。',
    )
    parser.add_argument(
        '--prediction-prefix',
        help='Persistent CocoMetric outfile prefix (writes .bbox.json).',
    )
    parser.add_argument(
        '--output-json',
        help='Write machine-readable framework metrics to this JSON file.',
    )
    parser.add_argument(
        '--override-json',
        help='JSON object (or @file) of existing dotted config keys to override.',
    )
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # 固定随机种子 (推理随机性来源: 初始噪声 proposals + box_renewal)
    set_seed(args.seed)

    cfg = Config.fromfile(args.config)
    # MMEngine dumps the resolved runtime config using ``cfg.filename``.  The
    # input may itself be the immutable preregistered ``resolved_config.py``;
    # give the runtime dump a distinct basename so evaluation cannot overwrite
    # the evidence artifact it is meant to verify.
    # Config overrides normal attribute assignment and would otherwise store
    # ``_filename`` as a configuration key without changing ``cfg.filename``.
    cfg.pop('_filename', None)
    object.__setattr__(cfg, '_filename', 'runtime_test_config.py')
    cfg.work_dir = os.path.dirname(args.checkpoint) or 'work_dirs/test'

    for key, value in sorted(_load_overrides(args.override_json).items()):
        _set_dotted(cfg, key, value)

    if args.sampling_steps is not None:
        cfg.model.bbox_head.sampling_timesteps = args.sampling_steps

    if args.solver_type is not None:
        cfg.model.bbox_head.solver_type = args.solver_type

    # 生成 SwanLab 实验名 (推理测试独立项目)
    if args.exp_name is None:
        solver = args.solver_type or cfg.model.bbox_head.get(
            'solver_type', 'unknown'
        )
        steps = (
            args.sampling_steps
            if args.sampling_steps is not None
            else cfg.model.bbox_head.get('sampling_timesteps', 0)
        )
        args.exp_name = f'{solver}_{steps}step'

    # 覆盖 SwanLab 配置: 推理独立项目 + 明确实验名
    # 注意: cfg.vis_backends 与 cfg.visualizer['vis_backends'] 可能是不同对象, 需同时修改
    _vis_backends_list = cfg.get('vis_backends', [])
    _visualizer_backends = cfg.get('visualizer', {}).get('vis_backends', [])
    for _vis_backends in [_vis_backends_list, _visualizer_backends]:
        for backend in _vis_backends:
            if (
                isinstance(backend, dict)
                and backend.get('type') == 'SwanlabVisBackend'
            ):
                backend.setdefault('init_kwargs', {})
                backend['init_kwargs']['project'] = _INFERENCE_SWANLAB_PROJECT
                backend['init_kwargs']['experiment_name'] = args.exp_name

    print(
        f'[SwanLab] project={_INFERENCE_SWANLAB_PROJECT}, exp_name={args.exp_name}, seed={args.seed}'
    )

    # 切换为测试模式
    # Runner.test() 使用 cfg.test_dataloader/test_evaluator, 因此需要根据
    # --dataset 参数将 val 配置映射到 test 位:
    #   --dataset val  (默认): 用 val_dataloader/val_evaluator (评估验证集)
    #   --dataset test          : 用原始 test_dataloader/test_evaluator (评估测试集)
    if args.dataset == 'val':
        cfg.test_dataloader = cfg.val_dataloader
        cfg.test_evaluator = cfg.val_evaluator

    # Per-image AP dump 模式: 让 CocoMetric 把预测写到稳定路径 (而非默认临时目录)
    # 否则 compute_metrics 返回后临时目录会被立即清理, 无法重算 per-image AP。
    _pred_temp_dir = None
    if args.prediction_prefix is not None:
        _pred_prefix = os.path.abspath(args.prediction_prefix)
        os.makedirs(os.path.dirname(_pred_prefix), exist_ok=True)
        _set_evaluator_outfile_prefix(cfg.test_evaluator, _pred_prefix)
        print(f'[prediction] persistent output prefix: {_pred_prefix}')
    elif args.dump_per_image is not None:
        _pred_temp_dir = tempfile.mkdtemp(prefix='mmdet_preds_')
        _pred_prefix = os.path.join(_pred_temp_dir, 'preds')
        _set_evaluator_outfile_prefix(cfg.test_evaluator, _pred_prefix)
        print(f'[dump-per-image] 预测临时目录: {_pred_temp_dir}')
        print(f'[dump-per-image] 输出目标: {args.dump_per_image}')

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)

    # 运行测试
    metrics = runner.test()
    print(f'\n{"=" * 60}')
    print(f'Test Results ({args.dataset}) [{args.exp_name}, seed={args.seed}]')
    print(f'{"=" * 60}')
    for k, v in sorted(metrics.items()):
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    if args.output_json is not None:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'config': os.path.abspath(args.config),
            'checkpoint': os.path.abspath(args.checkpoint),
            'split': args.dataset,
            'inference_seed': args.seed,
            'overrides': _load_overrides(args.override_json),
            'metrics': {
                key: (float(value) if hasattr(value, '__float__') else value)
                for key, value in metrics.items()
            },
        }
        temporary = output_path.with_suffix(output_path.suffix + '.tmp')
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + '\n'
        )
        temporary.replace(output_path)
        print(f'[metrics] machine-readable output: {output_path}')

    # Per-image AP dump (用于 C3 统计显著性检验)
    if args.dump_per_image is not None:
        try:
            dump_per_image_ap(runner, args.dump_per_image)
        finally:
            # 清理临时预测目录 (per-image AP 已写出, 不再需要原始预测)
            if _pred_temp_dir is not None:
                shutil.rmtree(_pred_temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
