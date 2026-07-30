"""生成 ReFlow coupling 文件: 用 +DPM-Solver++ 对训练集推理, 保存 (x_0^pred, x_1^noise)

REFLOW_HEAD_DISTILL_IMPL_PLAN.md §1.5:
  对每张训练图:
    1. 固定 seed 生成 x_raw_noise = randn(1, num_proposals, 4)  (x_1^noise)
    2. +DPM-Solver++ 推理 (DPM-Solver++ 4步), 关闭 box_renewal/ensemble/pruning (保持 proposal 对应)
    3. 保存 {img_id: {'noise': x_1, 'x0_pred': x_0^pred}}  (raw/扩散空间, [num_proposals, 4])
  matched_idx 不存储 — 训练时在线重算 (基于 GT, 与 criterion target 一致, §1.5 v2 简化)

输出格式 (torch.save):
  {img_id (int): {'noise': tensor[num_proposals, 4],
                  'x0_pred': tensor[num_proposals, 4]}}

耦合文件由 reflow_standard_24obj.py 的 reflow_coupling_path 加载,
head._build_training_targets 在 reflow 模式下从其中读取 x_start=x0_pred, noise=noise.

Usage:
  python experiments/runners/generate_reflow_couplings.py \\
      experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \\
      --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \\
      --output work_dirs/reflow_couplings/train_couplings.pt \\
      --gpu-id 0 --seed 42 --num-groups 1

  # K=5 组 (训练时轮换, 保留 OT 多样性, §1.5):
  python experiments/runners/generate_reflow_couplings.py ... --num-groups 5
  # → 输出 train_couplings_group0.pt ... group4.pt

注意:
  - 生成时关闭 box_renewal (否则替换低置信 proposal, 破坏对应关系, PD-RF 教训)
  - 训练时 box_renewal 仅影响 eval (predict), 不影响 loss; 生成关 vs eval 开的不一致
    是已知风险 (§1.7), 通过 box_renewal 关闭生成保证 proposal 对应
  - bs=1 + shuffle=False: 保证 img_id ↔ batch_idx 稳定, 便于按 seed 复现
"""

import argparse
import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from mmengine.config import Config
from mmengine.runner import Runner


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='生成 ReFlow coupling (+DPM-Solver++ 推理)')
    parser.add_argument('config', help='+DPM-Solver++ config 文件路径')
    parser.add_argument('--checkpoint', required=True, help='+DPM-Solver++ checkpoint 路径')
    parser.add_argument('--output', default='work_dirs/reflow_couplings/train_couplings.pt',
                        help='输出 coupling 文件路径 (K=1) 或前缀 (K>1, _group{k}.pt)')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='基础种子 (每组 +100000 偏移)')
    parser.add_argument('--num-groups', type=int, default=1,
                        help='生成 K 组 coupling (不同 seed, 训练时轮换, §1.5)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='调试用: 仅处理前 N 张图 (None=全部)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    cfg = Config.fromfile(args.config)
    # 用训练集作为迭代源 (生成 coupling 针对训练集)
    # bs=1 + shuffle=False: 保证 img_id ↔ batch_idx 稳定, 便于按 seed 复现
    cfg.test_dataloader = dict(cfg.train_dataloader)
    cfg.test_dataloader['shuffle'] = False
    cfg.test_dataloader['batch_size'] = 1
    # test_evaluator 不参与 (我们手动迭代, 不调用 runner.test())
    # 但 Runner.from_cfg 要求 test_evaluator 存在, 沿用 base 配置即可

    # Runner.from_cfg 要求 work_dir 存在; 若 +DPM-Solver++ config 未定义则给临时目录
    # (coupling 生成不需要 work_dir, 仅满足 Runner 构建要求)
    if 'work_dir' not in cfg:
        cfg.work_dir = 'work_dirs/reflow_couplings_temp'

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)

    # 获取裸 detector (可能被 MMDataParallel 包装)
    det = runner.model
    if hasattr(det, 'module'):
        det = det.module

    # 关闭 box_renewal/ensemble/pruning: 保证单条干净轨迹, proposal 对应不变
    head = det.bbox_head
    head.box_renewal = False
    head.use_ensemble = False
    head.topk_pruning_enabled = False
    head.use_pcse = False
    head.use_ccbr = False
    det.eval()

    dataloader = runner.test_loop.dataloader
    n_total = len(dataloader)
    if args.max_images is not None:
        n_total = min(n_total, args.max_images)
    print(f'[ReFlow] 生成 coupling: {n_total} 图, K={args.num_groups} 组, seed={args.seed}')
    print(f'[ReFlow] 模型: box_renewal=False, ensemble=False, pruning=False (干净轨迹)')

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    for group in range(args.num_groups):
        group_seed = args.seed + group * 100000
        couplings = {}
        set_seed(group_seed)

        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if args.max_images is not None and idx >= args.max_images:
                    break
                # 每图固定 seed → 确定性初始噪声 (bs=1, idx 即图序号)
                torch.manual_seed(group_seed + idx)
                # val_step: data_preprocessor + predict; predict 内部设置
                #   _last_x_raw_initial (x_1) 与 _last_x0_final (x_0^pred)
                det.val_step(batch)

                x_raw_initial = head._last_x_raw_initial  # [1, num_proposals, 4]
                x0_final = head._last_x0_final            # [1, num_proposals, 4]

                data_samples = batch['data_samples']
                for i, ds in enumerate(data_samples):
                    img_id = ds.metainfo['img_id']
                    couplings[int(img_id)] = {
                        'noise': x_raw_initial[i].cpu().clone(),
                        'x0_pred': x0_final[i].cpu().clone(),
                    }

                if (idx + 1) % 100 == 0 or (idx + 1) == n_total:
                    print(f'  [group {group}] {idx + 1}/{n_total} '
                          f'(img_id={list(couplings)[-1]})')

        if args.num_groups > 1:
            out_path = args.output.replace('.pt', f'_group{group}.pt')
        else:
            out_path = args.output
        torch.save(couplings, out_path)
        print(f'[ReFlow] group {group} 完成: {len(couplings)} couplings → {out_path}')

    print(f'[ReFlow] 全部完成. reflow_coupling_path 应指向: {args.output}')


if __name__ == '__main__':
    main()
