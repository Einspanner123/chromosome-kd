#!/usr/bin/env python3
"""批量 test 评估脚本: 串行评估 12 个耦合消融 checkpoint, 每个完成后立即更新 DB。

修补配置问题:
  - D1 (Chromosome20240904): test_dataloader/test_evaluator 误指 valid/, 改为 test/
  - D2 (24 Chromosomes Object): test_dataloader/test_evaluator 误指 D1 路径, 改为 D2 路径;
    format_only=True 改为 False (否则不计算 mAP)

用法:
  python tools/experiment_db/batch_test_eval.py --gpu-id 0
  python tools/experiment_db/batch_test_eval.py --gpu-id 0 --only d1   # 只跑 D1
  python tools/experiment_db/batch_test_eval.py --gpu-id 0 --dry-run   # 只打印不执行
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

# 将项目根目录加入 sys.path, 使 mmengine.Config.fromfile 能导入 experiments.* 自定义模块
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

# 加载 dumped config 并修补的辅助
from mmengine.config import Config

# === 实验定义 ===
# 每项: (db_experiment_id, dataset, config_path, checkpoint_path)
EXPERIMENTS = [
    # --- D1 Random (standard aug) ---
    ('work_dirs/multi_seed_aug/rf_heun_adaln/seed_42',
     'd1', 'work_dirs/multi_seed_aug/rf_heun_adaln/seed_42/rf_heun_adaln.py',
     'work_dirs/multi_seed_aug/rf_heun_adaln/seed_42/best_coco_bbox_mAP_epoch_102.pth'),
    ('work_dirs/multi_seed_aug/rf_heun_adaln/seed_123',
     'd1', 'work_dirs/multi_seed_aug/rf_heun_adaln/seed_123/rf_heun_adaln.py',
     'work_dirs/multi_seed_aug/rf_heun_adaln/seed_123/best_coco_bbox_mAP_epoch_101.pth'),
    ('work_dirs/multi_seed_aug/rf_heun_adaln/seed_789',
     'd1', 'work_dirs/multi_seed_aug/rf_heun_adaln/seed_789/rf_heun_adaln.py',
     'work_dirs/multi_seed_aug/rf_heun_adaln/seed_789/best_coco_bbox_mAP_epoch_75.pth'),
    # --- D1 StochOT (standard aug) ---
    ('work_dirs/multi_seed/stochot_eps5_old/seed_42',
     'd1', 'work_dirs/multi_seed/stochot_eps5_old/seed_42/stochot_eps5_old_multiseed.py',
     'work_dirs/multi_seed/stochot_eps5_old/seed_42/best_coco_bbox_mAP_epoch_60.pth'),
    ('work_dirs/multi_seed/stochot_eps5_old/seed_123',
     'd1', 'work_dirs/multi_seed/stochot_eps5_old/seed_123/stochot_eps5_old_multiseed.py',
     'work_dirs/multi_seed/stochot_eps5_old/seed_123/best_coco_bbox_mAP_epoch_57.pth'),
    ('work_dirs/multi_seed/stochot_eps5_old/seed_789',
     'd1', 'work_dirs/multi_seed/stochot_eps5_old/seed_789/stochot_eps5_old_multiseed.py',
     'work_dirs/multi_seed/stochot_eps5_old/seed_789/best_coco_bbox_mAP_epoch_69.pth'),
    # --- D2 StochOT ---
    ('work_dirs/24obj_ablation/sinkhorn/seed_42',
     'd2', 'work_dirs/24obj_ablation/sinkhorn/seed_42/chromo_24obj_sinkhorn.py',
     'work_dirs/24obj_ablation/sinkhorn/seed_42/best_coco_bbox_mAP_epoch_53.pth'),
    ('work_dirs/24obj_ablation/sinkhorn/seed_123',
     'd2', 'work_dirs/24obj_ablation/sinkhorn/seed_123/chromo_24obj_sinkhorn_seed123.py',
     'work_dirs/24obj_ablation/sinkhorn/seed_123/best_coco_bbox_mAP_epoch_87.pth'),
    ('work_dirs/24obj_ablation/sinkhorn/seed_789',
     'd2', 'work_dirs/24obj_ablation/sinkhorn/seed_789/chromo_24obj_sinkhorn_seed789.py',
     'work_dirs/24obj_ablation/sinkhorn/seed_789/best_coco_bbox_mAP_epoch_44.pth'),
    # --- D2 Random ---
    ('work_dirs/24obj_ablation/random/seed_42',
     'd2', 'work_dirs/24obj_ablation/random/seed_42/chromo_24obj_random.py',
     'work_dirs/24obj_ablation/random/seed_42/best_coco_bbox_mAP_epoch_59.pth'),
    ('work_dirs/24obj_ablation/random/seed_123',
     'd2', 'work_dirs/24obj_ablation/random/seed_123/chromo_24obj_random.py',
     'work_dirs/24obj_ablation/random/seed_123/best_coco_bbox_mAP_epoch_115.pth'),
    ('work_dirs/24obj_ablation/random/seed_789',
     'd2', 'work_dirs/24obj_ablation/random/seed_789/chromo_24obj_random.py',
     'work_dirs/24obj_ablation/random/seed_789/best_coco_bbox_mAP_epoch_82.pth'),
]

# === 路径修补 ===
D1_DATA_ROOT = 'data/Chromosome20240904_NoAug_NoResize_coco/'
D2_DATA_ROOT = 'data/24_chromosomes_object/coco/'


def patch_config(cfg, dataset):
    """修补 test_dataloader 和 test_evaluator 路径, 返回修补后的 Config。"""
    if dataset == 'd1':
        # D1: valid/ → test/, 确保 format_only=False
        data_root = D1_DATA_ROOT
        test_ann_rel = 'test/_annotations.coco.json'
        test_img_prefix = 'test/'
    else:
        # D2: 改为 D2 路径, format_only=True → False
        data_root = D2_DATA_ROOT
        test_ann_rel = 'test/_annotations.coco.json'
        test_img_prefix = 'test/'

    test_ann_abs = data_root + test_ann_rel

    # 修补 test_dataloader
    if hasattr(cfg, 'test_dataloader'):
        cfg.test_dataloader.dataset.ann_file = test_ann_rel
        cfg.test_dataloader.dataset.data_root = data_root
        if 'data_prefix' in cfg.test_dataloader.dataset:
            cfg.test_dataloader.dataset.data_prefix = dict(img=test_img_prefix)
        elif 'data_prefix' in cfg.test_dataloader.dataset:
            cfg.test_dataloader.dataset['data_prefix'] = dict(img=test_img_prefix)

    # 修补 test_evaluator
    if hasattr(cfg, 'test_evaluator'):
        cfg.test_evaluator.ann_file = test_ann_abs
        cfg.test_evaluator.format_only = False
        # 移除 outfile_prefix 避免写入冲突
        if 'outfile_prefix' in cfg.test_evaluator:
            cfg.test_evaluator.outfile_prefix = None

    return cfg


def run_eval(config_path, checkpoint, gpu_id, dataset):
    """运行单个 test 评估, 返回 (mAP, AP50, AP75, AP_S, AP_M, AP_L) 或 None。"""
    cfg = Config.fromfile(config_path)
    cfg = patch_config(cfg, dataset)

    # 写修补后的 config 到临时文件
    tmp_dir = tempfile.mkdtemp(prefix='test_eval_')
    patched_config = os.path.join(tmp_dir, 'patched_config.py')
    cfg.dump(patched_config)

    # 运行 test.py
    cmd = [
        sys.executable, 'experiments/runners/test.py',
        patched_config,
        '--checkpoint', checkpoint,
        '--dataset', 'test',
        '--gpu-id', str(gpu_id),
        '--seed', '42',
    ]
    print(f'\n[CMD] {" ".join(cmd)}')
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=os.getcwd())
    except subprocess.TimeoutExpired:
        print(f'  [TIMEOUT] 评估超时 (>600s)')
        return None
    elapsed = time.time() - t0

    output = result.stdout + '\n' + result.stderr
    if result.returncode != 0:
        print(f'  [FAIL] returncode={result.returncode}, elapsed={elapsed:.1f}s')
        print(f'  stderr tail: {result.stderr[-500:]}')
        return None

    # 解析 mAP (CocoMetric 输出格式)
    # 匹配 "bbox_mAP: 0.7380" 或 "mAP: 0.7380" 等
    mAP = AP50 = AP75 = AP_S = AP_M = AP_L = None
    for line in output.split('\n'):
        # mmdet CocoMetric 输出: "copypaste: 0.738 0.926 ..." 或 "bbox_mAP": 0.738
        m = re.search(r"bbox_mAP[:\s]+([\d.]+)", line)
        if m and mAP is None:
            mAP = float(m.group(1))
        m = re.search(r"bbox_mAP_50[:\s]+([\d.]+)", line)
        if m and AP50 is None:
            AP50 = float(m.group(1))
        m = re.search(r"bbox_mAP_75[:\s]+([\d.]+)", line)
        if m and AP75 is None:
            AP75 = float(m.group(1))
        m = re.search(r"bbox_mAP_s[:\s]+([\d.]+)", line, re.I)
        if m and AP_S is None:
            AP_S = float(m.group(1))
        m = re.search(r"bbox_mAP_m[:\s]+([\d.]+)", line, re.I)
        if m and AP_M is None:
            AP_M = float(m.group(1))
        m = re.search(r"bbox_mAP_l[:\s]+([\d.]+)", line, re.I)
        if m and AP_L is None:
            AP_L = float(m.group(1))

    # 备用: 从 copypaste 行解析
    if mAP is None:
        cp = re.search(r"copypaste:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", output)
        if cp:
            mAP, AP50, AP75, AP_S, AP_M, AP_L = [float(x) for x in cp.groups()]

    if mAP is not None:
        print(f'  [OK] mAP={mAP:.4f}, AP50={AP50}, AP75={AP75}, elapsed={elapsed:.1f}s')
    else:
        print(f'  [WARN] mAP 解析失败, elapsed={elapsed:.1f}s')
        print(f'  output tail: {output[-800:]}')

    # 清理临时 config
    try:
        os.remove(patched_config)
        os.rmdir(tmp_dir)
    except Exception:
        pass

    return (mAP, AP50, AP75, AP_S, AP_M, AP_L)


def update_db(experiment_id, mAP, AP50, AP75, AP_S, AP_M, AP_L, db_path):
    """更新 DB: 插入或更新 test evaluation 记录。"""
    import sqlite3
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # 检查是否已有 test eval
    c.execute('SELECT eval_id FROM evaluation WHERE experiment_id=? AND split=?',
              (experiment_id, 'test'))
    existing = c.fetchone()
    if existing:
        c.execute('''UPDATE evaluation SET mAP=?, AP50=?, AP75=?, AP_small=?, AP_medium=?, AP_large=?, source=?
                     WHERE experiment_id=? AND split=?''',
                  (mAP, AP50, AP75, AP_S, AP_M, AP_L, 'batch_test_eval', experiment_id, 'test'))
        action = 'UPDATED'
    else:
        c.execute('''INSERT INTO evaluation
                     (experiment_id, split, mAP, AP50, AP75, AP_small, AP_medium, AP_large, source)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (experiment_id, 'test', mAP, AP50, AP75, AP_S, AP_M, AP_L, 'batch_test_eval'))
        action = 'INSERTED'
    conn.commit()
    conn.close()
    return action


def main():
    parser = argparse.ArgumentParser(description='批量 test 评估')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--only', choices=['d1', 'd2'], default=None, help='只跑 D1 或 D2')
    parser.add_argument('--filter', default=None, help='只跑 experiment_id 包含该子串的实验 (重试场景)')
    parser.add_argument('--dry-run', action='store_true', help='只打印计划不执行')
    parser.add_argument('--db', default='tools/experiment_db/experiments.db')
    args = parser.parse_args()

    exps = EXPERIMENTS
    if args.only:
        exps = [e for e in exps if e[1] == args.only]
    if args.filter:
        exps = [e for e in exps if args.filter in e[0]]

    print(f'=== 批量 test 评估: {len(exps)} 个实验, GPU={args.gpu_id} ===\n')

    if args.dry_run:
        for exp_id, dataset, cfg, ckpt in exps:
            exists = '✓' if os.path.exists(ckpt) else '✗'
            print(f'  [{dataset}] {exp_id}  ckpt={exists}  {ckpt}')
        return

    results = []
    for i, (exp_id, dataset, cfg_path, ckpt_path) in enumerate(exps, 1):
        print(f'\n{"="*70}')
        print(f'[{i}/{len(exps)}] {exp_id} (dataset={dataset})')
        print(f'  config: {cfg_path}')
        print(f'  ckpt:   {ckpt_path}')

        if not os.path.exists(ckpt_path):
            print(f'  [SKIP] checkpoint 不存在')
            results.append((exp_id, None, 'checkpoint_missing'))
            continue

        metrics = run_eval(cfg_path, ckpt_path, args.gpu_id, dataset)
        if metrics and metrics[0] is not None:
            mAP, AP50, AP75, AP_S, AP_M, AP_L = metrics
            action = update_db(exp_id, mAP, AP50, AP75, AP_S, AP_M, AP_L, args.db)
            print(f'  [DB] {action}: {exp_id} test mAP={mAP:.4f}')
            results.append((exp_id, mAP, 'ok'))
        else:
            print(f'  [DB] 跳过 (评估失败或 mAP 解析失败)')
            results.append((exp_id, None, 'eval_failed'))

    # 汇总
    print(f'\n{"="*70}')
    print('=== 汇总 ===')
    for exp_id, mAP, status in results:
        mAP_str = f'{mAP:.4f}' if mAP is not None else 'N/A'
        print(f'  {exp_id:<55} mAP={mAP_str}  ({status})')

    ok = sum(1 for _, _, s in results if s == 'ok')
    print(f'\n完成: {ok}/{len(exps)} 成功')


if __name__ == '__main__':
    main()
