#!/usr/bin/env python
"""修复 ε 消融实验数据库中的错误标注和缺失数据。

问题 1: ablation/ 下 6 个实验的 coupling/ε/ot_matcher/ot_sample 标注错误
  - DB 错误标注为 coupling=random, ε=None, aug=standard
  - 实际应为 ot_flow 耦合、ε=1/2/5、ot_matcher=sinkhorn、ot_sample=True
  - 原因: 旧配置使用 ot_coupling/ot_epsilon 直接在 bbox_head 下 (非 coupling=dict 格式),
    build_experiment_db.py 的 parse_dumped_config 只查找 coupling=dict, 导致解析失败

问题 2: ablation_old/ 下 5 个实验 val mAP 缺失
  - DB 中 best_val_mAP=None, 但有 best checkpoint
  - 原因: work_dir 下无日志文件 (.log) 和 scalars.json, 日志解析失败
  - 解决: 从 checkpoint 的 message_hub.runtime_info.best_score 提取 mAP
  - 同时从 checkpoint meta.cfg 字符串和 HistoryBuffer 提取配置和 per-epoch 指标

用法:
  conda run -n chromo python tools/experiment_db/fix_eps_ablation_db.py
  conda run -n chromo python tools/experiment_db/fix_eps_ablation_db.py --dry-run
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

# ─── 路径配置 ───────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments.db')
ROSS_ROOT = '/media/ross/8TB/linkst/chromo/chromosome-kd'

# ─── 数据集路径模式 ─────────────────────────────────────────
D1_DATA_ROOT_PATTERN = 'Chromosome20240904'
D2_DATA_ROOT_PATTERN = '24_chromosomes_object'

# ─── Problem 1: ablation/ 下 6 个实验 ──────────────────────
ABLATION_EXPERIMENTS = [
    ('work_dirs/ablation/stochot_eps1', 'stochot_eps1.py'),
    ('work_dirs/ablation/stochot_eps2', 'stochot_eps2.py'),
    ('work_dirs/ablation/stochot_eps5', 'stochot_eps5.py'),
    ('work_dirs/ablation/adaln_stochot_eps1', 'adaln_stochot_eps1.py'),
    ('work_dirs/ablation/adaln_stochot_eps2', 'adaln_stochot_eps2.py'),
    ('work_dirs/ablation/adaln_stochot_eps5', 'adaln_stochot_eps5.py'),
]

# ─── Problem 2: ablation_old/ 下 5 个实验 ──────────────────
ABLATION_OLD_EXPERIMENTS = [
    'work_dirs/ablation_old/stochot_eps1',
    'work_dirs/ablation_old/stoch_eps05_seed123',
    'work_dirs/ablation_old/nonlinear_trajectory_e43_eps2',
    'work_dirs/ablation_old/nonlinear_trajectory_e43_eps3',
    'work_dirs/ablation_old/reproduce_0751_stochot_eps5_v2',
]


# ═══════════════════════════════════════════════════════════
# 增强策略分类 (复用 build_experiment_db.py 逻辑)
# ═══════════════════════════════════════════════════════════

def classify_augmentation_from_steps(step_types):
    """根据 pipeline 步骤类型分类增强策略并计算 hash.

    Returns:
        dict: pipeline_hash, pipeline_name, has_*, step_types_json, description
    """
    has_random_choice_resize = False
    has_random_crop = False
    has_random_flip = False
    has_fixed_resize = False

    for st in step_types:
        if st == 'RandomFlip':
            has_random_flip = True
        elif st == 'Resize':
            has_fixed_resize = True
        elif st == 'RandomChoiceResize':
            has_random_choice_resize = True
        elif st == 'RandomCrop':
            has_random_crop = True
        elif st == 'RandomChoice':
            # RandomChoice 包含子 transforms (RandomChoiceResize, RandomCrop)
            has_random_choice_resize = True  # 通常包含多尺度 Resize

    if has_random_choice_resize or has_random_crop:
        pipeline_name = 'standard'
        desc = '标准增强 (RandomFlip + RandomChoiceResize + RandomCrop)'
    elif has_fixed_resize and has_random_flip and not has_random_choice_resize:
        pipeline_name = 'simple'
        desc = '简单增强 (Resize + RandomFlip)'
    else:
        pipeline_name = 'custom'
        desc = f'自定义增强 ({", ".join(step_types)})'

    pipeline_hash = hashlib.md5('|'.join(step_types).encode()).hexdigest()[:16]

    return {
        'pipeline_hash': pipeline_hash,
        'pipeline_name': pipeline_name,
        'has_random_choice_resize': has_random_choice_resize,
        'has_random_crop': has_random_crop,
        'has_random_flip': has_random_flip,
        'has_fixed_resize': has_fixed_resize,
        'step_types_json': json.dumps(step_types),
        'description': desc,
    }


def extract_train_pipeline_steps_from_cfg_str(cfg_str):
    """从配置文本中提取 train_pipeline 的顶层步骤类型列表.

    只提取列表中直接子元素 (dict) 的 type 字段, 不递归进 transforms/scales 等嵌套结构.
    通过深度跟踪实现: depth=1 为列表内部, depth=2 为顶层 dict 内部, depth>=3 为嵌套.

    Returns:
        list[str]: 顶层步骤类型列表, 如 ['LoadImageFromFile', 'LoadAnnotations', ...]
    """
    # 定位 train_pipeline = [ 开始位置
    idx = cfg_str.find('train_pipeline = [')
    if idx < 0:
        idx = cfg_str.find('train_pipeline=[')
    if idx < 0:
        return []

    # 找到匹配的闭合方括号 (同时跟踪 [] 和 ())
    depth = 0
    start = cfg_str.find('[', idx)
    if start < 0:
        return []
    end = start
    for i in range(start, len(cfg_str)):
        if cfg_str[i] in '[(':
            depth += 1
        elif cfg_str[i] in '])':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    pipeline_section = cfg_str[start:end]

    # 遍历字符, 跟踪深度, 只在 depth==2 (顶层 dict 内部) 时提取 type='XXX'
    # depth=1: 列表内部 (顶层 dict 之间)
    # depth=2: 顶层 dict 内部 (type= 在这里)
    # depth>=3: 嵌套结构 (transforms=[], scales=[] 等)
    step_types = []
    depth = 0
    i = 0
    while i < len(pipeline_section):
        c = pipeline_section[i]
        if c in '[(':
            depth += 1
            i += 1
        elif c in '])':
            depth -= 1
            i += 1
        elif depth == 2 and pipeline_section[i:i+6] == "type='":
            # 找到 type='...' 在顶层 dict 中
            end_q = pipeline_section.find("'", i + 6)
            if end_q > 0:
                t = pipeline_section[i + 6:end_q]
                step_types.append(t)
                i = end_q + 1
            else:
                i += 1
        elif depth == 2 and pipeline_section[i:i+6] == 'type="':
            end_q = pipeline_section.find('"', i + 6)
            if end_q > 0:
                t = pipeline_section[i + 6:end_q]
                step_types.append(t)
                i = end_q + 1
            else:
                i += 1
        else:
            i += 1

    return step_types


# ═══════════════════════════════════════════════════════════
# Problem 1: 解析 ablation/ 配置文件
# ═══════════════════════════════════════════════════════════

def parse_ablation_config(config_path):
    """解析 ablation/ 下的配置文件 (ot_ 格式).

    这些配置使用 ot_coupling=True, ot_epsilon=X, ot_matcher='sinkhorn', ot_sample=True
    直接在 bbox_head 下, 而非 coupling=dict(...) 格式.

    Returns:
        dict: coupling_type, ot_epsilon, ot_matcher, ot_sample, coupling_mode,
              time_conditioning, data_root
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取 ot_ 字段 (直接在 bbox_head 下)
    ot_coupling_m = re.search(r'ot_coupling\s*=\s*(True|False)', content)
    ot_epsilon_m = re.search(r'ot_epsilon\s*=\s*([\d.]+)', content)
    ot_matcher_m = re.search(r"ot_matcher\s*=\s*'([^']+)'", content)
    ot_sample_m = re.search(r'ot_sample\s*=\s*(True|False)', content)
    time_cond_m = re.search(r"time_conditioning\s*=\s*'([^']+)'", content)
    data_root_m = re.search(r"data_root\s*=\s*'([^']+)'", content)

    ot_coupling = ot_coupling_m.group(1) == 'True' if ot_coupling_m else False
    ot_sample = ot_sample_m.group(1) == 'True' if ot_sample_m else None

    # coupling_type 映射: ot_coupling=True → 'ot_flow'
    coupling_type = 'ot_flow' if ot_coupling else 'random'
    # coupling_mode: ot_sample=True → 'multinomial'
    coupling_mode = 'multinomial' if ot_sample else None

    return {
        'coupling_type': coupling_type,
        'ot_epsilon': float(ot_epsilon_m.group(1)) if ot_epsilon_m else None,
        'ot_matcher': ot_matcher_m.group(1) if ot_matcher_m else None,
        'ot_sample': ot_sample,
        'coupling_mode': coupling_mode,
        'time_conditioning': time_cond_m.group(1) if time_cond_m else 'none',
        'data_root': data_root_m.group(1) if data_root_m else None,
    }


def fix_problem1(conn, dry_run=False):
    """修复 Problem 1: ablation/ 下 6 个实验的配置标注.

    更新 config 表的 coupling_type, ot_epsilon, ot_matcher, ot_sample, coupling_mode 字段.
    """
    print('\n' + '=' * 70)
    print('问题 1: 修复 ablation/ 下 6 个实验的 coupling/ε/ot_matcher/ot_sample 标注')
    print('=' * 70)

    results = []

    for work_dir, config_name in ABLATION_EXPERIMENTS:
        config_path_abs = os.path.join(ROSS_ROOT, work_dir, config_name)
        experiment_id = work_dir

        # 读取旧值
        cur = conn.cursor()
        cur.execute('SELECT * FROM config WHERE config_path = ?', (config_path_abs,))
        old_cfg = cur.fetchone()
        if not old_cfg:
            print(f'  [WARN] DB 中未找到 config_path={config_path_abs}, 跳过')
            results.append({'experiment_id': experiment_id, 'status': 'config_not_found'})
            continue

        old_cfg = dict(old_cfg)
        old_coupling = old_cfg['coupling_type']
        old_epsilon = old_cfg['ot_epsilon']
        old_matcher = old_cfg['ot_matcher']
        old_sample = old_cfg['ot_sample']
        old_mode = old_cfg['coupling_mode']

        # 解析配置文件
        if not os.path.isfile(config_path_abs):
            print(f'  [WARN] 配置文件不存在: {config_path_abs}, 跳过')
            results.append({'experiment_id': experiment_id, 'status': 'file_not_found'})
            continue

        parsed = parse_ablation_config(config_path_abs)

        new_coupling = parsed['coupling_type']
        new_epsilon = parsed['ot_epsilon']
        new_matcher = parsed['ot_matcher']
        new_sample = parsed['ot_sample']
        new_mode = parsed['coupling_mode']

        # 检查是否有变化
        changed = (
            old_coupling != new_coupling or
            old_epsilon != new_epsilon or
            old_matcher != new_matcher or
            old_sample != new_sample or
            old_mode != new_mode
        )

        results.append({
            'experiment_id': experiment_id,
            'status': 'fixed' if changed else 'no_change',
            'old': {
                'coupling_type': old_coupling,
                'ot_epsilon': old_epsilon,
                'ot_matcher': old_matcher,
                'ot_sample': old_sample,
                'coupling_mode': old_mode,
            },
            'new': {
                'coupling_type': new_coupling,
                'ot_epsilon': new_epsilon,
                'ot_matcher': new_matcher,
                'ot_sample': new_sample,
                'coupling_mode': new_mode,
            },
        })

        if changed and not dry_run:
            cur.execute('''
                UPDATE config
                SET coupling_type = ?, ot_epsilon = ?, ot_matcher = ?,
                    ot_sample = ?, coupling_mode = ?
                WHERE config_path = ?
            ''', (new_coupling, new_epsilon, new_matcher, new_sample, new_mode, config_path_abs))
            conn.commit()

    # 打印结果
    print(f'\n{"实验":<45} {"字段":<18} {"旧值":<20} → {"新值"}')
    print('-' * 110)
    for r in results:
        if r['status'] in ('config_not_found', 'file_not_found'):
            print(f'  {r["experiment_id"]}: {r["status"]}')
            continue
        exp_short = r['experiment_id'].replace('work_dirs/ablation/', '')
        old = r['old']
        new = r['new']
        for field in ['coupling_type', 'ot_epsilon', 'ot_matcher', 'ot_sample', 'coupling_mode']:
            o = old[field]
            n = new[field]
            if o != n:
                print(f'  {exp_short:<43} {field:<18} {str(o):<20} → {n}')
        if r['status'] == 'no_change':
            print(f'  (无变化)')

    return results


# ═══════════════════════════════════════════════════════════
# Problem 2: 从 checkpoint 提取 mAP 和配置
# ═══════════════════════════════════════════════════════════

def parse_cfg_string(cfg_str):
    """从 checkpoint 中的 cfg 字符串提取配置信息.

    支持两种格式:
    1. coupling=dict(type='ot_flow', epsilon=2.0, coupling_mode='multinomial') (旧格式)
    2. ot_coupling=True, ot_epsilon=5.0, ot_matcher='sinkhorn', ot_sample=True (新格式)
    """
    coupling_type = 'random'
    ot_epsilon = None
    ot_matcher = None
    ot_sample = None
    coupling_mode = None
    time_conditioning = 'none'
    data_root = ''
    solver_type = 'unknown'
    rf_schedule = None
    rf_shift = None
    num_proposals = None
    sampling_timesteps = None
    num_classes = None
    batch_size = None
    max_epochs = None

    # 尝试 coupling=dict(...) 格式
    coupling_match = re.search(r'coupling\s*=\s*dict\(([^)]+)\)', cfg_str, re.DOTALL)
    if coupling_match:
        coupling_str = coupling_match.group(1)
        ct = re.search(r"type\s*=\s*'([^']+)'", coupling_str)
        eps = re.search(r'epsilon\s*=\s*([\d.]+)', coupling_str)
        cm = re.search(r"coupling_mode\s*=\s*'([^']+)'", coupling_str)
        if ct:
            coupling_type = ct.group(1)
        if eps:
            ot_epsilon = float(eps.group(1))
        if cm:
            coupling_mode = cm.group(1)
        # sinkhorn_stochastic 隐含 ot_sample=True
        if coupling_type == 'sinkhorn_stochastic':
            ot_sample = True
            coupling_mode = coupling_mode or 'multinomial'
    else:
        # 尝试 ot_ 格式
        ot_coupling_m = re.search(r'ot_coupling\s*=\s*(True|False)', cfg_str)
        ot_epsilon_m = re.search(r'ot_epsilon\s*=\s*([\d.]+)', cfg_str)
        ot_matcher_m = re.search(r"ot_matcher\s*=\s*'([^']+)'", cfg_str)
        ot_sample_m = re.search(r'ot_sample\s*=\s*(True|False)', cfg_str)

        if ot_coupling_m and ot_coupling_m.group(1) == 'True':
            coupling_type = 'ot_flow'
        if ot_epsilon_m:
            ot_epsilon = float(ot_epsilon_m.group(1))
        if ot_matcher_m:
            ot_matcher = ot_matcher_m.group(1)
        if ot_sample_m:
            ot_sample = ot_sample_m.group(1) == 'True'
            if ot_sample:
                coupling_mode = 'multinomial'

    # time_conditioning
    tc_m = re.search(r"time_conditioning\s*=\s*'([^']+)'", cfg_str)
    if tc_m:
        time_conditioning = tc_m.group(1)

    # data_root (取第一个出现, 通常是全局或 train_dataloader 中的)
    dr_m = re.search(r"data_root\s*=\s*'([^']+)'", cfg_str)
    if dr_m:
        data_root = dr_m.group(1)

    # solver_type
    st_m = re.search(r"solver_type\s*=\s*'([^']+)'", cfg_str)
    if st_m:
        solver_type = st_m.group(1)

    # rf_schedule, rf_shift
    rs_m = re.search(r"rf_schedule\s*=\s*'([^']+)'", cfg_str)
    if rs_m:
        rf_schedule = rs_m.group(1)
    rsh_m = re.search(r'rf_shift\s*=\s*([\d.]+)', cfg_str)
    if rsh_m:
        rf_shift = float(rsh_m.group(1))

    # num_proposals, sampling_timesteps, num_classes
    np_m = re.search(r'num_proposals\s*=\s*(\d+)', cfg_str)
    if np_m:
        num_proposals = int(np_m.group(1))
    st_m = re.search(r'sampling_timesteps\s*=\s*(\d+)', cfg_str)
    if st_m:
        sampling_timesteps = int(st_m.group(1))
    nc_m = re.search(r'num_classes\s*=\s*(\d+)', cfg_str)
    if nc_m:
        num_classes = int(nc_m.group(1))

    # batch_size, max_epochs
    bs_m = re.search(r'batch_size\s*=\s*(\d+)', cfg_str)
    if bs_m:
        batch_size = int(bs_m.group(1))
    me_m = re.search(r'max_epochs\s*=\s*(\d+)', cfg_str)
    if me_m:
        max_epochs = int(me_m.group(1))

    # 数据集识别
    dataset = 'unknown'
    if D1_DATA_ROOT_PATTERN in data_root:
        dataset = 'D1'
    elif D2_DATA_ROOT_PATTERN in data_root:
        dataset = 'D2'

    # train_pipeline
    step_types = extract_train_pipeline_steps_from_cfg_str(cfg_str)

    # EarlyStopping
    has_early_stopping = 'EarlyStoppingHook' in cfg_str

    return {
        'coupling_type': coupling_type,
        'ot_epsilon': ot_epsilon,
        'ot_matcher': ot_matcher,
        'ot_sample': ot_sample,
        'coupling_mode': coupling_mode,
        'time_conditioning': time_conditioning,
        'data_root': data_root,
        'dataset': dataset,
        'solver_type': solver_type,
        'rf_schedule': rf_schedule,
        'rf_shift': rf_shift,
        'num_proposals': num_proposals,
        'sampling_timesteps': sampling_timesteps,
        'num_classes': num_classes,
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'has_early_stopping': has_early_stopping,
        'train_pipeline_steps': step_types,
    }


def extract_from_checkpoint(ckpt_path):
    """从 checkpoint 提取 best_score, per-epoch mAP, 配置信息.

    Returns:
        dict: best_mAP, best_epoch (from filename), per_epoch_mAP, best_detail, config, seed
    """
    import torch
    import numpy as np

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ri = ckpt['message_hub']['runtime_info']
    log_scalars = ckpt['message_hub']['log_scalars']

    # best_score (来自 runtime_info)
    best_score = ri.get('best_score')

    # best_epoch 从 checkpoint 文件名提取 (最可靠)
    fname_epoch_m = re.search(r'best_coco_bbox_mAP_epoch_(\d+)\.pth', os.path.basename(ckpt_path))
    best_epoch_from_fname = int(fname_epoch_m.group(1)) if fname_epoch_m else None

    # per-epoch mAP (来自 HistoryBuffer._log_history)
    per_epoch_mAP = []
    best_mAP = best_score

    mAP_buf = log_scalars.get('val/coco/bbox_mAP')
    if mAP_buf is not None and hasattr(mAP_buf, '_log_history'):
        hist = mAP_buf._log_history
        if len(hist) > 0:
            per_epoch_mAP = hist.tolist()
            best_idx = int(np.argmax(hist))
            best_mAP = float(hist[best_idx])

    # 如果 best_score 和 per-epoch max 不一致, 优先 best_score (更可靠)
    if best_score is not None and per_epoch_mAP:
        if abs(best_score - best_mAP) > 0.0005:
            best_mAP = best_score

    # best epoch 详细指标: 使用 HistoryBuffer 中最接近 best_mAP 的 epoch
    best_detail = {}
    best_epoch_hist = None
    if per_epoch_mAP:
        best_idx = int(np.argmax(per_epoch_mAP))
        best_epoch_hist = best_idx + 1  # 1-indexed
        for key, label in [
            ('val/coco/bbox_mAP_50', 'AP50'),
            ('val/coco/bbox_mAP_75', 'AP75'),
            ('val/coco/bbox_mAP_s', 'AP_small'),
            ('val/coco/bbox_mAP_m', 'AP_medium'),
            ('val/coco/bbox_mAP_l', 'AP_large'),
        ]:
            buf = log_scalars.get(key)
            if buf is not None and hasattr(buf, '_log_history'):
                hist = buf._log_history
                if best_idx < len(hist):
                    best_detail[label] = float(hist[best_idx])

    # best_epoch: 优先使用 checkpoint 文件名中的 epoch (最可靠)
    best_epoch = best_epoch_from_fname or best_epoch_hist

    best_detail['mAP'] = best_mAP
    best_detail['epoch'] = best_epoch

    # 配置信息 (从 meta.cfg 字符串)
    cfg_str = ckpt['meta'].get('cfg', '')
    config_info = parse_cfg_string(cfg_str) if cfg_str else None

    # seed
    seed = ri.get('seed')

    return {
        'best_mAP': best_mAP,
        'best_epoch': best_epoch,
        'per_epoch_mAP': per_epoch_mAP,
        'best_detail': best_detail,
        'config': config_info,
        'seed': seed,
    }


def upsert_aug_pipeline(conn, aug_info):
    """插入或更新增强策略 (如果不存在)."""
    conn.execute('''
        INSERT OR IGNORE INTO aug_pipeline
            (pipeline_hash, pipeline_name, has_random_choice_resize, has_random_crop,
             has_random_flip, has_fixed_resize, step_types_json, description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        aug_info['pipeline_hash'],
        aug_info['pipeline_name'],
        aug_info['has_random_choice_resize'],
        aug_info['has_random_crop'],
        aug_info['has_random_flip'],
        aug_info['has_fixed_resize'],
        aug_info['step_types_json'],
        aug_info['description'],
    ))


def fix_problem2(conn, dry_run=False):
    """修复 Problem 2: ablation_old/ 下 5 个实验的 val mAP 缺失.

    从 checkpoint 提取 best_score 和 per-epoch mAP, 更新 experiment 表.
    同时从 checkpoint meta.cfg 提取配置, 创建/更新 config 表.

    处理逻辑:
    - best_val_mAP: 仅在为 None 时更新 (不覆盖已有值)
    - best_val_epoch: 不修改 (已有值来自 checkpoint 文件名, 更可靠)
    - config: 仅在 config_path 为 NULL 时创建新记录
    - evaluation: 如果不存在则插入
    - stability: 如果不存在则插入
    """
    import glob

    print('\n' + '=' * 70)
    print('问题 2: 修复 ablation_old/ 下 5 个实验的 val mAP 缺失')
    print('=' * 70)

    results = []

    for work_dir in ABLATION_OLD_EXPERIMENTS:
        experiment_id = work_dir
        abs_work_dir = os.path.join(ROSS_ROOT, work_dir)

        # 查找 best checkpoint
        ckpt_pattern = os.path.join(abs_work_dir, 'best_coco_bbox_mAP_epoch_*.pth')
        ckpts = glob.glob(ckpt_pattern)
        if not ckpts:
            print(f'  [WARN] 未找到 checkpoint: {work_dir}, 跳过')
            results.append({'experiment_id': experiment_id, 'status': 'no_ckpt'})
            continue

        # 取 epoch 最大的 checkpoint
        def extract_epoch(path):
            m = re.search(r'best_coco_bbox_mAP_epoch_(\d+)\.pth', os.path.basename(path))
            return int(m.group(1)) if m else 0

        ckpt_path = max(ckpts, key=extract_epoch)

        # 读取旧值
        cur = conn.cursor()
        cur.execute('SELECT * FROM experiment WHERE experiment_id = ?', (experiment_id,))
        old_exp = cur.fetchone()
        if not old_exp:
            print(f'  [WARN] DB 中未找到 experiment_id={experiment_id}, 跳过')
            results.append({'experiment_id': experiment_id, 'status': 'exp_not_found'})
            continue

        old_exp = dict(old_exp)
        old_val_mAP = old_exp['best_val_mAP']
        old_val_epoch = old_exp['best_val_epoch']
        old_config_path = old_exp['config_path']

        # 从 checkpoint 提取数据
        try:
            data = extract_from_checkpoint(ckpt_path)
        except Exception as e:
            print(f'  [ERROR] 提取 checkpoint 数据失败 {work_dir}: {e}')
            results.append({'experiment_id': experiment_id, 'status': 'extract_error'})
            continue

        ckpt_val_mAP = data['best_mAP']
        ckpt_val_epoch = data['best_epoch']
        config_info = data['config']

        # 决定是否更新 best_val_mAP (仅在 None 时更新)
        mAP_changed = False
        new_val_mAP = old_val_mAP
        if old_val_mAP is None and ckpt_val_mAP is not None:
            new_val_mAP = ckpt_val_mAP
            mAP_changed = True
            if not dry_run:
                cur.execute('''
                    UPDATE experiment SET best_val_mAP = ? WHERE experiment_id = ?
                ''', (new_val_mAP, experiment_id))
                conn.commit()

        # best_val_epoch: 如果为 None 则从 checkpoint 补充
        epoch_changed = False
        new_val_epoch = old_val_epoch
        if old_val_epoch is None and ckpt_val_epoch is not None:
            new_val_epoch = ckpt_val_epoch
            epoch_changed = True
            if not dry_run:
                cur.execute('''
                    UPDATE experiment SET best_val_epoch = ? WHERE experiment_id = ?
                ''', (new_val_epoch, experiment_id))
                conn.commit()

        # 验证已有值与 checkpoint 值是否一致
        mAP_match = True
        if old_val_mAP is not None and ckpt_val_mAP is not None:
            if abs(old_val_mAP - ckpt_val_mAP) > 0.001:
                mAP_match = False

        # 创建 config 表 (仅在 config_path 为 NULL 时)
        config_status = 'config_exists'
        if old_config_path is None and config_info:
            # 使用 work_dir 下的合成 config_path (配置从 checkpoint 提取)
            config_name = os.path.basename(work_dir) + '.py'
            config_path = os.path.join(abs_work_dir, config_name)

            # 分类增强策略
            step_types = config_info.get('train_pipeline_steps', [])
            aug_info = classify_augmentation_from_steps(step_types) if step_types else None

            if aug_info and not dry_run:
                upsert_aug_pipeline(conn, aug_info)
                conn.commit()

            if not dry_run:
                cur.execute('''
                    INSERT INTO config
                        (config_path, source_config_path, dataset, data_root, aug_pipeline_hash,
                         coupling_type, ot_epsilon, ot_matcher, ot_sample, coupling_mode,
                         time_conditioning, solver_type, rf_schedule, rf_shift,
                         batch_size, max_epochs, num_classes, num_proposals, sampling_timesteps,
                         has_early_stopping)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    config_path,
                    f'<extracted_from_checkpoint:{os.path.basename(ckpt_path)}>',
                    config_info['dataset'],
                    config_info['data_root'],
                    aug_info['pipeline_hash'] if aug_info else None,
                    config_info['coupling_type'],
                    config_info['ot_epsilon'],
                    config_info['ot_matcher'],
                    config_info['ot_sample'],
                    config_info['coupling_mode'],
                    config_info['time_conditioning'],
                    config_info['solver_type'],
                    config_info['rf_schedule'],
                    config_info['rf_shift'],
                    config_info['batch_size'],
                    config_info['max_epochs'],
                    config_info['num_classes'],
                    config_info['num_proposals'],
                    config_info['sampling_timesteps'],
                    config_info['has_early_stopping'],
                ))
                conn.commit()

                cur.execute('''
                    UPDATE experiment SET config_path = ? WHERE experiment_id = ?
                ''', (config_path, experiment_id))
                conn.commit()
                config_status = 'config_created'
            else:
                config_status = 'config_would_create'

        # 插入 evaluation 记录 (如果不存在)
        eval_status = 'eval_exists'
        if not dry_run:
            cur.execute('''
                SELECT COUNT(*) FROM evaluation
                WHERE experiment_id = ? AND split = 'val'
            ''', (experiment_id,))
            eval_count = cur.fetchone()[0]

            if eval_count == 0:
                best_detail = data['best_detail']
                cur.execute('''
                    INSERT INTO evaluation
                        (experiment_id, split, mAP, AP50, AP75, AP_small, AP_medium, AP_large,
                         epoch, checkpoint_path, evaluated_at, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    experiment_id,
                    'val',
                    best_detail.get('mAP'),
                    best_detail.get('AP50'),
                    best_detail.get('AP75'),
                    best_detail.get('AP_small'),
                    best_detail.get('AP_medium'),
                    best_detail.get('AP_large'),
                    best_detail.get('epoch'),
                    old_exp['best_checkpoint'],
                    None,
                    'checkpoint_meta',
                ))
                conn.commit()
                eval_status = 'eval_created'
        else:
            eval_status = 'eval_would_check'

        # 插入 stability 记录 (如果不存在且有 per-epoch mAP)
        stability_status = 'no_data'
        if data['per_epoch_mAP'] and len(data['per_epoch_mAP']) >= 2:
            if not dry_run:
                cur.execute('''
                    SELECT COUNT(*) FROM stability WHERE experiment_id = ?
                ''', (experiment_id,))
                stab_count = cur.fetchone()[0]

                if stab_count == 0:
                    per_epoch = data['per_epoch_mAP']
                    last_30 = per_epoch[-30:]
                    n = len(last_30)
                    mean_val = sum(last_30) / n
                    variance = sum((x - mean_val) ** 2 for x in last_30) / n
                    std = variance ** 0.5
                    cv = std / mean_val if mean_val > 0 else None
                    range_val = max(last_30) - min(last_30)
                    best_mAP = max(per_epoch)
                    threshold = best_mAP * 0.99
                    best_1pct_count = sum(1 for x in last_30 if x >= threshold)

                    cur.execute('''
                        INSERT OR REPLACE INTO stability
                            (experiment_id, last_30_std, last_30_cv, last_30_range,
                             best_1pct_count, total_epochs, per_epoch_mAP_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        experiment_id,
                        std,
                        cv,
                        range_val,
                        best_1pct_count,
                        len(per_epoch),
                        json.dumps(per_epoch),
                    ))
                    conn.commit()
                    stability_status = 'stability_created'
                else:
                    stability_status = 'stability_exists'
            else:
                stability_status = 'stability_would_check'

        results.append({
            'experiment_id': experiment_id,
            'status': 'fixed' if mAP_changed else ('verified' if mAP_match else 'mismatch'),
            'old_val_mAP': old_val_mAP,
            'new_val_mAP': new_val_mAP,
            'ckpt_val_mAP': ckpt_val_mAP,
            'old_val_epoch': old_val_epoch,
            'new_val_epoch': new_val_epoch,
            'mAP_match': mAP_match,
            'config_status': config_status,
            'eval_status': eval_status,
            'stability_status': stability_status,
            'config_info': config_info,
            'per_epoch_count': len(data['per_epoch_mAP']),
        })

    # 打印结果
    print(f'\n{"实验":<50} {"字段":<16} {"DB旧值":<12} {"ckpt值":<12} {"操作"}')
    print('-' * 110)
    for r in results:
        if r['status'] in ('no_ckpt', 'exp_not_found', 'extract_error'):
            print(f'  {r["experiment_id"]}: {r["status"]}')
            continue

        exp_short = r['experiment_id'].replace('work_dirs/ablation_old/', '')
        old_v = r['old_val_mAP']
        ckpt_v = r['ckpt_val_mAP']
        new_v = r['new_val_mAP']

        old_str = f'{old_v:.3f}' if old_v is not None else 'None'
        ckpt_str = f'{ckpt_v:.3f}' if ckpt_v is not None else 'N/A'

        if r['status'] == 'fixed':
            action = f'已更新 → {new_v:.4f}'
        elif r['status'] == 'verified':
            action = '已验证一致'
        else:
            action = f'不一致! (DB={old_v}, ckpt={ckpt_v})'

        print(f'  {exp_short:<48} best_val_mAP   {old_str:<12} {ckpt_str:<12} {action}')

        ci = r.get('config_info')
        if ci:
            aug_steps = ci.get('train_pipeline_steps', [])
            aug_name = classify_augmentation_from_steps(aug_steps)['pipeline_name'] if aug_steps else 'unknown'
            print(f'  {"":<48} coupling_type  {"":<12} {ci["coupling_type"]:<12} (ckpt config)')
            print(f'  {"":<48} ot_epsilon     {"":<12} {str(ci["ot_epsilon"]):<12} (ckpt config)')
            print(f'  {"":<48} aug_pipeline   {"":<12} {aug_name:<12} ({len(aug_steps)} steps)')
            print(f'  {"":<48} config_status  {"":<24} {r["config_status"]}')
            print(f'  {"":<48} eval_status    {"":<24} {r["eval_status"]}')
            print(f'  {"":<48} stability      {"":<24} {r["stability_status"]}')
            print(f'  {"":<48} per_epoch_mAPs {"":<24} {r["per_epoch_count"]}')

    return results


# ═══════════════════════════════════════════════════════════
# 验证报告
# ═══════════════════════════════════════════════════════════

def print_verification_report(conn, p1_results, p2_results):
    """打印验证报告: 修复前后的对比."""
    print('\n')
    print('╔' + '═' * 68 + '╗')
    print('║' + ' 验证报告: 修复前后对比 '.center(68) + '║')
    print('╠' + '═' * 68 + '╣')

    # Problem 1
    print('║ ' + '问题 1: ablation/ 配置标注修复'.ljust(67) + '║')
    print('║' + '─' * 68 + '║')
    print('║ {:<42} {:<10} {:<6} {:<8}'.format(
        '实验', 'coupling', 'ε', 'aug').ljust(68) + '║')
    print('║' + '─' * 68 + '║')

    for r in p1_results:
        if r['status'] in ('config_not_found', 'file_not_found'):
            continue
        exp = r['experiment_id'].replace('work_dirs/ablation/', '')
        old = r['old']
        new = r['new']
        # aug 没变化 (始终是 standard), 只报告 coupling/ε
        line = f'║ {exp:<42} {old["coupling_type"]:>5}→{new["coupling_type"]:<5} {str(old["ot_epsilon"]):>3}→{str(new["ot_epsilon"]):<3}'
        print(line.ljust(69) + '║')

    # Problem 2
    print('║' + '─' * 68 + '║')
    print('║ ' + '问题 2: ablation_old/ val mAP 修复'.ljust(67) + '║')
    print('║' + '─' * 68 + '║')
    print('║ {:<42} {:<10} {:<10} {:<8}'.format(
        '实验', 'DB mAP', 'ckpt mAP', '状态').ljust(68) + '║')
    print('║' + '─' * 68 + '║')

    for r in p2_results:
        if r['status'] in ('no_ckpt', 'exp_not_found', 'extract_error'):
            continue
        exp = r['experiment_id'].replace('work_dirs/ablation_old/', '')
        old_v = f'{r["old_val_mAP"]:.3f}' if r['old_val_mAP'] is not None else 'None'
        ckpt_v = f'{r["ckpt_val_mAP"]:.3f}' if r['ckpt_val_mAP'] is not None else 'N/A'

        if r['status'] == 'fixed':
            status = '已修复'
        elif r['status'] == 'verified':
            status = '已验证'
        else:
            status = '不一致!'

        line = f'║ {exp:<42} {old_v:<10} {ckpt_v:<10} {status:<8}'
        print(line.ljust(69) + '║')

    # ε=0.5 特别报告
    print('║' + '─' * 68 + '║')
    print('║ ' + '关键数据: ε=0.5 实验 (stoch_eps05_seed123)'.ljust(67) + '║')
    print('║' + '─' * 68 + '║')
    for r in p2_results:
        if 'stoch_eps05' in r['experiment_id'] and r['status'] not in ('no_ckpt', 'exp_not_found', 'extract_error'):
            ci = r.get('config_info', {})
            line1 = f'║   mAP = {r["ckpt_val_mAP"]:.4f} (epoch {r["new_val_epoch"]})'
            line2 = f'║   coupling = {ci.get("coupling_type")}, ε = {ci.get("ot_epsilon")}'
            print(line1.ljust(69) + '║')
            print(line2.ljust(69) + '║')

    print('╚' + '═' * 68 + '╝')


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='修复 ε 消融实验数据库')
    parser.add_argument('--db', default=DB_PATH, help='SQLite 数据库路径')
    parser.add_argument('--dry-run', action='store_true', help='只打印不写入')
    args = parser.parse_args()

    print(f'数据库: {args.db}')
    print(f'ROSS 根目录: {ROSS_ROOT}')
    print(f'模式: {"dry-run (只打印)" if args.dry_run else "写入"}')

    if not os.path.isfile(args.db):
        print(f'[ERROR] 数据库不存在: {args.db}')
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # 修复 Problem 1
    p1_results = fix_problem1(conn, dry_run=args.dry_run)

    # 修复 Problem 2
    p2_results = fix_problem2(conn, dry_run=args.dry_run)

    # 验证报告
    print_verification_report(conn, p1_results, p2_results)

    # 总结
    p1_fixed = sum(1 for r in p1_results if r['status'] == 'fixed')
    p1_no_change = sum(1 for r in p1_results if r['status'] == 'no_change')
    p2_fixed = sum(1 for r in p2_results if r['status'] == 'fixed')
    p2_verified = sum(1 for r in p2_results if r['status'] == 'verified')
    p2_mismatch = sum(1 for r in p2_results if r['status'] == 'mismatch')

    print(f'\n=== 总结 ===')
    print(f'问题 1: {p1_fixed} 个实验配置已修复, {p1_no_change} 个无变化')
    print(f'问题 2: {p2_fixed} 个实验 val mAP 已修复, {p2_verified} 个已验证一致, {p2_mismatch} 个不一致')

    if args.dry_run:
        print('\n[dry-run 模式] 未实际写入数据库. 去掉 --dry-run 执行实际修复.')

    conn.close()


if __name__ == '__main__':
    main()
