"""构建实验数据库: 遍历 work_dirs, 解析 dumped configs + scalars.json, 写入 SQLite.

用法:
  python tools/experiment_db/build_experiment_db.py                    # 扫描本地 work_dirs/
  python tools/experiment_db/build_experiment_db.py --server ross      -- 指定服务器标签
  python tools/experiment_db/build_experiment_db.py --db /path/to.db   -- 指定数据库路径
  python tools/experiment_db/build_experiment_db.py --verbose          -- 详细输出

依赖: mmengine, mmdet (conda activate chromo)
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# 确保项目根目录在 Python path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ─── 数据集路径模式 ─────────────────────────────────────────
D1_DATA_ROOT_PATTERN = 'Chromosome20240904'
D2_DATA_ROOT_PATTERN = '24_chromosomes_object'


# ─── 增强策略分类 ───────────────────────────────────────────

def classify_augmentation(train_pipeline):
    """分类数据增强策略.

    Returns:
        dict: {
            'pipeline_hash': str,
            'pipeline_name': 'standard' | 'simple' | 'custom',
            'has_random_choice_resize': bool,
            'has_random_crop': bool,
            'has_random_flip': bool,
            'has_fixed_resize': bool,
            'step_types': list[str],
            'description': str,
        }
    """
    if not train_pipeline or not isinstance(train_pipeline, (list, tuple)):
        return _unknown_augmentation()

    step_types = []
    has_random_choice_resize = False
    has_random_crop = False
    has_random_flip = False
    has_fixed_resize = False

    for step in train_pipeline:
        if not isinstance(step, dict):
            continue
        step_type = step.get('type', 'unknown')
        step_types.append(step_type)

        if step_type == 'RandomFlip':
            has_random_flip = True
        elif step_type == 'Resize':
            has_fixed_resize = True
        elif step_type == 'RandomChoiceResize':
            has_random_choice_resize = True
        elif step_type == 'RandomCrop':
            has_random_crop = True
        elif step_type == 'RandomChoice':
            # RandomChoice 包含子 transforms, 递归检查
            transforms = step.get('transforms', [])
            if isinstance(transforms, list):
                for sub in transforms:
                    if isinstance(sub, list):
                        for s in sub:
                            if isinstance(s, dict):
                                t = s.get('type', '')
                                if t == 'RandomChoiceResize':
                                    has_random_choice_resize = True
                                elif t == 'RandomCrop':
                                    has_random_crop = True
                    elif isinstance(sub, dict):
                        t = sub.get('type', '')
                        if t == 'RandomChoiceResize':
                            has_random_choice_resize = True
                        elif t == 'RandomCrop':
                            has_random_crop = True

    # 分类
    if has_random_choice_resize or has_random_crop:
        pipeline_name = 'standard'
        desc = '标准增强 (RandomFlip + RandomChoiceResize + RandomCrop)'
    elif has_fixed_resize and has_random_flip and not has_random_choice_resize:
        pipeline_name = 'simple'
        desc = '简单增强 (Resize + RandomFlip)'
    else:
        pipeline_name = 'custom'
        desc = f'自定义增强 ({", ".join(step_types)})'

    # hash: 基于步骤类型序列
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


def _unknown_augmentation():
    return {
        'pipeline_hash': 'unknown',
        'pipeline_name': 'unknown',
        'has_random_choice_resize': False,
        'has_random_crop': False,
        'has_random_flip': False,
        'has_fixed_resize': False,
        'step_types_json': '[]',
        'description': '未知 (train_pipeline 未找到)',
    }


# ─── 配置解析 ───────────────────────────────────────────────

def parse_dumped_config(config_path):
    """解析 dumped config, 提取模型设置和数据增强策略.

    对旧配置 (引用已迁移的 projects.LDMDet 模块), 预处理移除 custom_imports 后再解析.

    Returns:
        dict or None: 配置信息
    """
    import tempfile

    try:
        from mmengine.config import Config
    except ImportError:
        # Metric/run ingestion remains useful on lightweight maintenance hosts.
        # Existing parsed config rows are preserved by upsert_experiment.
        return None

    try:
        cfg = Config.fromfile(config_path)
    except Exception:
        # 旧配置可能引用已迁移的模块 (projects.LDMDet.*), 预处理移除 custom_imports
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # 移除 custom_imports 行 (避免导入失败)
            content = re.sub(
                r'custom_imports\s*=\s*dict\([^)]*\)',
                'custom_imports = dict(imports=[], allow_failed_imports=True)',
                content,
                flags=re.DOTALL,
            )
            # 写入临时文件
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, dir=_PROJECT_ROOT
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                cfg = Config.fromfile(tmp_path)
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            print(f'  [WARN] 无法解析配置 {config_path}: {e}')
            return None

    # 数据集识别
    data_root = ''
    dataset = 'unknown'
    try:
        train_dataset = cfg.train_dataloader.dataset
        data_root = train_dataset.get('data_root', '')
        if D1_DATA_ROOT_PATTERN in data_root:
            dataset = 'D1'
        elif D2_DATA_ROOT_PATTERN in data_root:
            dataset = 'D2'
    except Exception:
        pass

    # 增强策略
    train_pipeline = cfg.get('train_pipeline', [])
    aug_info = classify_augmentation(train_pipeline)

    # 模型设置
    model = cfg.get('model', {})
    bbox_head = model.get('bbox_head', {})
    coupling = bbox_head.get('coupling', {})
    single_head = bbox_head.get('single_head', {})

    coupling_type = coupling.get('type', 'none') if coupling else 'none'
    # 如果没有 coupling 字段, 默认是 random
    if not coupling:
        coupling_type = 'random'

    ot_epsilon = coupling.get('epsilon', None)
    ot_matcher = coupling.get('matcher', None) or coupling.get('ot_matcher', None)
    ot_sample = coupling.get('sample', None) or coupling.get('ot_sample', None)
    coupling_mode = coupling.get('coupling_mode', None) or coupling.get('mode', None)

    time_conditioning = single_head.get('time_conditioning', 'none')
    solver_type = bbox_head.get('solver_type', 'unknown')
    rf_schedule = bbox_head.get('rf_schedule', None)
    rf_shift = bbox_head.get('rf_shift', None)
    num_proposals = bbox_head.get('num_proposals', None)
    sampling_timesteps = bbox_head.get('sampling_timesteps', None)
    num_classes = bbox_head.get('num_classes', cfg.get('num_classes', None))

    # 训练设置
    batch_size = cfg.train_dataloader.get('batch_size', None) if hasattr(cfg, 'train_dataloader') else None
    max_epochs = cfg.train_cfg.get('max_epochs', None) if hasattr(cfg, 'train_cfg') else None

    # EarlyStopping
    has_early_stopping = False
    custom_hooks = cfg.get('custom_hooks', [])
    if isinstance(custom_hooks, list):
        for hook in custom_hooks:
            if isinstance(hook, dict) and hook.get('type') == 'EarlyStoppingHook':
                has_early_stopping = True
                break

    return {
        'config_path': config_path,
        'source_config_path': '',  # dumped config 本身就是源
        'dataset': dataset,
        'data_root': data_root,
        'aug_pipeline_hash': aug_info['pipeline_hash'],
        'coupling_type': coupling_type,
        'ot_epsilon': ot_epsilon,
        'ot_matcher': ot_matcher,
        'ot_sample': ot_sample,
        'coupling_mode': coupling_mode,
        'time_conditioning': time_conditioning,
        'solver_type': solver_type,
        'rf_schedule': rf_schedule,
        'rf_shift': rf_shift,
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'num_classes': num_classes,
        'num_proposals': num_proposals,
        'sampling_timesteps': sampling_timesteps,
        'has_early_stopping': has_early_stopping,
        '_aug_info': aug_info,  # 内部使用
    }


# ─── 指标提取 ───────────────────────────────────────────────

def find_scalars_json(work_dir):
    """查找 scalars.json 文件."""
    patterns = [
        os.path.join(work_dir, '**', 'vis_data', 'scalars.json'),
        os.path.join(work_dir, 'vis_data', 'scalars.json'),
    ]
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            # 取最新的
            files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            return files[0]
    return None


def extract_metrics_from_scalars(scalars_path):
    """从 scalars.json 提取逐 epoch val mAP.

    Returns:
        dict: {
            'per_epoch_mAP': list[float],
            'best_mAP': float,
            'best_epoch': int,
            'total_epochs': int,
        }
    """
    if not scalars_path or not os.path.isfile(scalars_path):
        return None

    try:
        with open(scalars_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f'  [WARN] 无法读取 scalars.json: {e}')
        return None

    per_epoch_mAP = []
    per_epoch_detail = []  # 包含 AP50, AP75 等

    for line in lines:
        try:
            entry = json.loads(line.strip())
        except json.JSONDecodeError:
            continue

        mAP = entry.get('coco/bbox_mAP')
        if mAP is not None:
            per_epoch_mAP.append(mAP)
            per_epoch_detail.append({
                'mAP': mAP,
                'AP50': entry.get('coco/bbox_mAP_50'),
                'AP75': entry.get('coco/bbox_mAP_75'),
                'AP_small': entry.get('coco/bbox_mAP_s'),
                'AP_medium': entry.get('coco/bbox_mAP_m'),
                'AP_large': entry.get('coco/bbox_mAP_l'),
                'epoch': entry.get('epoch'),
                'step': entry.get('step'),
            })

    if not per_epoch_mAP:
        return None

    best_idx = max(range(len(per_epoch_mAP)), key=lambda i: per_epoch_mAP[i])
    best_mAP = per_epoch_mAP[best_idx]
    best_epoch = per_epoch_detail[best_idx].get('epoch', best_idx + 1)

    return {
        'per_epoch_mAP': per_epoch_mAP,
        'per_epoch_detail': per_epoch_detail,
        'best_mAP': best_mAP,
        'best_epoch': best_epoch,
        'total_epochs': len(per_epoch_mAP),
    }


def find_log_files(work_dir):
    """查找训练日志文件 (train.log + timestamp 子目录下的 .log)."""
    logs = []
    train_log = os.path.join(work_dir, 'train.log')
    if os.path.isfile(train_log):
        logs.append(train_log)
    for item in os.listdir(work_dir):
        subdir = os.path.join(work_dir, item)
        if os.path.isdir(subdir) and re.match(r'\d{8}_\d{6}', item):
            for f in os.listdir(subdir):
                if f.endswith('.log') and f.startswith(item):
                    logs.append(os.path.join(subdir, f))
    return logs


def extract_metrics_from_logs(work_dir, best_epoch_ckpt=None):
    """从训练日志提取 mAP 指标 (scalars.json 不可用时的 fallback).

    提取策略:
      1. "best score: X.XXX" 行 → best mAP
      2. best checkpoint epoch 的 "Epoch(val) [N]" 行 → val mAP at best epoch
      3. 所有 "Epoch(val)" 行 → per-epoch mAP 列表

    Returns:
        dict (同 extract_metrics_from_scalars 格式) 或 None
    """
    log_files = find_log_files(work_dir)
    if not log_files:
        return None

    per_epoch_mAP = []
    per_epoch_detail = []
    best_score_from_log = None

    for log_path in log_files:
        try:
            with open(log_path, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # 提取 "best score: X.XXX"
                    if 'best score:' in line:
                        m = re.search(r'best score:\s*([\d.]+)', line)
                        if m:
                            best_score_from_log = float(m.group(1))

                    # 提取 "Epoch(val) [N][... coco/bbox_mAP: X.XXX"
                    if 'Epoch(val)' in line and 'coco/bbox_mAP:' in line:
                        ep_m = re.search(r'Epoch\(val\)\s*\[(\d+)\]', line)
                        map_m = re.search(r'coco/bbox_mAP:\s*([\d.]+)', line)
                        ap50_m = re.search(r'coco/bbox_mAP_50:\s*([\d.]+)', line)
                        ap75_m = re.search(r'coco/bbox_mAP_75:\s*([\d.]+)', line)
                        aps_m = re.search(r'coco/bbox_mAP_s:\s*([\d.]+)', line)
                        apm_m = re.search(r'coco/bbox_mAP_m:\s*([\d.]+)', line)
                        apl_m = re.search(r'coco/bbox_mAP_l:\s*([\d.]+)', line)

                        if ep_m and map_m:
                            ep = int(ep_m.group(1))
                            mAP = float(map_m.group(1))
                            if mAP > 0:  # 跳过初始 0 值
                                per_epoch_mAP.append(mAP)
                                per_epoch_detail.append({
                                    'mAP': mAP,
                                    'AP50': float(ap50_m.group(1)) if ap50_m else None,
                                    'AP75': float(ap75_m.group(1)) if ap75_m else None,
                                    'AP_small': float(aps_m.group(1)) if aps_m else None,
                                    'AP_medium': float(apm_m.group(1)) if apm_m else None,
                                    'AP_large': float(apl_m.group(1)) if apl_m else None,
                                    'epoch': ep,
                                    'step': None,
                                })
        except Exception:
            continue

    if not per_epoch_mAP and best_score_from_log is None:
        return None

    # 确定最佳 mAP
    if per_epoch_mAP:
        best_idx = max(range(len(per_epoch_mAP)), key=lambda i: per_epoch_mAP[i])
        best_mAP = per_epoch_mAP[best_idx]
        best_epoch = per_epoch_detail[best_idx].get('epoch', best_epoch_ckpt)
    else:
        # 仅有 best score 行, 无逐 epoch 数据
        best_mAP = best_score_from_log
        best_epoch = best_epoch_ckpt

    # 如果 best_score_from_log 存在且与 per-epoch max 不同, 优先 best_score (更准确)
    if best_score_from_log is not None and per_epoch_mAP:
        if abs(best_score_from_log - best_mAP) > 0.0005:
            best_mAP = best_score_from_log  # best score 行更可靠
            # 找到最接近的 epoch
            for d in per_epoch_detail:
                if abs(d['mAP'] - best_score_from_log) < 0.001:
                    best_epoch = d.get('epoch', best_epoch_ckpt)
                    break

    return {
        'per_epoch_mAP': per_epoch_mAP,
        'per_epoch_detail': per_epoch_detail,
        'best_mAP': best_mAP,
        'best_epoch': best_epoch,
        'total_epochs': len(per_epoch_mAP),
    }


def compute_stability(per_epoch_mAP):
    """计算训练稳定性指标.

    Args:
        per_epoch_mAP: 逐 epoch val mAP 列表

    Returns:
        dict: 稳定性指标
    """
    if not per_epoch_mAP or len(per_epoch_mAP) < 2:
        return None

    # 最后 30 epoch
    last_30 = per_epoch_mAP[-30:]
    n = len(last_30)

    mean_val = sum(last_30) / n
    variance = sum((x - mean_val) ** 2 for x in last_30) / n
    std = variance ** 0.5
    cv = std / mean_val if mean_val > 0 else None
    range_val = max(last_30) - min(last_30)

    # best 1% 内的 epoch 数
    best_mAP = max(per_epoch_mAP)
    threshold = best_mAP * 0.99  # best 1% = 99% of best
    best_1pct_count = sum(1 for x in last_30 if x >= threshold)

    return {
        'last_30_std': std,
        'last_30_cv': cv,
        'last_30_range': range_val,
        'best_1pct_count': best_1pct_count,
        'total_epochs': len(per_epoch_mAP),
        'per_epoch_mAP_json': json.dumps(per_epoch_mAP),
    }


# ─── 实验目录发现 ───────────────────────────────────────────

def find_experiment_dirs(work_dirs_root):
    """递归查找所有包含 dumped config 或 checkpoint 的实验目录.

    过滤规则:
      1. 必须有 best_coco_bbox_mAP_epoch_*.pth 或 scalars.json (训练证据)
      2. 跳过 backup 目录 (LDMDet_backup, experiments_backup 等)
      3. 跳过 vis_data 子目录

    Returns:
        list[dict]: [{'work_dir': str, 'config_path': str, 'has_ckpt': bool}]
    """
    experiments = []
    work_dirs_root = os.path.abspath(work_dirs_root)

    if not os.path.isdir(work_dirs_root):
        print(f'[ERROR] work_dirs 目录不存在: {work_dirs_root}')
        return experiments

    # 需要跳过的目录名模式
    SKIP_DIR_PATTERNS = [
        'vis_data',
        'backup', 'LDMDet_backup', 'ldmdet_backup', 'experiments_backup',
        '__pycache__', '.git', 'diagnostics',
    ]

    def should_skip_dir(dirname):
        for pattern in SKIP_DIR_PATTERNS:
            if pattern in dirname:
                return True
        return False

    # 递归遍历
    for root, dirs, files in os.walk(work_dirs_root):
        # 过滤子目录
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]

        # 查找 checkpoint (训练证据)
        ckpt_files = [f for f in files if f.startswith('best_coco_bbox_mAP_epoch_') and f.endswith('.pth')]
        # 查找 scalars.json (训练证据)
        has_scalars = any(f == 'scalars.json' for f in files)
        # 查找 vis_data 子目录中的 scalars.json
        if not has_scalars:
            for d in dirs:
                scalars_check = os.path.join(root, d, 'scalars.json')
                if os.path.isfile(scalars_check):
                    has_scalars = True
                    break

        # 必须有训练证据
        if not ckpt_files and not has_scalars:
            continue

        # 查找 dumped config (.py 文件, 排除 __init__.py)
        config_files = [f for f in files if f.endswith('.py') and f != '__init__.py']

        # 如果有多个 config, 取第一个 (通常只有一个)
        config_path = None
        if config_files:
            # 优先选择与目录名匹配的 config
            dir_name = os.path.basename(root)
            matching = [f for f in config_files if dir_name in f or f.replace('.py', '') in dir_name]
            if matching:
                config_path = os.path.join(root, matching[0])
            else:
                config_path = os.path.join(root, config_files[0])

        experiments.append({
            'work_dir': os.path.relpath(root, _PROJECT_ROOT),
            'config_path': config_path,
            'has_ckpt': len(ckpt_files) > 0,
            'ckpt_files': ckpt_files,
        })

    return experiments


def find_best_ckpt(work_dir):
    """从 best_coco_bbox_mAP_epoch_*.pth 文件名提取 best epoch."""
    pattern = os.path.join(_PROJECT_ROOT, work_dir, 'best_coco_bbox_mAP_epoch_*.pth')
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None, None
    best_epoch = -1
    best_ckpt = None
    for ckpt in ckpts:
        m = re.search(r'best_coco_bbox_mAP_epoch_(\d+)\.pth', os.path.basename(ckpt))
        if m:
            ep = int(m.group(1))
            if ep > best_epoch:
                best_epoch = ep
                best_ckpt = os.path.relpath(ckpt, _PROJECT_ROOT)
    if best_ckpt is not None:
        return best_epoch, best_ckpt
    return None, os.path.relpath(ckpts[0], _PROJECT_ROOT)


# ─── SQLite 操作 ────────────────────────────────────────────

def init_db(db_path):
    """初始化数据库, 执行 schema.sql."""
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'schema.sql')
    conn = sqlite3.connect(db_path)
    # Older databases may contain duplicate evaluation rows from repeated scans.
    # Deduplicate before creating the new unique identity index in schema.sql.
    conn.execute('''CREATE TABLE IF NOT EXISTS evaluation (
        eval_id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT NOT NULL,
        split TEXT NOT NULL, mAP REAL, AP50 REAL, AP75 REAL, AP_small REAL,
        AP_medium REAL, AP_large REAL, epoch INTEGER, checkpoint_path TEXT,
        evaluated_at TEXT, source TEXT)''')
    conn.execute('''DELETE FROM evaluation WHERE eval_id NOT IN (
        SELECT MAX(eval_id) FROM evaluation
        GROUP BY experiment_id, split, source, IFNULL(epoch, -1),
                 IFNULL(checkpoint_path, ''))''')
    conn.executescript(open(schema_path).read())
    # Historical databases may contain config paths whose config row was never
    # imported. Null the broken reference while retaining the path in the run
    # directory itself; this restores FK integrity without inventing metadata.
    conn.execute('''UPDATE experiment SET config_path=NULL
        WHERE config_path IS NOT NULL
          AND config_path NOT IN (SELECT config_path FROM config)''')
    conn.commit()
    return conn


def upsert_aug_pipeline(conn, aug_info):
    """插入或更新增强策略."""
    conn.execute('''
        INSERT OR REPLACE INTO aug_pipeline
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


def upsert_config(conn, cfg_info):
    """插入或更新配置."""
    conn.execute('''
        INSERT OR REPLACE INTO config
            (config_path, source_config_path, dataset, data_root, aug_pipeline_hash,
             coupling_type, ot_epsilon, ot_matcher, ot_sample, coupling_mode,
             time_conditioning, solver_type, rf_schedule, rf_shift,
             batch_size, max_epochs, num_classes, num_proposals, sampling_timesteps,
             has_early_stopping)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        cfg_info['config_path'],
        cfg_info.get('source_config_path', ''),
        cfg_info['dataset'],
        cfg_info['data_root'],
        cfg_info['aug_pipeline_hash'],
        cfg_info['coupling_type'],
        cfg_info['ot_epsilon'],
        cfg_info['ot_matcher'],
        cfg_info['ot_sample'],
        cfg_info['coupling_mode'],
        cfg_info['time_conditioning'],
        cfg_info['solver_type'],
        cfg_info['rf_schedule'],
        cfg_info['rf_shift'],
        cfg_info['batch_size'],
        cfg_info['max_epochs'],
        cfg_info['num_classes'],
        cfg_info['num_proposals'],
        cfg_info['sampling_timesteps'],
        cfg_info['has_early_stopping'],
    ))


def upsert_experiment(conn, exp_info):
    """插入或更新实验，同时保留轻量扫描无法重建的已有字段."""
    conn.execute('''
        INSERT INTO experiment
            (experiment_id, name, work_dir, server, config_path,
             swanlab_project, swanlab_run_id, seed, status,
             best_val_mAP, best_val_epoch, best_checkpoint,
             training_start, training_end, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(experiment_id) DO UPDATE SET
            name=excluded.name,
            work_dir=excluded.work_dir,
            server=excluded.server,
            config_path=COALESCE(excluded.config_path, experiment.config_path),
            swanlab_project=COALESCE(excluded.swanlab_project, experiment.swanlab_project),
            swanlab_run_id=COALESCE(excluded.swanlab_run_id, experiment.swanlab_run_id),
            seed=COALESCE(excluded.seed, experiment.seed),
            status=CASE WHEN excluded.status='unknown' THEN experiment.status
                        ELSE excluded.status END,
            best_val_mAP=COALESCE(excluded.best_val_mAP, experiment.best_val_mAP),
            best_val_epoch=COALESCE(excluded.best_val_epoch, experiment.best_val_epoch),
            best_checkpoint=COALESCE(excluded.best_checkpoint, experiment.best_checkpoint),
            training_start=COALESCE(excluded.training_start, experiment.training_start),
            training_end=COALESCE(excluded.training_end, experiment.training_end),
            notes=CASE WHEN excluded.notes='' THEN experiment.notes ELSE excluded.notes END
    ''', (
        exp_info['experiment_id'],
        exp_info.get('name', ''),
        exp_info['work_dir'],
        exp_info.get('server', 'unknown'),
        exp_info.get('config_path'),
        exp_info.get('swanlab_project'),
        exp_info.get('swanlab_run_id'),
        exp_info.get('seed'),
        exp_info.get('status', 'unknown'),
        exp_info.get('best_val_mAP'),
        exp_info.get('best_val_epoch'),
        exp_info.get('best_checkpoint'),
        exp_info.get('training_start'),
        exp_info.get('training_end'),
        exp_info.get('notes', ''),
    ))


def insert_evaluation(conn, experiment_id, split, metrics, source='training_log'):
    """幂等写入评估结果；重复扫描不会制造重复行."""
    conn.execute('''
        INSERT OR REPLACE INTO evaluation
            (experiment_id, split, mAP, AP50, AP75, AP_small, AP_medium, AP_large,
             epoch, checkpoint_path, evaluated_at, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        experiment_id,
        split,
        metrics.get('mAP'),
        metrics.get('AP50'),
        metrics.get('AP75'),
        metrics.get('AP_small'),
        metrics.get('AP_medium'),
        metrics.get('AP_large'),
        metrics.get('epoch'),
        metrics.get('checkpoint_path'),
        metrics.get('evaluated_at'),
        source,
    ))


def upsert_stability(conn, experiment_id, stability_info):
    """插入或更新稳定性指标."""
    conn.execute('''
        INSERT OR REPLACE INTO stability
            (experiment_id, last_30_std, last_30_cv, last_30_range,
             best_1pct_count, total_epochs, per_epoch_mAP_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        experiment_id,
        stability_info['last_30_std'],
        stability_info['last_30_cv'],
        stability_info['last_30_range'],
        stability_info['best_1pct_count'],
        stability_info['total_epochs'],
        stability_info['per_epoch_mAP_json'],
    ))


# ─── 主流程 ─────────────────────────────────────────────────

def extract_seed_from_path(work_dir):
    """从 work_dir 路径提取 seed."""
    m = re.search(r'seed[_/](\d+)', work_dir)
    if m:
        return int(m.group(1))
    return None


def infer_run_status(abs_work_dir, metrics, cfg_info, best_ckpt_path):
    """Infer completion without treating a best checkpoint as a completion flag.

    A best checkpoint is normally written early in training. Completion requires
    reaching max_epochs or an explicit terminal marker. A recently updated,
    incomplete log is classified as running.
    """
    total_epochs = metrics.get('total_epochs', 0) if metrics else 0
    max_epochs = cfg_info.get('max_epochs') if cfg_info else None
    if max_epochs and total_epochs >= max_epochs:
        return 'completed'

    terminal_patterns = (
        'Training completed', 'training completed', 'EarlyStopping',
        'early stopping', 'Run finished', 'run finished')
    log_files = glob.glob(os.path.join(abs_work_dir, '*.log'))
    latest_mtime = 0.0
    for log_path in log_files:
        try:
            latest_mtime = max(latest_mtime, os.path.getmtime(log_path))
            with open(log_path, 'rb') as f:
                f.seek(max(0, os.path.getsize(log_path) - 65536))
                tail = f.read().decode('utf-8', errors='ignore')
            if any(marker in tail for marker in terminal_patterns):
                return 'completed'
        except OSError:
            pass

    if metrics:
        # A log updated in the last six hours is unambiguously active. Older
        # incomplete runs remain 'unknown' rather than being falsely completed.
        now = datetime.now(timezone.utc).timestamp()
        if latest_mtime and now - latest_mtime < 6 * 3600:
            return 'running'
        # Without a max-epoch signal, an old log does not prove either
        # completion or activity. Returning unknown lets the upsert preserve
        # the previously audited status.
        return 'unknown'
    return 'unknown' if best_ckpt_path else 'unknown'


def process_experiment(conn, exp_dir, server='unknown', verbose=False,
                       namespace_server=False):
    """处理单个实验目录."""
    work_dir = exp_dir['work_dir']
    experiment_id = f'{server}:{work_dir}' if namespace_server else work_dir
    abs_work_dir = os.path.join(_PROJECT_ROOT, work_dir)

    if verbose:
        print(f'\n处理: {work_dir}')

    # 1. 解析配置
    cfg_info = None
    aug_info = _unknown_augmentation()
    if exp_dir['config_path']:
        abs_config = os.path.join(_PROJECT_ROOT, exp_dir['config_path'])
        cfg_info = parse_dumped_config(abs_config)
        if cfg_info:
            aug_info = cfg_info['_aug_info']
            del cfg_info['_aug_info']

    # 2. 提取指标 (优先 scalars.json, fallback 到 log 文件)
    scalars_path = find_scalars_json(abs_work_dir)
    metrics = extract_metrics_from_scalars(scalars_path)

    # 3. 提取 best checkpoint
    best_epoch_ckpt, best_ckpt_path = find_best_ckpt(work_dir)

    # 2b. 若 scalars.json 不可用, 从 log 文件提取 (fallback)
    if metrics is None:
        metrics = extract_metrics_from_logs(abs_work_dir, best_epoch_ckpt)

    # 4. 确定状态
    status = infer_run_status(abs_work_dir, metrics, cfg_info, best_ckpt_path)

    # 5. 提取 seed
    seed = extract_seed_from_path(work_dir)

    # 6. 写入数据库
    if aug_info:
        upsert_aug_pipeline(conn, aug_info)

    if cfg_info:
        upsert_config(conn, cfg_info)

    exp_info = {
        'experiment_id': experiment_id,
        'name': os.path.basename(work_dir),
        'work_dir': work_dir,
        'server': server,
        'config_path': exp_dir['config_path'] if cfg_info else None,
        'seed': seed,
        'status': status,
        'best_val_mAP': metrics['best_mAP'] if metrics else None,
        'best_val_epoch': metrics['best_epoch'] if metrics else best_epoch_ckpt,
        'best_checkpoint': best_ckpt_path,
    }
    upsert_experiment(conn, exp_info)

    # 7. 写入评估结果 (val)
    if metrics and metrics.get('per_epoch_detail'):
        best_detail = metrics['per_epoch_detail'][max(range(len(metrics['per_epoch_detail'])),
                                                      key=lambda i: metrics['per_epoch_detail'][i]['mAP'])]
        insert_evaluation(conn, experiment_id, 'val', {
            'mAP': best_detail['mAP'],
            'AP50': best_detail.get('AP50'),
            'AP75': best_detail.get('AP75'),
            'AP_small': best_detail.get('AP_small'),
            'AP_medium': best_detail.get('AP_medium'),
            'AP_large': best_detail.get('AP_large'),
            'epoch': best_detail.get('epoch'),
            'checkpoint_path': best_ckpt_path,
        }, source='training_log')

    # 8. 写入稳定性指标
    if metrics and len(metrics['per_epoch_mAP']) >= 2:
        stability = compute_stability(metrics['per_epoch_mAP'])
        if stability:
            upsert_stability(conn, experiment_id, stability)

    if verbose:
        aug_name = aug_info.get('pipeline_name', 'unknown')
        best_mAP = metrics['best_mAP'] if metrics else 'N/A'
        print(f'  增强: {aug_name} | 数据集: {cfg_info["dataset"] if cfg_info else "unknown"} | '
              f'耦合: {cfg_info["coupling_type"] if cfg_info else "unknown"} | '
              f'best mAP: {best_mAP} | 状态: {status}')

    return True


def main():
    parser = argparse.ArgumentParser(description='构建实验数据库')
    parser.add_argument('--work-dirs', default='work_dirs',
                        help='work_dirs 目录 (默认: work_dirs)')
    parser.add_argument('--db', default=os.path.join(os.path.dirname(__file__), 'experiments.db'),
                        help='SQLite 数据库路径')
    parser.add_argument('--server', default='ross',
                        help='服务器标签 (ross/workstation)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='详细输出')
    parser.add_argument('--dry-run', action='store_true',
                        help='只扫描不写入数据库')
    parser.add_argument('--namespace-server', action='store_true',
                        help='实验 ID 使用 server:work_dir，合并多服务器物理运行时启用')
    args = parser.parse_args()

    work_dirs_root = os.path.join(_PROJECT_ROOT, args.work_dirs)
    print(f'扫描目录: {work_dirs_root}')
    print(f'服务器: {args.server}')
    print(f'数据库: {args.db}')

    # 发现实验目录
    experiments = find_experiment_dirs(work_dirs_root)
    print(f'发现 {len(experiments)} 个实验目录')

    if args.dry_run:
        print('\n--- Dry Run 模式: 只扫描不写入 ---')
        for exp in experiments:
            print(f'  {exp["work_dir"]} (config: {exp["config_path"] is not None}, ckpt: {exp["has_ckpt"]})')
        return

    # 初始化数据库
    conn = init_db(args.db)

    # 处理每个实验
    success = 0
    failed = 0
    for exp in experiments:
        try:
            process_experiment(conn, exp, server=args.server, verbose=args.verbose,
                               namespace_server=args.namespace_server)
            success += 1
        except Exception as e:
            print(f'  [ERROR] 处理 {exp["work_dir"]} 失败: {e}')
            failed += 1

    conn.commit()

    # 统计
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM experiment')
    total = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM config')
    total_cfg = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM evaluation')
    total_eval = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM stability')
    total_stab = cursor.fetchone()[0]
    cursor.execute('SELECT pipeline_name, COUNT(*) FROM aug_pipeline GROUP BY pipeline_name')
    aug_stats = cursor.fetchall()

    print(f'\n=== 摄入完成 ===')
    print(f'成功: {success} | 失败: {failed}')
    print(f'数据库统计:')
    print(f'  实验: {total}')
    print(f'  配置: {total_cfg}')
    print(f'  评估: {total_eval}')
    print(f'  稳定性: {total_stab}')
    print(f'  增强策略: {aug_stats}')

    conn.close()


if __name__ == '__main__':
    main()
