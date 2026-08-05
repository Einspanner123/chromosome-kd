"""添加 workstation 上的 ε 消融实验到 SQLite 实验数据库.

这些实验不在本地挂载的 ross 8TB 上 (或 ross 上仅有 checkpoint 无日志/config),
需要通过 SSH 从 workstation 获取 config、日志和 scalars.json 数据.

实验列表:
  1. stoch_eps05_20260617_010941  — ε=0.5, D1 NoAug (workstation 独有)
  2. stoch_eps05_seed123           — ε=0.5, D1 NoAug (ross ablation_old/ 有同名, val=None)
  3. nonlinear_trajectory_e43_eps2 — ε=2, D1 standard (ross ablation_old/ 有同名, val=None)
  4. nonlinear_trajectory_e43_eps2_real — ε=2, D1 standard (workstation 独有)
  5. nonlinear_trajectory_e43_eps3 — ε=3, D1 standard (ross ablation_old/ 有同名, val=None)
  6. ablation_old/stochot_eps1     — ε=1, D1 standard (ross ablation_old/ 有同名, val=None)
  7. ablation_old/stochot_eps2     — ε=2, D1 standard (ross ablation_old/ 有同名, val=0.749)

用法:
  python tools/experiment_db/add_workstation_eps_ablation.py
依赖: mmengine, mmdet (conda activate chromo)
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments.db')

# ─── SSH 配置 ───────────────────────────────────────────────
SSH_HOST = 'linkst@100.99.131.26'
SSH_PASS = '000928'
WS_PROJECT = '/home/linkst/workplace/chromo/chromosome-kd'
WS_WORK_DIRS = f'{WS_PROJECT}/work_dirs'

# ─── 复用 build_experiment_db 的解析函数 ────────────────────
from build_experiment_db import (  # noqa: E402
    parse_dumped_config,
    classify_augmentation,
    upsert_aug_pipeline,
    upsert_config,
    extract_metrics_from_scalars,
    D1_DATA_ROOT_PATTERN,
    D2_DATA_ROOT_PATTERN,
)


# ─── 实验定义 ───────────────────────────────────────────────
# ws_rel_path:  workstation 上 work_dirs 下的相对路径
# experiment_id: 数据库中的 experiment_id (与 ross 上一致的使用 ross 路径)
# server:       'workstation' 或 'ross+workstation'
# on_ross:      是否在 ross 上存在 (同 experiment_id)
EXPERIMENTS = [
    # 1. workstation 独有 — ε=0.5 NoAug seed42
    {
        'ws_rel_path': 'multi_seed/stoch_eps05_20260617_010941/seed_42',
        'experiment_id': 'work_dirs/multi_seed/stoch_eps05_20260617_010941/seed_42',
        'server': 'workstation',
        'on_ross': False,
    },
    # 2. ε=0.5 NoAug seed123 — ross ablation_old/ 有同名 (val=None)
    {
        'ws_rel_path': 'stoch_eps05_seed123',
        'experiment_id': 'work_dirs/ablation_old/stoch_eps05_seed123',
        'server': 'ross+workstation',
        'on_ross': True,
    },
    # 3. ε=2 ot_flow — ross ablation_old/ 有同名 (val=None)
    {
        'ws_rel_path': 'nonlinear_trajectory_e43_eps2',
        'experiment_id': 'work_dirs/ablation_old/nonlinear_trajectory_e43_eps2',
        'server': 'ross+workstation',
        'on_ross': True,
    },
    # 4. ε=2 ot_flow real — workstation 独有
    {
        'ws_rel_path': 'nonlinear_trajectory_e43_eps2_real',
        'experiment_id': 'work_dirs/nonlinear_trajectory_e43_eps2_real',
        'server': 'workstation',
        'on_ross': False,
    },
    # 5. ε=3 ot_flow — ross ablation_old/ 有同名 (val=None)
    {
        'ws_rel_path': 'nonlinear_trajectory_e43_eps3',
        'experiment_id': 'work_dirs/ablation_old/nonlinear_trajectory_e43_eps3',
        'server': 'ross+workstation',
        'on_ross': True,
    },
    # 6. ε=1 ot_flow — ross ablation_old/ 有同名 (val=None)
    {
        'ws_rel_path': 'ablation_old/stochot_eps1',
        'experiment_id': 'work_dirs/ablation_old/stochot_eps1',
        'server': 'ross+workstation',
        'on_ross': True,
    },
    # 7. ε=2 ot_flow — ross ablation_old/ 有同名 (val=0.749, 已有值跳过)
    {
        'ws_rel_path': 'ablation_old/stochot_eps2',
        'experiment_id': 'work_dirs/ablation_old/stochot_eps2',
        'server': 'ross+workstation',
        'on_ross': True,
    },
]


# ─── SSH 工具函数 ───────────────────────────────────────────

def ssh_exec(command, timeout=60):
    """通过 SSH 在 workstation 上执行命令, 返回 stdout."""
    full_cmd = [
        'sshpass', '-p', SSH_PASS,
        'ssh', '-o', 'StrictHostKeyChecking=no',
        '-o', 'ConnectTimeout=10',
        SSH_HOST, command,
    ]
    result = subprocess.run(
        full_cmd, capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        print(f'  [SSH WARN] returncode={result.returncode}: {result.stderr[:200]}')
    return result.stdout


def fetch_experiment_data(ws_rel_path):
    """从 workstation 获取单个实验的所有数据.

    通过单次 SSH 调用获取:
      - config .py 文件路径和内容
      - best_coco_bbox_mAP_epoch_*.pth 文件名
      - 所有 scalars.json 文件内容
      - 所有 .log 文件中的 mAP/best score 行

    Returns:
        dict: {
            'config_path': str,       # workstation 上 config 绝对路径
            'config_content': str,    # config 文件内容
            'best_ckpt': str,         # best checkpoint 文件名
            'scalars_list': [str],    # 各 scalars.json 内容列表
            'log_grep': str,          # 所有日志中 mAP 相关行
        }
    """
    ws_dir = f'{WS_WORK_DIRS}/{ws_rel_path}'

    # 单次 SSH 调用, 用唯一分隔符输出所有数据
    shell_script = f"""\
DIR="{ws_dir}"

echo "<<<CONFIG_PATH>>>"
CFG=$(find "$DIR" -maxdepth 1 -name "*.py" ! -name "__init__.py" 2>/dev/null | head -1)
echo "$CFG"
echo "<<<CONFIG_CONTENT>>>"
cat "$CFG" 2>/dev/null
echo "<<<END_CONFIG>>>"

echo "<<<BEST_CKPT>>>"
find "$DIR" -maxdepth 1 -name "best_coco_bbox_mAP_epoch_*.pth" 2>/dev/null | sort

echo "<<<SCALARS_START>>>"
for f in $(find "$DIR" -name "scalars.json" -path "*/vis_data/*" 2>/dev/null | sort); do
    echo "<<<SCALARS_FILE>>>"
    echo "$f"
    echo "<<<SCALARS_CONTENT>>>"
    cat "$f" 2>/dev/null
    echo "<<<END_SCALARS>>>"
done
echo "<<<SCALARS_END>>>"
"""
    # 日志单独获取 (可能较大, 只 grep mAP 行)
    log_grep_cmd = f"""\
DIR="{ws_dir}"
for f in $(find "$DIR" -name "*.log" 2>/dev/null | sort); do
    echo "<<<LOG_FILE>>>"
    echo "$f"
    echo "<<<LOG_GREP>>>"
    grep -h 'best score\\|Epoch(val).*coco/bbox_mAP' "$f" 2>/dev/null
    echo "<<<END_LOG>>>"
done
"""

    raw = ssh_exec(shell_script, timeout=30)
    log_raw = ssh_exec(log_grep_cmd, timeout=30)

    # 解析输出
    data = {
        'config_path': '',
        'config_content': '',
        'best_ckpt': '',
        'scalars_list': [],
        'log_grep': '',
    }

    # 解析 config
    m = re.search(r'<<<CONFIG_PATH>>>\n(.*?)\n<<<CONFIG_CONTENT>>>\n(.*?)\n<<<END_CONFIG>>>',
                  raw, re.DOTALL)
    if m:
        data['config_path'] = m.group(1).strip()
        data['config_content'] = m.group(2)

    # 解析 best ckpt
    m = re.search(r'<<<BEST_CKPT>>>\n(.*?)\n<<<SCALARS_START>>>',
                  raw, re.DOTALL)
    if m:
        ckpts = [line.strip() for line in m.group(1).strip().split('\n') if line.strip()]
        if ckpts:
            # 取 epoch 最大的
            best_ep = -1
            best_name = ''
            for ck in ckpts:
                ep_m = re.search(r'best_coco_bbox_mAP_epoch_(\d+)\.pth', ck)
                if ep_m:
                    ep = int(ep_m.group(1))
                    if ep > best_ep:
                        best_ep = ep
                        best_name = os.path.basename(ck)
            data['best_ckpt'] = best_name

    # 解析 scalars.json
    scalars_section = re.findall(
        r'<<<SCALARS_CONTENT>>>\n(.*?)\n<<<END_SCALARS>>>',
        raw, re.DOTALL,
    )
    data['scalars_list'] = scalars_section

    # 解析 log grep
    log_sections = re.findall(r'<<<LOG_GREP>>>\n(.*?)\n<<<END_LOG>>>', log_raw, re.DOTALL)
    data['log_grep'] = '\n'.join(log_sections)

    return data


# ─── 指标提取 ───────────────────────────────────────────────

def extract_metrics_from_data(scalars_list, log_grep, best_epoch_ckpt=None):
    """从获取的 scalars.json 内容和 log grep 内容中提取 best mAP.

    优先使用 scalars.json (更精确), fallback 到 log grep.

    Returns:
        dict: {
            'per_epoch_mAP': list[float],
            'per_epoch_detail': list[dict],
            'best_mAP': float,
            'best_epoch': int,
            'total_epochs': int,
        } or None
    """
    # 1. 尝试 scalars.json (合并所有文件)
    all_entries = []
    for scalars_content in scalars_list:
        for line in scalars_content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            mAP = entry.get('coco/bbox_mAP')
            if mAP is not None and mAP > 0:
                all_entries.append({
                    'mAP': mAP,
                    'AP50': entry.get('coco/bbox_mAP_50'),
                    'AP75': entry.get('coco/bbox_mAP_75'),
                    'AP_small': entry.get('coco/bbox_mAP_s'),
                    'AP_medium': entry.get('coco/bbox_mAP_m'),
                    'AP_large': entry.get('coco/bbox_mAP_l'),
                    'epoch': entry.get('epoch'),
                    'step': entry.get('step'),
                })

    # 2. 尝试 log grep (Epoch(val) 行)
    log_entries = []
    best_score_from_log = None
    for line in log_grep.split('\n'):
        # 提取 "best score: X.XXX" (注意行尾可能有句号)
        if 'best score:' in line:
            m = re.search(r'best score:\s*([\d]+(?:\.\d+)?)', line)
            if m:
                best_score_from_log = float(m.group(1))

        # 提取 "Epoch(val) [N]...coco/bbox_mAP: X.XXXX"
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
                if mAP > 0:
                    log_entries.append({
                        'mAP': mAP,
                        'AP50': float(ap50_m.group(1)) if ap50_m else None,
                        'AP75': float(ap75_m.group(1)) if ap75_m else None,
                        'AP_small': float(aps_m.group(1)) if aps_m else None,
                        'AP_medium': float(apm_m.group(1)) if apm_m else None,
                        'AP_large': float(apl_m.group(1)) if apl_m else None,
                        'epoch': ep,
                        'step': None,
                    })

    # 合并 scalars + log entries (去重: 同 epoch 取 scalars 优先)
    epoch_map = {}
    all_combined = []
    for entry in all_entries + log_entries:
        ep = entry.get('epoch')
        if ep is not None and ep not in epoch_map:
            epoch_map[ep] = len(all_combined)
            all_combined.append(entry)
        elif ep is not None:
            # scalars 优先 (已在前), 不覆盖
            pass
        else:
            all_combined.append(entry)

    if not all_combined and best_score_from_log is None:
        return None

    # 确定最佳 mAP
    if all_combined:
        best_idx = max(range(len(all_combined)), key=lambda i: all_combined[i]['mAP'])
        best_mAP = all_combined[best_idx]['mAP']
        best_epoch = all_combined[best_idx].get('epoch', best_epoch_ckpt)
    else:
        best_mAP = best_score_from_log
        best_epoch = best_epoch_ckpt

    # 如果 best_score_from_log 存在且与 per-epoch max 不同, 优先 best_score
    if best_score_from_log is not None and all_combined:
        if abs(best_score_from_log - best_mAP) > 0.0005:
            best_mAP = best_score_from_log
            for d in all_combined:
                if abs(d['mAP'] - best_score_from_log) < 0.001:
                    best_epoch = d.get('epoch', best_epoch_ckpt)
                    break

    per_epoch_mAP = [e['mAP'] for e in all_combined]

    return {
        'per_epoch_mAP': per_epoch_mAP,
        'per_epoch_detail': all_combined,
        'best_mAP': best_mAP,
        'best_epoch': best_epoch,
        'total_epochs': len(all_combined),
    }


# ─── 配置解析 (带 regex fallback) ───────────────────────────

def parse_config_with_fallback(config_content, config_path):
    """解析 config 内容, 优先用 mmengine, 失败则用正则提取.

    Returns:
        dict or None: 配置信息 (同 parse_dumped_config 格式, 含 _aug_info)
    """
    # 写入临时文件, 用 mmengine 解析
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False, dir=_PROJECT_ROOT,
    ) as tmp:
        tmp.write(config_content)
        tmp_path = tmp.name

    try:
        cfg_info = parse_dumped_config(tmp_path)
        if cfg_info:
            # 替换 config_path 为 workstation 路径
            cfg_info['config_path'] = config_path
            return cfg_info
    except Exception as e:
        print(f'  [WARN] mmengine 解析失败: {e}')
    finally:
        os.unlink(tmp_path)

    # Fallback: 正则提取
    print('  [INFO] 使用正则提取配置信息')
    return parse_config_regex(config_content, config_path)


def parse_config_regex(content, config_path):
    """用正则从 config 内容中提取关键信息."""
    # data_root
    data_root = ''
    dataset = 'unknown'
    m = re.search(r"data_root\s*=\s*['\"]([^'\"]+)['\"]", content)
    if m:
        data_root = m.group(1)
    if D1_DATA_ROOT_PATTERN in data_root:
        dataset = 'D1'
    elif D2_DATA_ROOT_PATTERN in data_root:
        dataset = 'D2'

    # coupling type
    coupling_type = 'none'
    # 匹配 coupling=dict(...type='xxx'...) 或 coupling = dict(...)
    coupling_block = re.search(
        r'coupling\s*=\s*dict\(([^)]*(?:\([^)]*\))*[^)]*)\)',
        content, re.DOTALL,
    )
    if coupling_block:
        coupling_text = coupling_block.group(1)
        type_m = re.search(r"type\s*=\s*['\"](\w+)['\"]", coupling_text)
        if type_m:
            coupling_type = type_m.group(1)
        else:
            coupling_type = 'random'
    else:
        # 检查是否有 coupling 字段
        if 'coupling' in content:
            coupling_type = 'random'

    # ot_epsilon
    ot_epsilon = None
    if coupling_block:
        eps_m = re.search(r'epsilon\s*=\s*([\d.]+)', coupling_block.group(1))
        if eps_m:
            ot_epsilon = float(eps_m.group(1))

    # coupling_mode
    coupling_mode = None
    if coupling_block:
        mode_m = re.search(r"coupling_mode\s*=\s*['\"](\w+)['\"]", coupling_block.group(1))
        if mode_m:
            coupling_mode = mode_m.group(1)

    # ot_matcher
    ot_matcher = None
    if coupling_block:
        matcher_m = re.search(r"matcher\s*=\s*['\"](\w+)['\"]", coupling_block.group(1))
        if matcher_m:
            ot_matcher = matcher_m.group(1)

    # ot_sample
    ot_sample = None
    if coupling_block:
        sample_m = re.search(r"sample\s*=\s*(True|False)", coupling_block.group(1))
        if sample_m:
            ot_sample = sample_m.group(1) == 'True'

    # train_pipeline — 提取步骤类型
    step_types = []
    has_random_choice_resize = False
    has_random_crop = False
    has_random_flip = False
    has_fixed_resize = False

    # 检查 train_pipeline 块中的 type
    pipeline_block = re.search(
        r'train_pipeline\s*=\s*\[(.*?)\]',
        content, re.DOTALL,
    )
    if pipeline_block:
        pipeline_text = pipeline_block.group(1)
        # 提取所有 type='xxx'
        types = re.findall(r"type\s*=\s*['\"](\w+)['\"]", pipeline_text)
        for t in types:
            if t not in ('PackDetInputs',):  # 跳过非增强步骤
                step_types.append(t)
            if t == 'RandomFlip':
                has_random_flip = True
            elif t == 'Resize':
                has_fixed_resize = True
            elif t == 'RandomChoiceResize':
                has_random_choice_resize = True
            elif t == 'RandomCrop':
                has_random_crop = True
        # RandomChoice 包含子 transforms
        if 'RandomChoice' in pipeline_text:
            step_types.insert(0, 'RandomChoice')

    # 分类
    if has_random_choice_resize or has_random_crop:
        pipeline_name = 'standard'
        desc = '标准增强 (RandomFlip + RandomChoiceResize + RandomCrop)'
    elif has_fixed_resize and has_random_flip and not has_random_choice_resize:
        pipeline_name = 'simple'
        desc = '简单增强 (Resize + RandomFlip)'
    else:
        pipeline_name = 'custom'
        desc = f'自定义增强 ({", ".join(step_types) if step_types else "unknown"})'

    import hashlib
    pipeline_hash = hashlib.md5('|'.join(step_types).encode()).hexdigest()[:16]

    aug_info = {
        'pipeline_hash': pipeline_hash,
        'pipeline_name': pipeline_name,
        'has_random_choice_resize': has_random_choice_resize,
        'has_random_crop': has_random_crop,
        'has_random_flip': has_random_flip,
        'has_fixed_resize': has_fixed_resize,
        'step_types_json': json.dumps(step_types),
        'description': desc,
    }

    # 其他模型设置
    solver_type = 'unknown'
    solver_m = re.search(r"solver_type\s*=\s*['\"](\w+)['\"]", content)
    if solver_m:
        solver_type = solver_m.group(1)

    rf_schedule = None
    rf_m = re.search(r"rf_schedule\s*=\s*['\"](\w+)['\"]", content)
    if rf_m:
        rf_schedule = rf_m.group(1)

    rf_shift = None
    shift_m = re.search(r'rf_shift\s*=\s*([\d.]+)', content)
    if shift_m:
        rf_shift = float(shift_m.group(1))

    num_proposals = None
    np_m = re.search(r'num_proposals\s*=\s*(\d+)', content)
    if np_m:
        num_proposals = int(np_m.group(1))

    sampling_timesteps = None
    st_m = re.search(r'sampling_timesteps\s*=\s*(\d+)', content)
    if st_m:
        sampling_timesteps = int(st_m.group(1))

    num_classes = None
    nc_m = re.search(r'num_classes\s*=\s*(\d+)', content)
    if nc_m:
        num_classes = int(nc_m.group(1))

    batch_size = None
    bs_m = re.search(r'batch_size\s*=\s*(\d+)', content)
    if bs_m:
        batch_size = int(bs_m.group(1))

    max_epochs = None
    me_m = re.search(r'max_epochs\s*=\s*(\d+)', content)
    if me_m:
        max_epochs = int(me_m.group(1))

    # time_conditioning
    time_conditioning = 'none'
    tc_m = re.search(r"time_conditioning\s*=\s*['\"](\w+)['\"]", content)
    if tc_m:
        time_conditioning = tc_m.group(1)

    # EarlyStopping
    has_early_stopping = 'EarlyStoppingHook' in content

    return {
        'config_path': config_path,
        'source_config_path': '',
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
        '_aug_info': aug_info,
    }


# ─── 主流程 ─────────────────────────────────────────────────

def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    results = []  # 记录每个实验的处理结果

    for exp_def in EXPERIMENTS:
        ws_rel = exp_def['ws_rel_path']
        exp_id = exp_def['experiment_id']
        server = exp_def['server']
        on_ross = exp_def['on_ross']

        print(f'\n{"=" * 60}')
        print(f'处理: {exp_id}')
        print(f'  workstation 路径: work_dirs/{ws_rel}')
        print(f'  server: {server}')

        # 1. 从 workstation 获取数据
        data = fetch_experiment_data(ws_rel)

        if not data['config_content']:
            print('  [ERROR] 无法获取 config 内容, 跳过')
            results.append({'experiment_id': exp_id, 'status': 'error', 'reason': 'no config'})
            continue

        # 2. 解析 config
        cfg_info = parse_config_with_fallback(
            data['config_content'], data['config_path'],
        )

        if cfg_info:
            aug_info = cfg_info['_aug_info']
            del cfg_info['_aug_info']
            print(f'  config: dataset={cfg_info["dataset"]}, '
                  f'coupling={cfg_info["coupling_type"]}, '
                  f'ε={cfg_info["ot_epsilon"]}, '
                  f'aug={aug_info["pipeline_name"]}')
        else:
            aug_info = None
            print('  [WARN] config 解析失败, 仅更新 mAP')

        # 3. 提取 best checkpoint epoch
        best_ckpt_name = data['best_ckpt']
        best_epoch_ckpt = None
        if best_ckpt_name:
            ep_m = re.search(r'best_coco_bbox_mAP_epoch_(\d+)\.pth', best_ckpt_name)
            if ep_m:
                best_epoch_ckpt = int(ep_m.group(1))
            print(f'  best ckpt: {best_ckpt_name} (epoch {best_epoch_ckpt})')
        else:
            print('  [WARN] 未找到 best checkpoint')

        # 4. 提取 metrics
        metrics = extract_metrics_from_data(
            data['scalars_list'], data['log_grep'], best_epoch_ckpt,
        )

        if metrics:
            print(f'  best mAP: {metrics["best_mAP"]} @ epoch {metrics["best_epoch"]} '
                  f'(total {metrics["total_epochs"]} epochs)')
        else:
            print('  [WARN] 无法提取 mAP 指标')

        # 5. 检查 DB 中是否已存在
        c.execute(
            'SELECT experiment_id, best_val_mAP, best_val_epoch, config_path, server '
            'FROM experiment WHERE experiment_id = ?',
            (exp_id,),
        )
        existing = c.fetchone()

        val_updated = False
        val_skipped = False

        if existing:
            existing_val = existing[1]
            existing_epoch = existing[2]
            existing_config = existing[3]
            existing_server = existing[4]

            print(f'  DB 已存在: val={existing_val}, epoch={existing_epoch}, '
                  f'config={existing_config is not None}, server={existing_server}')

            # 决定是否更新 val
            if existing_val is not None:
                # DB 中已有 val, 跳过 (不覆盖)
                print(f'  → 跳过 val 更新 (DB 已有 val={existing_val})')
                val_skipped = True
                final_val = existing_val
                final_epoch = existing_epoch
            elif metrics and metrics['best_mAP'] is not None:
                # DB 中 val=None, workstation 有值, 更新
                final_val = metrics['best_mAP']
                final_epoch = metrics['best_epoch'] or best_epoch_ckpt
                print(f'  → 更新 val: None → {final_val}')
                val_updated = True
            else:
                # DB 中已有 val, 跳过 val 更新
                # 但如果 epoch=None, 用 workstation 的 best_epoch_ckpt 补充
                final_val = existing_val
                final_epoch = existing_epoch

            # 更新 server 和 config_path
            # 如果 config_path 为 None 且我们解析到了 config, 更新它
            update_config = cfg_info if (existing_config is None and cfg_info) else None

            # 构建 UPDATE 语句
            set_clauses = ['server = ?']
            set_values = [server]

            if val_updated:
                set_clauses.append('best_val_mAP = ?')
                set_values.append(final_val)
                set_clauses.append('best_val_epoch = ?')
                set_values.append(final_epoch)
                set_clauses.append('best_checkpoint = ?')
                set_values.append(best_ckpt_name if best_ckpt_name else None)
                set_clauses.append('status = ?')
                set_values.append('completed')
            elif best_epoch_ckpt is not None and existing_epoch is None:
                # val 已有但不更新, 仅补充 epoch 和 checkpoint (如果 DB 中 epoch=None)
                final_epoch = best_epoch_ckpt
                set_clauses.append('best_val_epoch = ?')
                set_values.append(final_epoch)
                set_clauses.append('best_checkpoint = ?')
                set_values.append(best_ckpt_name if best_ckpt_name else None)
                print(f'  → 补充 epoch: None → {final_epoch} (val 保持 {final_val})')

            if update_config:
                set_clauses.append('config_path = ?')
                set_values.append(update_config['config_path'])

            set_values.append(exp_id)
            c.execute(
                f'UPDATE experiment SET {", ".join(set_clauses)} WHERE experiment_id = ?',
                set_values,
            )

            # 如果需要, 插入 config
            if update_config and aug_info:
                upsert_aug_pipeline(conn, aug_info)
                upsert_config(conn, update_config)
                print(f'  → 插入 config: {update_config["config_path"]}')

        else:
            # 新实验, INSERT
            final_val = metrics['best_mAP'] if metrics else None
            final_epoch = (metrics['best_epoch'] if metrics else None) or best_epoch_ckpt

            c.execute('''INSERT INTO experiment
                (experiment_id, name, work_dir, server, config_path, seed, status,
                 best_val_mAP, best_val_epoch, best_checkpoint)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (exp_id,
                 os.path.basename(exp_id),
                 exp_id,
                 server,
                 cfg_info['config_path'] if cfg_info else None,
                 None,  # seed (无法从路径提取)
                 'completed' if final_val is not None else 'unknown',
                 final_val,
                 final_epoch,
                 best_ckpt_name if best_ckpt_name else None))
            print(f'  → 新增实验: val={final_val}, epoch={final_epoch}')

            if cfg_info and aug_info:
                upsert_aug_pipeline(conn, aug_info)
                upsert_config(conn, cfg_info)
                print(f'  → 插入 config: {cfg_info["config_path"]}')

        # 6. 插入 val evaluation (如果 DB 中没有且我们有 metrics)
        if metrics and metrics.get('per_epoch_detail'):
            best_detail = max(metrics['per_epoch_detail'], key=lambda d: d['mAP'])

            # 检查是否已有 val evaluation
            c.execute(
                'SELECT eval_id FROM evaluation WHERE experiment_id = ? AND split = ?',
                (exp_id, 'val'),
            )
            if c.fetchone():
                print(f'  → val evaluation 已存在, 跳过')
            else:
                c.execute('''INSERT INTO evaluation
                    (experiment_id, split, mAP, AP50, AP75, AP_small, AP_medium, AP_large,
                     epoch, checkpoint_path, source)
                    VALUES (?, 'val', ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (exp_id,
                     best_detail['mAP'],
                     best_detail.get('AP50'),
                     best_detail.get('AP75'),
                     best_detail.get('AP_small'),
                     best_detail.get('AP_medium'),
                     best_detail.get('AP_large'),
                     best_detail.get('epoch'),
                     best_ckpt_name if best_ckpt_name else None,
                     'training_log'))
                print(f'  → 插入 val evaluation: mAP={best_detail["mAP"]}')

        # 记录结果
        results.append({
            'experiment_id': exp_id,
            'status': 'updated' if existing else 'inserted',
            'val': final_val,
            'epoch': final_epoch,
            'val_updated': val_updated,
            'val_skipped': val_skipped,
            'dataset': cfg_info['dataset'] if cfg_info else 'unknown',
            'coupling_type': cfg_info['coupling_type'] if cfg_info else 'unknown',
            'ot_epsilon': cfg_info['ot_epsilon'] if cfg_info else None,
            'aug': aug_info['pipeline_name'] if aug_info else 'unknown',
            'server': server,
        })

    conn.commit()

    # ─── 结果汇总 ────────────────────────────────────────────
    print(f'\n{"=" * 60}')
    print('=== 入库结果汇总 ===')
    print(f'{"experiment_id":<55} {"val":>6} {"ep":>4} {"ε":>4} {"dataset":>3} {"aug":>10} {"server":>18}')
    print('-' * 110)
    for r in results:
        eps_str = f'{r["ot_epsilon"]:.1f}' if r['ot_epsilon'] is not None else '?'
        val_str = f'{r["val"]:.3f}' if r['val'] is not None else 'None'
        ep_str = str(r['epoch']) if r['epoch'] is not None else '?'
        print(f'{r["experiment_id"]:<55} {val_str:>6} {ep_str:>4} {eps_str:>4} '
              f'{r["dataset"]:>3} {r["aug"]:>10} {r["server"]:>18}')

    # ─── 特别报告: ε=0.5 和 ε=3 (之前缺失的数据点) ──────────
    print(f'\n{"=" * 60}')
    print('=== 特别报告: ε=0.5 和 ε=3 (之前完全缺失的数据点) ===')
    for eps_target in [0.5, 3.0]:
        print(f'\n--- ε={eps_target} ---')
        c.execute('''SELECT e.experiment_id, e.best_val_mAP, e.best_val_epoch, e.server,
                            cfg.coupling_type, cfg.ot_epsilon, aug.pipeline_name
                     FROM experiment e
                     LEFT JOIN config cfg ON e.config_path = cfg.config_path
                     LEFT JOIN aug_pipeline aug ON cfg.aug_pipeline_hash = aug.pipeline_hash
                     WHERE cfg.ot_epsilon = ?
                     ORDER BY e.experiment_id''', (eps_target,))
        for r in c.fetchall():
            print(f'  {r[0]}: val={r[1]}, epoch={r[2]}, server={r[3]}, '
                  f'coupling={r[4]}, aug={r[6]}')

    # ─── 最终验证: DB 中所有 ε 消融实验 ──────────────────────
    print(f'\n{"=" * 60}')
    print('=== 最终验证: DB 中所有 ε 消融实验 ===')
    c.execute('''SELECT e.experiment_id, e.best_val_mAP, e.best_val_epoch,
                        cfg.ot_epsilon, cfg.dataset, cfg.coupling_type, aug.pipeline_name,
                        e.server
                 FROM experiment e
                 LEFT JOIN config cfg ON e.config_path = cfg.config_path
                 LEFT JOIN aug_pipeline aug ON cfg.aug_pipeline_hash = aug.pipeline_hash
                 WHERE cfg.ot_epsilon IS NOT NULL
                 ORDER BY cfg.ot_epsilon, e.experiment_id''')
    print(f'{"experiment_id":<55} {"val":>6} {"ep":>4} {"ε":>4} {"D":>3} {"coupling":>22} {"aug":>10} {"server":>18}')
    print('-' * 130)
    for r in c.fetchall():
        val_str = f'{r[1]:.3f}' if r[1] is not None else 'None'
        ep_str = str(r[2]) if r[2] is not None else '?'
        eps_str = f'{r[3]:.1f}' if r[3] is not None else '?'
        ds_str = r[4] or '?'
        coup_str = r[5] or '?'
        aug_str = r[6] or '?'
        srv_str = r[7] or '?'
        print(f'{r[0]:<55} {val_str:>6} {ep_str:>4} {eps_str:>4} {ds_str:>3} '
              f'{coup_str:>22} {aug_str:>10} {srv_str:>18}')

    conn.close()
    print(f'\n✅ 数据库更新完成')


if __name__ == '__main__':
    main()
