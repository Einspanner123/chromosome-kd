"""离线求解器对比: Heun (8 NFE) vs DPM-Solver++ (7/9 NFE)。

直接加载 SOTA checkpoint，在验证集上对比推理精度。
不重新训练——纯粹比较求解器。
mmengine 的冗长输出被重定向到临时日志，结果写入专用 results.txt。

用法:
  CUDA_VISIBLE_DEVICES=0 conda run -n chromo python projects/LDMDet/tools/compare_solver.py
  cat work_dirs/_compare_solver/results.txt   # 查看结果
"""

import io
import os
import re
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

_PROJECT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _PROJECT)

import mmdet  # noqa
import projects.LDMDet.mods  # noqa

CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'configs',
    '_legacy',
    'ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py',
)
CKPT = (
    'work_dirs/reproduce_0751_stochot_eps5_v2/best_coco_bbox_mAP_epoch_59.pth'
)
_TMP_BASE = 'work_dirs/_compare_solver'
_RESULT_FILE = os.path.join(_PROJECT, _TMP_BASE, 'results.txt')


def _parse_coco_metrics(text: str) -> dict:
    """从 COCO evaluator 输出中解析 AP/AR。"""
    m = {}
    for pattern, key in [
        (
            r'AP.*IoU=0\.50:0\.95.*area=\s+all\s+\|\s+maxDets=100\s*\]\s*=\s*([\d.]+)',
            'AP',
        ),
        (
            r'AP.*IoU=0\.50\s+\|\s+area=\s+all\s+\|\s+maxDets=1000\s*\]\s*=\s*([\d.]+)',
            'AP50',
        ),
        (
            r'AP.*IoU=0\.75\s+\|\s+area=\s+all\s+\|\s+maxDets=1000\s*\]\s*=\s*([\d.]+)',
            'AP75',
        ),
        (
            r'AP.*IoU=0\.50:0\.95.*area=\s+small\s+\|\s+maxDets=1000\s*\]\s*=\s*([\d.]+)',
            'APs',
        ),
        (
            r'AP.*IoU=0\.50:0\.95.*area=medium\s+\|\s+maxDets=1000\s*\]\s*=\s*([\d.]+)',
            'APm',
        ),
        (
            r'AP.*IoU=0\.50:0\.95.*area=\s+large\s+\|\s+maxDets=1000\s*\]\s*=\s*([\d.]+)',
            'APl',
        ),
        (
            r'AR.*IoU=0\.50:0\.95.*area=\s+all\s+\|\s+maxDets=100\s*\]\s*=\s*([\d.]+)',
            'AR',
        ),
    ]:
        match = re.search(pattern, text)
        if match:
            m[key] = float(match.group(1))
    return m


def run_eval(config_path, ckpt_path, solver_type, sampling_timesteps, label):
    from mmengine.config import Config

    cfg = Config.fromfile(config_path)
    cfg.model.bbox_head.solver_type = solver_type
    cfg.model.bbox_head.sampling_timesteps = sampling_timesteps
    nfe = (
        sampling_timesteps * 2
        if solver_type == 'heun'
        else (sampling_timesteps + 1)
    )

    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = cfg.val_evaluator

    work_dir = os.path.join(_PROJECT, _TMP_BASE, label.replace(' ', '_'))
    os.makedirs(work_dir, exist_ok=True)
    cfg.work_dir = work_dir
    cfg.load_from = ckpt_path
    cfg.visualizer = None
    cfg.log_level = 'ERROR'

    from mmengine.runner import Runner

    verbose_log = os.path.join(_PROJECT, _TMP_BASE, f'verbose_{label}.log')
    buf = io.StringIO()
    with open(verbose_log, 'w') as vl:
        vl.write(
            f'=== {label}: {solver_type}, steps={sampling_timesteps}, NFE~{nfe} ===\n\n'
        )
        with redirect_stdout(buf), redirect_stderr(buf):
            try:
                runner = Runner.from_cfg(cfg)
                runner.test()
            except Exception as e:
                import traceback

                traceback.print_exc()
        vl.write(buf.getvalue())

    metrics = _parse_coco_metrics(buf.getvalue())
    metrics['NFE'] = nfe
    metrics['label'] = label
    shutil.rmtree(work_dir, ignore_errors=True)
    return metrics


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    os.makedirs(os.path.join(_PROJECT, _TMP_BASE), exist_ok=True)

    results = []
    for label, solver, steps in [
        ('Heun 4-step (8 NFE)', 'heun', 4),
        ('DPM-2 6-step (7 NFE)', 'dpm_solver_pp', 6),
        ('DPM-2 8-step (9 NFE)', 'dpm_solver_pp', 8),
    ]:
        print(f'Running {label} ...')
        m = run_eval(CONFIG, CKPT, solver, steps, label.replace(' ', '_'))
        results.append(m)
        ap = m.get('AP', '?')
        print(f'  -> mAP = {ap}')

    lines = []
    lines.append(
        f'=== 求解器离线对比结果 === ({datetime.now().strftime("%Y-%m-%d %H:%M")})'
    )
    lines.append(f'权重: {CKPT}')
    lines.append('')
    header = f'{"配置":<28} {"NFE":>5} {"mAP":>8} {"AP50":>8} {"AP75":>8} {"APs":>8} {"APm":>8} {"APl":>8} {"AR":>8}'
    lines.append(header)
    lines.append('-' * len(header))
    for m in results:
        lines.append(
            f'{m.get("label", "?"):<28} {m.get("NFE", "?"):>5} '
            f'{m.get("AP", "?"):>8} {m.get("AP50", "?"):>8} {m.get("AP75", "?"):>8} '
            f'{m.get("APs", "?"):>8} {m.get("APm", "?"):>8} {m.get("APl", "?"):>8} '
            f'{m.get("AR", "?"):>8}'
        )

    # Delta vs Heun
    if results:
        heun_ap = results[0].get('AP', None)
        if heun_ap:
            lines.append('')
            lines.append('Delta vs Heun baseline:')
            for m in results[1:]:
                ap = m.get('AP', None)
                if ap:
                    lines.append(
                        f'  {m.get("label", "?"):<28} mAP = {ap:+.3f}'
                    )

    out = '\n'.join(lines)
    with open(_RESULT_FILE, 'w') as f:
        f.write(out + '\n')

    print(f'\n结果已写入: {_RESULT_FILE}')
    print(out)


if __name__ == '__main__':
    main()
