#!/usr/bin/env python3
"""Run one BoxChart-RF optimizer step without starting a training job."""

import os
import sys


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mmengine.config import Config
from mmengine.runner import Runner


CONFIG = (
    'experiments/configs/ldmdet/directions/boxchart_rf/'
    'boxchart_rf_24obj.py'
)


def main():
    cfg = Config.fromfile(CONFIG)
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    cfg.train_dataloader.pop('prefetch_factor', None)
    cfg.work_dir = '/tmp/boxchart_train_smoke'
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer', vis_backends=cfg.vis_backends,
        name='visualizer')

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    model = runner.model
    model.train()
    data = next(iter(runner.train_dataloader))
    optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
    log_vars = model.train_step(data, optim_wrapper)
    print('SMOKE_OK')
    for key, value in log_vars.items():
        print(key, float(value) if hasattr(value, 'item') else value)
    head = model.module.bbox_head if hasattr(model, 'module') else model.bbox_head
    print('chart_mean', head.box_chart.mean.tolist())
    finite = all(
        parameter.grad is None or parameter.grad.isfinite().all().item()
        for parameter in model.parameters())
    print('grad_finite', finite)


if __name__ == '__main__':
    main()
