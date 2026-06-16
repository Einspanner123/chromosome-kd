#!/usr/bin/env python3
"""LDMDet 推理速度 Benchmark
测量不同 coupling 策略 + 不同采样步数下的 FPS 和延迟。
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from mmdet.utils import register_all_modules; register_all_modules()
from mmengine.config import Config

def benchmark(config_path, checkpoint_path, steps_list, warmup=10, repeat=50):
    cfg = Config.fromfile(config_path)
    import experiments.mmdet_bridge.registry
    from mmdet.registry import MODELS

    model = MODELS.build(cfg.model).cuda().eval()
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cuda:0')
        model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)

    dummy = torch.randn(1, 3, 800, 1216).cuda()

    for steps in steps_list:
        model.bbox_head._sampler.sampling_timesteps = steps

        # warmup
        for _ in range(warmup):
            with torch.no_grad():
                model.predict(dummy, [type('M',(),{'img_shape':(800,1216),'scale_factor':None})()], rescale=False)

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(repeat):
            with torch.no_grad():
                model.predict(dummy, [type('M',(),{'img_shape':(800,1216),'scale_factor':None})()], rescale=False)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        fps = repeat / elapsed
        latency = elapsed / repeat * 1000
        print(f'{steps:2d} steps: {fps:5.1f} FPS  {latency:6.1f} ms')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('--checkpoint', default='')
    p.add_argument('--steps', default='1,2,4,8')
    args = p.parse_args()
    benchmark(args.config, args.checkpoint, [int(s) for s in args.steps.split(',')])
