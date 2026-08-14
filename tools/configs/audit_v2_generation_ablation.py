#!/usr/bin/env python3
"""Verify the prespecified scientific factors in the G0--G3 chain."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.experiments.v2_matrix import resolve_config  # noqa: E402

MATRICES = [
    'experiments/configs/v2/matrices/d1_inhouse1700_generation_ablation.yaml',
    'experiments/configs/v2/matrices/d2_taichung_generation_ablation.yaml',
]
METHODS = [
    'strict_g0_ddpm_linear_scaleshift',
    'strict_g1_rf_linear_scaleshift',
    'strict_g2_rf_shifted_scaleshift',
    'karyoflow',
]
EXPECTED = [
    ('ddpm', 'euler', 1, None, None, 'scale_shift'),
    ('rectified_flow', 'dpm_solver_pp', 4, 'linear', 1.0, 'scale_shift'),
    ('rectified_flow', 'dpm_solver_pp', 4, 'shifted', 3.0, 'scale_shift'),
    ('rectified_flow', 'dpm_solver_pp', 4, 'shifted', 3.0, 'adaln_zero'),
]


def signature(cfg):
    head = cfg.model.bbox_head
    return (head.diffusion_type, head.solver_type, head.sampling_timesteps,
            head.get('rf_schedule'), head.get('rf_shift'),
            head.single_head.time_conditioning)


def main() -> int:
    errors = []
    for matrix in MATRICES:
        configs = [resolve_config(matrix, method, 42)[0] for method in METHODS]
        signatures = [signature(cfg) for cfg in configs]
        if signatures != EXPECTED:
            errors.append(f'{matrix}: signatures={signatures!r}')
        reference = configs[0]
        for cfg in configs[1:]:
            for key in ('train_dataloader', 'val_dataloader', 'test_dataloader',
                        'optim_wrapper', 'param_scheduler', 'train_cfg'):
                if cfg.get(key) != reference.get(key):
                    errors.append(f'{matrix}: {key} differs across G stages')
        # G1->G2 changes only the time schedule; G2->G3 only conditioning.
        g1, g2, g3 = [cfg.model.bbox_head.to_dict() for cfg in configs[1:]]
        g1['rf_schedule'], g1['rf_shift'] = g2['rf_schedule'], g2['rf_shift']
        if g1 != g2:
            errors.append(f'{matrix}: G1->G2 contains non-schedule changes')
        g2['single_head']['time_conditioning'] = g3['single_head']['time_conditioning']
        if g2 != g3:
            errors.append(f'{matrix}: G2->G3 contains non-conditioning changes')
    if errors:
        print('V2 GENERATION ABLATION AUDIT: FAIL')
        for error in errors:
            print('-', error)
        return 1
    print('V2 GENERATION ABLATION AUDIT: PASS matrices=2 stages=4')
    print('G0->G1 is the prespecified DDPM-to-RF formulation boundary; '
          'G1->G2 isolates time shifting; G2->G3 isolates AdaLN-Zero.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
