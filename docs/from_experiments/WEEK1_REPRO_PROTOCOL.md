# LDMDet Week 1 Reproduction Protocol

## Goal

Week 1 has three deliverables:

1. Reproduce the historical best `eps=5` stochastic Sinkhorn run.
2. Use one benchmark protocol for all later speed comparisons.
3. Stop relying on stale `work_dirs` artifacts as the only source of truth.

## Source Of Truth

- Historical best run: `work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5/20260423_060304`
- Historical best metric: `bbox mAP = 0.751 @ epoch 86`
- Reconstructed config: `projects/LDMDet/configs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py`
- Historical backup code snapshot:
  `work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5/20260423_060304/LDMDet_backup`

## Known Drift

- The main branch had lost `ot_coupling / ot_matcher / ot_epsilon / ot_num_iters / ot_sample`
  support in `projects/LDMDet/mods/diffusiondet_head.py`.
- The original config file was missing from `projects/LDMDet/configs`.
- `train_monitor_queue.json` still contains stale status entries and must not be treated as runtime truth.

## Reproduction Command

Single-GPU train:

```bash
python tools/train.py \
  projects/LDMDet/configs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py \
  --work-dir work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5_repro
```

Recommended 3-seed sweep:

```bash
python tools/train.py \
  projects/LDMDet/configs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py \
  --work-dir work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5_seed1 \
  --cfg-options randomness.seed=3407
```

```bash
python tools/train.py \
  projects/LDMDet/configs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py \
  --work-dir work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5_seed2 \
  --cfg-options randomness.seed=3408
```

```bash
python tools/train.py \
  projects/LDMDet/configs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py \
  --work-dir work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5_seed3 \
  --cfg-options randomness.seed=3409
```

## Reproduction Acceptance

- Minimum success: model builds and starts training with the reconstructed config.
- Good reproduction: `best mAP >= 0.748`.
- Strong reproduction: mean over 3 seeds is within `0.003` of the historical `0.751`.

## Benchmark Protocol

Use one hardware/software environment for all later comparisons:

- GPU: same device for every run
- Precision: keep AMP choice fixed per table
- Input size: use the config default resize path
- Batch size: report exactly what is used
- Split: validation set for AP, fixed subset for latency if needed

Always report:

- config path
- checkpoint path
- seed
- best `mAP / AP50 / AP75`
- training wall-clock duration
- inference latency in ms/image
- FPS
- max CUDA memory

## Benchmark Script Status

- Existing script: `projects/LDMDet/tests/benchmark.py`
- Reusable part: timing pattern and CUDA event measurement
- Not acceptable as final benchmark source:
  it builds a synthetic hard-coded model instead of loading the actual experiment config/checkpoint

## Week 1 Checklist

- [ ] Run a config-build sanity check
- [ ] Launch 1 seed reproduction
- [ ] Verify that validation metrics are logged correctly
- [ ] Freeze benchmark protocol before any speed ablation
- [ ] Record results in one shared table
