"""Benchmark 24obj: DiffusionDet DDPM baseline — matches old benchmark_diffusiondet_24obj settings"""
_base_ = ['../../ldmdet/directions/nonlinear_trajectory/rf_heun_adaln.py']

# === Override to benchmark_24obj dataset ===
data_root = 'data/24_chromosomes_object/coco/'

classes = (
    'A1', 'A2', 'A3',
    'B4', 'B5',
    'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
    'D13', 'D14', 'D15',
    'E16', 'E17', 'E18',
    'F19', 'F20',
    'G21', 'G22',
    'X', 'Y',
)

METAINFO = dict(
    classes=classes,
    palette=[
        (220, 20, 60), (119, 11, 32), (0, 0, 142),
        (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192),
        (250, 170, 30), (100, 170, 30), (220, 220, 0),
        (175, 116, 175), (250, 0, 30), (165, 42, 42),
        (255, 77, 255), (0, 226, 252), (182, 182, 255),
        (0, 82, 0), (120, 166, 157),
        (110, 76, 0), (174, 57, 255),
        (199, 100, 0), (72, 0, 118),
    ],
)

# Update dataloader paths to 24obj data_root
train_dataloader = dict(
    dataset=dict(data_root=data_root),
)
val_dataloader = dict(
    dataset=dict(data_root=data_root),
)
val_evaluator = dict(
    ann_file=data_root + 'valid/_annotations.coco.json',
)
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# === DDPM settings (matching old benchmark_diffusiondet_24obj config) ===
model = dict(bbox_head=dict(
    diffusion_type='ddpm',
    solver_type='euler',
    sampling_timesteps=1,
    single_head=dict(time_conditioning='scale_shift'),
    use_ensemble=False,
))
