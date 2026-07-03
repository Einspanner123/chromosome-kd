"""Quick evaluation of chromogen_phase1 final model - reduced images."""
import sys
import os
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from projects.ChromoGen.tools.train import load_config, build_model
from projects.ChromoGen.evaluation import ChromoGenEvaluator
from projects.ChromoGen.dataset.chromo_dataset import ChromoGenDataset, collate_fn
from torch.utils.data import DataLoader

CFG_PATH = 'projects/ChromoGen/configs/chromogen_phase1_imgonly.py'
CKPT_PATH = 'work_dirs/chromogen_phase1/final_model.pt'
DEVICE = torch.device('cuda:0')

cfg = load_config(CFG_PATH)

# Load checkpoint info
checkpoint = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
epoch = checkpoint['epoch'] + 1
steps = checkpoint['global_step']
print(f'Model trained: {epoch} epochs, {steps} steps')
output_dir = cfg.get('output_dir')
print(f'Config output_dir: {output_dir}')
max_epochs_saved = checkpoint.get('config', {}).get('max_epochs', '?')
print(f'max_epochs in saved config: {max_epochs_saved}')

# Build model
model = build_model(cfg)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()
print('Model loaded successfully')

# Generate quick sample to test
print('Generating 8 sample images...')
class_labels = torch.zeros(8, 24, dtype=torch.long, device=DEVICE)
counts = torch.zeros(8, 24, dtype=torch.long, device=DEVICE)
class_labels[:, 0] = 0
counts[:, 0] = 5

with torch.no_grad():
    result = model.generate(
        class_labels=class_labels,
        counts=counts,
        num_inference_steps=10,
        guidance_scale=7.5,
        return_bboxes=False,
    )
gen_shape = result['images'].shape
print(f'Generated images shape: {gen_shape}')
print('Generation works!')

# Build evaluator
val_dataset = ChromoGenDataset(
    data_root=cfg['data_root'],
    ann_file=cfg['val_ann_file'],
    img_dir=cfg['val_img_dir'],
    image_size=cfg.get('image_size', 768),
    enable_bbox=False,
)
val_loader = DataLoader(
    val_dataset, batch_size=8, shuffle=False,
    num_workers=4, collate_fn=collate_fn, pin_memory=True,
)

num_eval_real = min(64, cfg.get('num_eval_real_images', 256))
evaluator = ChromoGenEvaluator.from_dataloader(val_loader, DEVICE, max_images=num_eval_real)
print(f'Evaluator ready with {num_eval_real} real images')

# Generate evaluation images
num_eval_gen = 64
batch_size_gen = 8
gen_images_list = []

print(f'Generating {num_eval_gen} images for FID...')
for start in range(0, num_eval_gen, batch_size_gen):
    n = min(batch_size_gen, num_eval_gen - start)
    cl = class_labels[:n]
    ct = counts[:n]
    with torch.no_grad():
        result = model.generate(
            class_labels=cl,
            counts=ct,
            num_inference_steps=50,
            guidance_scale=7.5,
            return_bboxes=False,
        )
    gen_images_list.append(result['images'].cpu())
    print(f'  Generated {start + n}/{num_eval_gen}')

gen_images = torch.cat(gen_images_list, dim=0)

# Evaluate
metrics = evaluator.evaluate(gen_images)
print()
print('=' * 50)
print('Evaluation Results (64 images)')
print('=' * 50)
fid = metrics['fid']
is_mean = metrics['is_mean']
is_std = metrics['is_std']
print(f'FID:    {fid:.2f}')
print(f'IS:     {is_mean:.2f} +/- {is_std:.2f}')
print('=' * 50)
print('Note: Metrics with 64 images are approximate.')
print('Full 256-image eval would be more reliable.')
