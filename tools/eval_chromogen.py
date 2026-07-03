"""Quick evaluation of chromogen_phase1 final model."""
import sys
import os
import torch
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from projects.ChromoGen.tools.train import load_config, build_model
from projects.ChromoGen.evaluation import ChromoGenEvaluator
from projects.ChromoGen.dataset.chromo_dataset import ChromoGenDataset, collate_fn
from torch.utils.data import DataLoader

CFG_PATH = 'projects/ChromoGen/configs/chromogen_phase1_imgonly.py'
CKPT_PATH = 'work_dirs/chromogen_phase1/final_model.pt'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load config
cfg = load_config(CFG_PATH)
output_dir = cfg.get('output_dir')
max_epochs = cfg.get('max_epochs')
print(f'Config loaded: output_dir={output_dir}')
print(f'max_epochs={max_epochs}')

# Load model
model = build_model(cfg)
checkpoint = torch.load(CKPT_PATH, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()
print(f'Model loaded from: {CKPT_PATH}')
epoch = checkpoint['epoch'] + 1
steps = checkpoint['global_step']
print(f'Trained for {epoch} epochs, {steps} steps')

# Build validation dataset + evaluator
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

num_eval_real = cfg.get('num_eval_real_images', 256)
evaluator = ChromoGenEvaluator.from_dataloader(val_loader, DEVICE, max_images=num_eval_real)
print(f'Evaluator ready with {num_eval_real} real images')

# Generate images
num_eval_gen = cfg.get('num_eval_gen_images', 256)
batch_size_gen = cfg.get('eval_gen_batch_size', 8)
gen_images_list = []
print(f'Generating {num_eval_gen} images...')
for start in range(0, num_eval_gen, batch_size_gen):
    n = min(batch_size_gen, num_eval_gen - start)
    idx = start % len(val_dataset)
    # Get conditions from val dataset
    batch = next(iter(val_loader))
    class_labels = batch['class_labels'][:n].to(DEVICE)
    counts = batch['counts'][:n].to(DEVICE)
    with torch.no_grad():
        result = model.generate(
            class_labels=class_labels,
            counts=counts,
            num_inference_steps=cfg.get('num_inference_steps', 50),
            guidance_scale=cfg.get('guidance_scale', 7.5),
            return_bboxes=False,
        )
    gen_images_list.append(result['images'].cpu())
    if (start // batch_size_gen + 1) % 4 == 0:
        print(f'  Generated {start + n}/{num_eval_gen}')

gen_images = torch.cat(gen_images_list, dim=0)
print(f'Generated images shape: {gen_images.shape}')

# Evaluate
metrics = evaluator.evaluate(gen_images)
print(f'\n========== Final Metrics ==========')
print(f'FID:    {metrics["fid"]:.2f}')
print(f'IS:     {metrics["is_mean"]:.2f} ± {metrics["is_std"]:.2f}')
print(f'===================================')
