"""ChromoGen训练入口

用法:
  python projects/ChromoGen/tools/train.py --config projects/ChromoGen/configs/chromogen_phase1_imgonly.py
  python projects/ChromoGen/tools/train.py --config projects/ChromoGen/configs/chromogen_phase2_joint.py
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from projects.ChromoGen.dataset.chromo_dataset import (
    ChromoGenDataset,
    collate_fn,
)
from projects.ChromoGen.evaluation import ChromoGenEvaluator
from projects.ChromoGen.models.chromogen_pipeline import ChromoGenPipeline

# SwanLab集成
try:
    import swanlab

    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False


def parse_args():
    parser = argparse.ArgumentParser(description='ChromoGen Training')
    parser.add_argument(
        '--config', type=str, required=True, help='配置文件路径'
    )
    parser.add_argument(
        '--resume', type=str, default=None, help='恢复训练的checkpoint路径'
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument(
        '--swanlab_project',
        type=str,
        default='chromosome-kd',
        help='SwanLab项目名',
    )
    parser.add_argument(
        '--swanlab_experiment',
        type=str,
        default=None,
        help='SwanLab实验名(默认自动生成)',
    )
    parser.add_argument(
        '--no_swanlab', action='store_true', help='禁用SwanLab日志'
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """从Python配置文件加载配置"""
    config = {}
    # 处理 _base_ 继承
    with open(config_path) as f:
        content = f.read()

    # 递归处理 _base_
    if '_base_' in content:
        import ast

        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '_base_':
                        base_paths = ast.literal_eval(node.value)
                        if isinstance(base_paths, str):
                            base_paths = [base_paths]
                        for bp in base_paths:
                            base_path = os.path.join(
                                os.path.dirname(config_path), bp
                            )
                            base_config = load_config(base_path)
                            config.update(base_config)

    # 执行配置文件
    with open(config_path) as _f:
        exec(compile(_f.read(), config_path, 'exec'), config)
    # 清理内置变量
    for k in list(config.keys()):
        if k.startswith('__'):
            del config[k]
    return config


def set_seed(seed: int):
    """设置随机种子"""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f'{seconds:.1f}s'
    elif seconds < 3600:
        return f'{seconds // 60:.0f}m{seconds % 60:.0f}s'
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f'{h:.0f}h{m:.0f}m'


def get_gpu_memory() -> str:
    """获取GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f'GPU: {allocated:.1f}/{reserved:.1f}GB'
    return ''


def build_model(cfg: dict) -> ChromoGenPipeline:
    """根据配置构建模型"""
    model = ChromoGenPipeline(
        vae_model=cfg.get('vae_model', 'stabilityai/sd-vae-ft-mse'),
        sample_size=cfg.get('sample_size', 96),
        unet_block_out_channels=cfg.get(
            'unet_block_out_channels', (320, 640, 1280, 1280)
        ),
        unet_attention_head_dim=cfg.get('unet_attention_head_dim', 8),
        cross_attention_dim=cfg.get('cross_attention_dim', 768),
        gradient_checkpointing=cfg.get('gradient_checkpointing', True),
        condition_embed_dim=cfg.get('condition_embed_dim', 768),
        condition_max_count=cfg.get('condition_max_count', 50),
        condition_dropout=cfg.get('condition_dropout', 0.1),
        enable_bbox_head=cfg.get('enable_bbox_head', False),
        bbox_feat_channels=cfg.get('bbox_feat_channels', 512),
        bbox_num_proposals=cfg.get('bbox_num_proposals', 100),
        bbox_num_heads=cfg.get('bbox_num_heads', 8),
        bbox_num_layers=cfg.get('bbox_num_layers', 3),
        bbox_snr_scale=cfg.get('bbox_snr_scale', 2.0),
        num_train_timesteps=cfg.get('num_train_timesteps', 1000),
        noise_schedule=cfg.get('noise_schedule', 'linear'),
        prediction_type=cfg.get('prediction_type', 'epsilon'),
        lambda_img=cfg.get('lambda_img', 1.0),
        lambda_bbox=cfg.get('lambda_bbox', 0.5),
        lambda_cls=cfg.get('lambda_cls', 0.5),
        cfg_dropout=cfg.get('cfg_dropout', 0.1),
    )
    return model


class EMAModel:
    """指数移动平均"""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply_shadow(self, model: torch.nn.Module):
        """将EMA参数应用到模型"""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        """恢复原始参数"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def train():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)

    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Using device: {device}')
    print(f'Config: enable_bbox_head={cfg.get("enable_bbox_head", False)}')

    # SwanLab初始化
    use_swanlab = HAS_SWANLAB and not args.no_swanlab
    if use_swanlab:
        experiment_name = (
            args.swanlab_experiment
            or f'ChromoGen-{"Phase2" if cfg.get("enable_bbox_head") else "Phase1"}'
        )
        swanlab.init(
            project=args.swanlab_project,
            experiment_name=experiment_name,
            config={
                'enable_bbox_head': cfg.get('enable_bbox_head', False),
                'image_size': cfg.get('image_size', 768),
                'batch_size': cfg.get('batch_size', 8),
                'learning_rate': cfg.get('learning_rate', 1e-4),
                'max_epochs': cfg.get('max_epochs', 100),
                'num_train_timesteps': cfg.get('num_train_timesteps', 1000),
                'prediction_type': cfg.get('prediction_type', 'epsilon'),
                'lambda_img': cfg.get('lambda_img', 1.0),
                'lambda_bbox': cfg.get('lambda_bbox', 0.5),
                'lambda_cls': cfg.get('lambda_cls', 0.5),
                'guidance_scale': cfg.get('guidance_scale', 7.5),
                'seed': args.seed,
            },
        )
        print(
            f'SwanLab initialized: project={args.swanlab_project}, experiment={experiment_name}'
        )
    elif not HAS_SWANLAB:
        print('SwanLab not installed, skipping experiment logging')

    # 构建数据集
    train_dataset = ChromoGenDataset(
        data_root=cfg['data_root'],
        ann_file=cfg['train_ann_file'],
        img_dir=cfg['train_img_dir'],
        image_size=cfg.get('image_size', 768),
        enable_bbox=cfg.get('enable_bbox_head', False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get('batch_size', 8),
        shuffle=True,
        num_workers=cfg.get('num_workers', 8),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    print(f'Training samples: {len(train_dataset)}')

    # 构建验证集和评估器
    evaluator = None
    val_ann_file = cfg.get('val_ann_file', 'valid/_annotations.coco.json')
    val_img_dir = cfg.get('val_img_dir', 'valid')
    if os.path.exists(os.path.join(cfg['data_root'], val_ann_file)):
        val_dataset = ChromoGenDataset(
            data_root=cfg['data_root'],
            ann_file=val_ann_file,
            img_dir=val_img_dir,
            image_size=cfg.get('image_size', 768),
            enable_bbox=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.get('batch_size', 8),
            shuffle=False,
            num_workers=cfg.get('num_workers', 8),
            collate_fn=collate_fn,
            pin_memory=True,
        )
        num_eval_real = cfg.get('num_eval_real_images', 256)
        evaluator = ChromoGenEvaluator.from_dataloader(
            val_loader, device, max_images=num_eval_real
        )
        print(
            f'Validation samples: {len(val_dataset)}, evaluator ready with {num_eval_real} real images'
        )
    else:
        print(
            f'No validation set found at {val_ann_file}, skipping evaluation'
        )

    # 构建模型
    model = build_model(cfg)
    model = model.to(device)

    # 可训练参数统计
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Trainable params: {trainable_params:,} / Total: {total_params:,}')

    # 优化器
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(),
        lr=cfg.get('learning_rate', 1e-4),
        weight_decay=cfg.get('weight_decay', 0.01),
        betas=(cfg.get('adam_beta1', 0.9), cfg.get('adam_beta2', 0.999)),
        eps=cfg.get('adam_epsilon', 1e-8),
    )

    # 学习率调度器
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )

    warmup_steps = cfg.get('lr_warmup_steps', 500)
    max_epochs = cfg.get('max_epochs', 100)
    total_steps = max_epochs * len(train_loader)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.001, total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7
    )
    scheduler = SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], [warmup_steps]
    )

    # EMA
    ema = EMAModel(model, decay=cfg.get('ema_decay', 0.9999))

    # AMP
    scaler = GradScaler(enabled=cfg.get('fp16', True))

    # 恢复训练
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        print(f'Resumed from epoch {start_epoch}')

    # 输出目录
    output_dir = cfg.get('output_dir', 'work_dirs/chromogen')
    os.makedirs(output_dir, exist_ok=True)

    # 训练循环
    print(f'{"=" * 60}')
    print(
        f'Training started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    )
    print(
        f'Total epochs: {max_epochs}, Steps/epoch: {len(train_loader)}, Total steps: {total_steps}'
    )
    print(
        f'Batch size: {cfg.get("batch_size", 8)}, FP16: {cfg.get("fp16", True)}'
    )
    print(
        f'Noise schedule: {cfg.get("noise_schedule", "linear")}, Prediction: {cfg.get("prediction_type", "epsilon")}'
    )
    print(f'{"=" * 60}')

    train_start_time = time.time()
    model.train()
    for epoch in range(start_epoch, max_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_loss_img = 0.0
        epoch_loss_bbox = 0.0
        epoch_loss_cls = 0.0

        # 滑动窗口计时统计
        step_times = []

        for step, batch in enumerate(train_loader):
            step_start_time = time.time()

            pixel_values = batch['pixel_values'].to(device)
            class_labels = batch['class_labels'].to(device)
            counts = batch['counts'].to(device)

            gt_bboxes = None
            gt_labels = None
            if cfg.get('enable_bbox_head', False) and 'bboxes' in batch:
                # 将bbox转为cxcywh归一化格式
                gt_bboxes = []
                gt_labels = []
                for i, (bboxes, labels) in enumerate(
                    zip(batch['bboxes'], batch['labels'])
                ):
                    if bboxes.numel() > 0:
                        orig_h, orig_w = batch['orig_sizes'][i]
                        # xyxy → cxcywh归一化
                        bboxes_f = bboxes.float()
                        x1, y1, x2, y2 = (
                            bboxes_f[:, 0],
                            bboxes_f[:, 1],
                            bboxes_f[:, 2],
                            bboxes_f[:, 3],
                        )
                        cx = ((x1 + x2) / 2) / orig_w
                        cy = ((y1 + y2) / 2) / orig_h
                        w = (x2 - x1) / orig_w
                        h = (y2 - y1) / orig_h
                        gt_bboxes.append(
                            torch.stack([cx, cy, w, h], dim=-1).to(device)
                        )
                        gt_labels.append(labels.to(device))
                    else:
                        gt_bboxes.append(bboxes.to(device))
                        gt_labels.append(labels.to(device))

            # 前向传播
            with autocast(enabled=cfg.get('fp16', True)):
                losses = model(
                    pixel_values=pixel_values,
                    class_labels=class_labels,
                    counts=counts,
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels,
                )

            loss = losses['loss_total'] / cfg.get(
                'gradient_accumulation_steps', 1
            )

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA更新
            ema.update(model)

            # 学习率调度
            scheduler.step()

            # 记录损失
            epoch_loss += loss.item()
            epoch_loss_img += losses['loss_img'].item()
            if 'loss_bbox' in losses:
                epoch_loss_bbox += losses['loss_bbox'].item()
            if 'loss_cls' in losses:
                epoch_loss_cls += losses['loss_cls'].item()

            global_step += 1

            # 记录步时
            step_time = time.time() - step_start_time
            step_times.append(step_time)

            # 日志
            log_every = cfg.get('log_every_n_steps', 50)
            if global_step % log_every == 0:
                lr = optimizer.param_groups[0]['lr']
                # 计算速度 (最近N步的平均)
                recent_steps = min(log_every, len(step_times))
                avg_step_time = sum(step_times[-recent_steps:]) / recent_steps
                steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                samples_per_sec = steps_per_sec * cfg.get('batch_size', 8)

                # ETA计算
                elapsed = time.time() - train_start_time
                steps_done = global_step - (start_epoch * len(train_loader))
                steps_total = total_steps
                steps_remaining = steps_total - steps_done
                eta_seconds = (
                    (steps_remaining / steps_per_sec)
                    if steps_per_sec > 0
                    else 0
                )

                # 当前epoch ETA
                epoch_steps_done = step + 1
                epoch_steps_total = len(train_loader)
                epoch_eta = (
                    ((epoch_steps_total - epoch_steps_done) * avg_step_time)
                    if avg_step_time > 0
                    else 0
                )

                gpu_info = get_gpu_memory()

                print(
                    f'Epoch [{epoch + 1}/{max_epochs}] Step [{step + 1}/{len(train_loader)}] '
                    f'Loss: {loss.item():.4f} (img: {losses["loss_img"].item():.4f}'
                    + (
                        f', bbox: {losses.get("loss_bbox", torch.tensor(0)).item():.4f}'
                        if cfg.get('enable_bbox_head')
                        else ''
                    )
                    + f') LR: {lr:.2e}'
                )
                print(
                    f'  Speed: {steps_per_sec:.1f} step/s | {samples_per_sec:.1f} samples/s | '
                    f'Step: {format_time(avg_step_time)} | '
                    f'Epoch ETA: {format_time(epoch_eta)} | '
                    f'Total ETA: {format_time(eta_seconds)} | '
                    f'Elapsed: {format_time(elapsed)} | '
                    f'{gpu_info}'
                )

                # SwanLab日志
                if use_swanlab:
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/loss_img': losses['loss_img'].item(),
                        'train/lr': lr,
                        'train/epoch': epoch,
                        'train/steps_per_sec': steps_per_sec,
                        'train/samples_per_sec': samples_per_sec,
                    }
                    if 'loss_bbox' in losses:
                        log_dict['train/loss_bbox'] = losses[
                            'loss_bbox'
                        ].item()
                    if 'loss_cls' in losses:
                        log_dict['train/loss_cls'] = losses['loss_cls'].item()
                    swanlab.log(log_dict, step=global_step)

        # Epoch统计
        epoch_time = time.time() - epoch_start_time
        n_steps = len(train_loader)
        avg_step_time = sum(step_times) / len(step_times) if step_times else 0
        elapsed_total = time.time() - train_start_time
        eta_total = (
            ((max_epochs - epoch - 1) * epoch_time) if epoch_time > 0 else 0
        )

        print(f'--- Epoch [{epoch + 1}/{max_epochs}] Complete ---')
        print(
            f'  Avg Loss: {epoch_loss / n_steps:.4f} '
            f'Img: {epoch_loss_img / n_steps:.4f}'
            + (
                f' BBox: {epoch_loss_bbox / n_steps:.4f} Cls: {epoch_loss_cls / n_steps:.4f}'
                if cfg.get('enable_bbox_head')
                else ''
            )
        )
        print(
            f'  Time: {format_time(epoch_time)} | '
            f'Avg Step: {format_time(avg_step_time)} | '
            f'Elapsed: {format_time(elapsed_total)} | '
            f'ETA: {format_time(eta_total)} | '
            f'Finish: {(datetime.now() + timedelta(seconds=eta_total)).strftime("%Y-%m-%d %H:%M")} | '
            f'{get_gpu_memory()}'
        )

        # SwanLab epoch日志
        if use_swanlab:
            epoch_log = {
                'train/epoch_avg_loss': epoch_loss / n_steps,
                'train/epoch_avg_loss_img': epoch_loss_img / n_steps,
                'train/epoch_time': epoch_time,
            }
            if cfg.get('enable_bbox_head'):
                epoch_log['train/epoch_avg_loss_bbox'] = (
                    epoch_loss_bbox / n_steps
                )
                epoch_log['train/epoch_avg_loss_cls'] = (
                    epoch_loss_cls / n_steps
                )
            swanlab.log(epoch_log, step=global_step)

        # 保存checkpoint
        if (epoch + 1) % cfg.get('save_every_n_epochs', 10) == 0:
            save_path = os.path.join(
                output_dir, f'checkpoint_epoch_{epoch + 1}.pt'
            )
            torch.save(
                {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': cfg,
                },
                save_path,
            )
            print(f'Saved checkpoint: {save_path}')

        # 采样可视化
        if (epoch + 1) % cfg.get('sample_every_n_epochs', 5) == 0:
            model.eval()
            ema.apply_shadow(model)
            with torch.no_grad():
                sample_labels = class_labels[: cfg.get('num_sample_images', 4)]
                sample_counts = counts[: cfg.get('num_sample_images', 4)]
                result = model.generate(
                    class_labels=sample_labels,
                    counts=sample_counts,
                    num_inference_steps=cfg.get('num_inference_steps', 50),
                    guidance_scale=cfg.get('guidance_scale', 7.5),
                    return_bboxes=cfg.get('enable_bbox_head', False),
                )
                # 保存样本图像
                from torchvision.utils import save_image

                sample_dir = os.path.join(output_dir, 'samples')
                os.makedirs(sample_dir, exist_ok=True)
                sample_path = os.path.join(
                    sample_dir, f'epoch_{epoch + 1}.png'
                )
                save_image(result['images'], sample_path, nrow=2)
                # SwanLab图像日志
                if use_swanlab:
                    swanlab.log(
                        {
                            'samples/generated': swanlab.Image(sample_path),
                        },
                        step=global_step,
                    )

                # 评估指标
                if (
                    evaluator is not None
                    and (epoch + 1) % cfg.get('eval_every_n_epochs', 10) == 0
                ):
                    num_eval_gen = cfg.get('num_eval_gen_images', 256)
                    batch_size_gen = cfg.get('eval_gen_batch_size', 8)
                    gen_images_list = []
                    # 分批生成以控制显存
                    for start in range(0, num_eval_gen, batch_size_gen):
                        n = min(batch_size_gen, num_eval_gen - start)
                        # 循环使用验证集条件
                        idx = start % len(val_dataset)
                        batch_labels = class_labels[:n]
                        batch_counts = counts[:n]
                        gen_result = model.generate(
                            class_labels=batch_labels,
                            counts=batch_counts,
                            num_inference_steps=cfg.get(
                                'num_inference_steps', 50
                            ),
                            guidance_scale=cfg.get('guidance_scale', 7.5),
                            return_bboxes=False,
                        )
                        gen_images_list.append(gen_result['images'].cpu())
                    gen_images = torch.cat(gen_images_list, dim=0)
                    metrics = evaluator.evaluate(gen_images)
                    print(
                        f'[Eval Epoch {epoch + 1}] FID: {metrics["fid"]:.2f}, IS: {metrics["is_mean"]:.2f}±{metrics["is_std"]:.2f}'
                    )
                    if use_swanlab:
                        swanlab.log(
                            {
                                'eval/fid': metrics['fid'],
                                'eval/is_mean': metrics['is_mean'],
                                'eval/is_std': metrics['is_std'],
                                'eval/epoch': epoch + 1,
                            },
                            step=global_step,
                        )
            ema.restore(model)
            model.train()

    # 保存最终模型
    total_time = time.time() - train_start_time
    final_path = os.path.join(output_dir, 'final_model.pt')
    torch.save(
        {
            'epoch': max_epochs - 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'config': cfg,
        },
        final_path,
    )
    print(f'{"=" * 60}')
    print('Training complete!')
    print(f'  Total time: {format_time(total_time)}')
    print(f'  Final model: {final_path}')
    print(f'  Finished at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"=" * 60}')

    # 关闭SwanLab
    if use_swanlab:
        swanlab.finish()


if __name__ == '__main__':
    train()
