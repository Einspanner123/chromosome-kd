import os
import shutil

from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class CopyProjectHook(Hook):
    """每次训练开始前将项目代码备份到 work_dir。

    这样可以确保每个实验对应的代码版本都被记录下来。
    """

    def __init__(self, src_path='projects/LDMDet', dst_name='LDMDet_backup'):
        self.src_path = src_path
        self.dst_name = dst_name

    def before_run(self, runner):
        # 尝试获取时间戳
        timestamp = getattr(runner, 'timestamp', None)

        # 如果 runner 没有 timestamp，尝试手动生成一个符合 MMEngine 习惯的时间戳
        if timestamp is None:
            import datetime

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # 构建目标路径：work_dir / timestamp / dst_name
        dst_path = os.path.join(runner.work_dir, timestamp, self.dst_name)

        # 确保目录存在
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # 获取项目根目录下的绝对路径
        # 考虑到是在项目根目录下执行训练，这里使用相对路径转绝对路径
        abs_src_path = os.path.abspath(self.src_path)

        runner.logger.info(
            f'Backing up project code from {abs_src_path} to {dst_path}...'
        )

        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)

        try:
            # 排除 pycache 和其他不需要的文件夹
            shutil.copytree(
                abs_src_path,
                dst_path,
                ignore=shutil.ignore_patterns(
                    '__pycache__', '*.pyc', 'work_dirs', 'data'
                ),
            )
            runner.logger.info('Project code backup completed.')
        except Exception as e:
            runner.logger.error(f'Failed to backup project code: {e!s}')


@HOOKS.register_module()
class PredictionVisHook(Hook):
    """验证阶段可视化预测框，保存带标注的图片。

    在 after_val_iter 中捕获预测结果，在 after_val_epoch 中统一绘制保存。
    不重新运行推理，直接使用验证循环中的预测结果。
    """

    def __init__(self, num_images: int = 4, score_thr: float = 0.01):
        self.num_images = num_images
        self.score_thr = score_thr
        self._captured = []  # (img_path, gt_bboxes, gt_labels, pred_bboxes, pred_scores, scale_factor)

    def before_val_epoch(self, runner):
        self._captured = []

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if len(self._captured) >= self.num_images:
            return
        if outputs is None or data_batch is None:
            return

        data_samples = data_batch.get('data_samples', [])
        if not data_samples:
            return

        # outputs 是 val_step 的返回值，包含预测结果
        if isinstance(outputs, list):
            for i, out in enumerate(outputs):
                if len(self._captured) >= self.num_images:
                    break
                if i >= len(data_samples):
                    break
                self._capture_single(data_samples[i], out)
        elif hasattr(outputs, 'pred_instances'):
            if data_samples:
                self._capture_single(data_samples[0], outputs)

    def _capture_single(self, ds, result):
        """从单个 data_sample 和 result 中提取信息。"""
        gt = ds.gt_instances if hasattr(ds, 'gt_instances') else None
        gt_bboxes = gt.bboxes.cpu().numpy() if gt is not None and len(gt.bboxes) > 0 else None
        gt_labels = gt.labels.cpu().numpy() if gt is not None and len(gt.labels) > 0 else None

        pred = result.pred_instances if hasattr(result, 'pred_instances') else None
        if pred is None or len(pred.bboxes) == 0:
            return

        pred_bboxes = pred.bboxes.cpu().numpy()
        pred_scores = pred.scores.cpu().numpy()

        # 过滤低分
        mask = pred_scores >= self.score_thr
        if not mask.any():
            return

        scale_factor = ds.scale_factor if hasattr(ds, 'scale_factor') else None

        self._captured.append({
            'img_path': ds.img_path if hasattr(ds, 'img_path') else None,
            'gt_bboxes': gt_bboxes,
            'gt_labels': gt_labels,
            'pred_bboxes': pred_bboxes[mask],
            'pred_scores': pred_scores[mask],
            'scale_factor': scale_factor,
        })

    def after_val_epoch(self, runner, metrics=None):
        import cv2
        import numpy as np
        from pathlib import Path

        if not self._captured:
            runner.logger.warning('[VisHook] 没有捕获到有效的预测结果')
            return

        vis_dir = Path(runner.work_dir) / 'vis_predictions'
        vis_dir.mkdir(parents=True, exist_ok=True)

        epoch = runner.epoch

        rng = np.random.RandomState(42)
        colors = {
            i: tuple(int(c) for c in rng.randint(50, 255, 3))
            for i in range(24)
        }

        saved = 0
        for item in self._captured:
            if saved >= self.num_images:
                break
            img_path = item['img_path']
            if img_path is None:
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            scale_factor = item['scale_factor']
            if isinstance(scale_factor, np.ndarray) and len(scale_factor) >= 2:
                sx, sy = float(scale_factor[0]), float(scale_factor[1])
            elif isinstance(scale_factor, (list, tuple)) and len(scale_factor) >= 2:
                sx, sy = float(scale_factor[0]), float(scale_factor[1])
            else:
                sx = sy = 1.0

            # 画 GT (彩色)
            if item['gt_bboxes'] is not None:
                for j, (bbox, label) in enumerate(zip(item['gt_bboxes'], item['gt_labels'])):
                    x1, y1, x2, y2 = bbox
                    x1 = int(x1 / sx)
                    y1 = int(y1 / sy)
                    x2 = int(x2 / sx)
                    y2 = int(y2 / sy)
                    color = colors.get(int(label), (255, 0, 0))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f'GT:{int(label)}', (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 画预测框 (绿色)
            for j, (bbox, score) in enumerate(zip(item['pred_bboxes'], item['pred_scores'])):
                x1, y1, x2, y2 = bbox
                x1 = int(x1 / sx)
                y1 = int(y1 / sy)
                x2 = int(x2 / sx)
                y2 = int(y2 / sy)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{score:.2f}', (x1, y2 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            save_name = f'epoch_{epoch:03d}_img{saved}_{Path(img_path).stem}.jpg'
            save_path = vis_dir / save_name
            cv2.imwrite(str(save_path), img)
            runner.logger.info(
                f'[VisHook] 保存: {save_name} '
                f'(GT={len(item["gt_bboxes"]) if item["gt_bboxes"] is not None else 0}, '
                f'Pred={len(item["pred_bboxes"])})'
            )
            saved += 1

        self._captured = []


@HOOKS.register_module()
class WeightSummaryHook(Hook):
    """可视化模型内部权重和梯度的变化。

    在 SwanLab 中以折线图形式显示权重的 L2 范数，并生成梯度热点图。
    """

    def __init__(self, interval=50, log_norm=True, log_heatmap=True):
        self.interval = interval
        self.log_norm = log_norm
        self.log_heatmap = log_heatmap

    def after_train_iter(
        self, runner, batch_idx: int, data_batch=None, outputs=None
    ):
        if not self.every_n_train_iters(runner, self.interval):
            return

        import matplotlib.pyplot as plt
        import numpy as np
        import swanlab

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        message_hub = runner.message_hub

        layer_names = []
        grad_energies = []

        layer_norms = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 记录权重范数
                if self.log_norm:
                    norm = param.data.norm(2).item()
                    layer_norms[f'weights_norm/{name}'] = norm

                # 如果有梯度，记录梯度范数
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    layer_norms[f'grads_norm/{name}'] = grad_norm
                    layer_names.append(name)
                    grad_energies.append(grad_norm)

        # 直接通过 swanlab.log 记录，绕过 message_hub 以确保图表生成
        if layer_norms:
            swanlab.log(layer_norms, step=runner.iter)

        # 生成梯度分布图 (Bar Chart)
        if self.log_heatmap and len(grad_energies) > 0:
            try:
                # 仅在主进程记录图片，避免 DDP 重复绘图
                if runner.rank == 0:
                    import matplotlib.pyplot as plt

                    # 1. 绘制梯度范数柱状图 (比 1D 热点图直观得多)
                    fig, ax = plt.subplots(figsize=(12, 6))
                    indices = np.arange(len(grad_energies))
                    ax.bar(indices, grad_energies, color='skyblue')
                    ax.set_yscale('log')  # 梯度量级差异大，用对数坐标
                    ax.set_title(
                        f'Gradient Norms across Layers (Iter {runner.iter})'
                    )
                    ax.set_xlabel('Layer Index')
                    ax.set_ylabel('Norm (Log Scale)')

                    # 2. 记录到 SwanLab (Media 选项卡)
                    swanlab.log(
                        {'gradient_activity': swanlab.Image(fig)},
                        step=runner.iter,
                    )
                    plt.close(fig)

                    # 3. 记录梯度直方图 (SwanLab 原生支持)
                    # 选取所有梯度的展平值进行分布分析
                    all_grads = []
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            all_grads.append(
                                param.grad.detach().cpu().numpy().flatten()
                            )

                    if all_grads:
                        concat_grads = np.concatenate(all_grads)
                        # 限制样本量，避免上传过大
                        if len(concat_grads) > 10000:
                            concat_grads = np.random.choice(
                                concat_grads, 10000, replace=False
                            )

                        swanlab.log(
                            {
                                'gradient_distribution': swanlab.Histogram(
                                    concat_grads
                                )
                            },
                            step=runner.iter,
                        )
            except Exception as e:
                runner.logger.warning(f'WeightSummaryHook Error: {e!s}')
