"""LDMDet 专用 mmdet Hooks — 可视化、SwanLab、代码备份"""

import json
import os

from mmengine.hooks import Hook
from mmdet.registry import HOOKS


# ──────────────────────────────────────────────
# CopyProjectHook — 训练前备份 ldmdet 代码
# ──────────────────────────────────────────────

@HOOKS.register_module(force=True)
class CopyProjectHook(Hook):
    def __init__(self, src_path='ldmdet', dst_name='ldmdet_backup'):
        self.src_path = src_path
        self.dst_name = dst_name

    def before_run(self, runner):
        import datetime
        import shutil
        timestamp = getattr(runner, 'timestamp', None)
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dst_path = os.path.join(runner.work_dir, timestamp, self.dst_name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        abs_src = os.path.abspath(self.src_path)
        runner.logger.info(f'Backing up ldmdet from {abs_src} to {dst_path}')
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        try:
            shutil.copytree(abs_src, dst_path, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            runner.logger.info('Backup complete.')
        except Exception as e:
            runner.logger.error(f'Backup failed: {e}')


# ──────────────────────────────────────────────
# PredictionVisHook — 固定图片集 + 全覆盖类 + SwanLab 上传
# ──────────────────────────────────────────────

@HOOKS.register_module(force=True)
class PredictionVisHook(Hook):
    """验证可视化 Hook。

    首次 epoch 自动扫描验证集，选取最少图片覆盖全部 24 类染色体，
    固定该图片集贯穿所有 epoch，用于 SwanLab 前后对比。
    """

    def __init__(self, num_images: int = 6, score_thr: float = 0.01, num_classes: int = 24):
        self.num_images = num_images
        self.score_thr = score_thr
        self.num_classes = num_classes
        self._ref_images = None       # 固定图片路径列表
        self._ref_loaded = False
        self._epoch_frames = {}       # epoch → [(stem, img_path, gt_bboxes, gt_labels, gt_labels_set)]

    # ── 图片选择 (首次 epoch) ───────────────────

    def before_val(self, runner):
        """验证开始前，确保已选定参考图片集"""
        from pathlib import Path
        ref_path = Path(runner.work_dir) / 'vis_reference.json'

        if self._ref_images is not None:
            return  # 已选定

        if ref_path.exists():
            with open(ref_path) as f:
                self._ref_images = json.load(f)
            runner.logger.info(f'[VisHook] loaded {len(self._ref_images)} reference images from {ref_path}')
            return

        # 首次：扫描验证集，贪心选择覆盖全类的最少图片
        val_loader = runner.val_dataloader
        if val_loader is None:
            runner.logger.warning('[VisHook] no val dataloader, skip reference selection')
            self._ref_images = []
            return

        selected = self._select_reference_images(val_loader, runner)
        self._ref_images = selected

        with open(ref_path, 'w') as f:
            json.dump(selected, f, indent=2)
        runner.logger.info(
            f'[VisHook] selected {len(selected)} reference images '
            f'covering classes: checking... saved to {ref_path}'
        )

    def _select_reference_images(self, val_loader, runner):
        """贪心选择最少图片覆盖全部类别"""
        import numpy as np

        # 收集所有图片的 class coverage
        candidates = []  # (img_path, set of class_ids)
        all_classes = set(range(self.num_classes))
        covered = set()

        for batch in val_loader:
            data_samples = batch.get('data_samples', [])
            for ds in data_samples:
                img_path = ds.img_path if hasattr(ds, 'img_path') else None
                if img_path is None:
                    continue
                gt = ds.gt_instances if hasattr(ds, 'gt_instances') else None
                if gt is None or len(gt.labels) == 0:
                    continue
                classes = set(gt.labels.cpu().tolist())
                candidates.append((img_path, classes))
                covered |= classes

        runner.logger.info(
            f'[VisHook] validation set: {len(candidates)} images, '
            f'{len(covered)}/{self.num_classes} classes present'
        )

        # 贪心：每次选覆盖最多未覆盖类的图片
        selected = []
        uncovered = all_classes
        remaining = candidates[:]

        while uncovered and remaining and len(selected) < self.num_images:
            best_idx = max(
                range(len(remaining)),
                key=lambda i: len(remaining[i][1] & uncovered)
            )
            best_path, best_classes = remaining.pop(best_idx)
            new = best_classes & uncovered
            if not new:
                continue
            selected.append(best_path)
            uncovered -= new

        runner.logger.info(
            f'[VisHook] greedy selection: {len(selected)} images '
            f'cover {self.num_classes - len(uncovered)}/{self.num_classes} classes'
        )
        return selected

    # ── 捕获预测 ────────────────────────────────

    def before_val_epoch(self, runner):
        self._epoch_frames[runner.epoch] = []

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self._ref_images is None:
            return
        if outputs is None or data_batch is None:
            return
        data_samples = data_batch.get('data_samples', [])
        if not data_samples:
            return

        ref_set = set(self._ref_images)
        epoch = runner.epoch

        if isinstance(outputs, list):
            for i, out in enumerate(outputs):
                if i >= len(data_samples):
                    break
                self._capture_if_reference(data_samples[i], out, ref_set, epoch)
        elif hasattr(outputs, 'pred_instances') and data_samples:
            self._capture_if_reference(data_samples[0], outputs, ref_set, epoch)

    def _capture_if_reference(self, ds, result, ref_set, epoch):
        img_path = ds.img_path if hasattr(ds, 'img_path') else None
        if img_path is None or img_path not in ref_set:
            return

        gt = ds.gt_instances if hasattr(ds, 'gt_instances') else None
        gt_bboxes = gt.bboxes.cpu().numpy() if gt is not None and len(gt.bboxes) > 0 else None
        gt_labels = gt.labels.cpu().numpy() if gt is not None and len(gt.labels) > 0 else None
        pred = result.pred_instances if hasattr(result, 'pred_instances') else None
        if pred is None or len(pred.bboxes) == 0:
            return

        pred_bboxes = pred.bboxes.cpu().numpy()
        pred_scores = pred.scores.cpu().numpy()
        mask = pred_scores >= self.score_thr
        if not mask.any():
            return

        from pathlib import Path
        stem = Path(img_path).stem
        scale_factor = ds.scale_factor if hasattr(ds, 'scale_factor') else None

        self._epoch_frames[epoch].append({
            'stem': stem,
            'img_path': img_path,
            'gt_bboxes': gt_bboxes,
            'gt_labels': gt_labels,
            'pred_bboxes': pred_bboxes[mask],
            'pred_scores': pred_scores[mask],
            'scale_factor': scale_factor,
        })

    # ── 绘制 + SwanLab ──────────────────────────

    def after_val_epoch(self, runner, metrics=None):
        from pathlib import Path
        import cv2
        import numpy as np

        epoch = runner.epoch
        frames = self._epoch_frames.get(epoch, [])
        if not frames:
            return

        vis_dir = Path(runner.work_dir) / 'vis_predictions'
        epoch_dir = vis_dir / f'epoch_{epoch:03d}'
        epoch_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.RandomState(42)
        colors = {i: tuple(int(c) for c in rng.randint(50, 255, 3)) for i in range(self.num_classes)}

        for item in frames:
            img = cv2.imread(item['img_path'])
            if img is None:
                continue

            scale_factor = item['scale_factor']
            if isinstance(scale_factor, (np.ndarray, list, tuple)) and len(scale_factor) >= 2:
                sx, sy = float(scale_factor[0]), float(scale_factor[1])
            else:
                sx = sy = 1.0

            # GT 框 (彩色，带标签)
            if item['gt_bboxes'] is not None:
                for bbox, label in zip(item['gt_bboxes'], item['gt_labels']):
                    x1, y1, x2, y2 = [int(v / f) for v, f in zip(bbox, [sx, sy, sx, sy])]
                    color = colors.get(int(label), (255, 0, 0))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f'GT:{int(label)}', (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 预测框 (绿色，带分数)
            for bbox, score in zip(item['pred_bboxes'], item['pred_scores']):
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{score:.2f}', (x1, y2 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            # 保存帧
            save_path = epoch_dir / f'{item["stem"]}.jpg'
            cv2.imwrite(str(save_path), img)

        runner.logger.info(f'[VisHook] epoch {epoch}: {len(frames)} frames saved')

        # 上传 SwanLab
        self._log_to_swanlab(epoch_dir, epoch, runner)

    # ── SwanLab 上传 ────────────────────────────

    def _log_to_swanlab(self, epoch_dir, epoch, runner):
        try:
            import swanlab
            import cv2
            from pathlib import Path

            jpgs = sorted(Path(epoch_dir).glob('*.jpg'))[:self.num_images]
            for p in jpgs:
                img = cv2.imread(str(p))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    swanlab.log({f'predictions/{p.stem}': swanlab.Image(img)}, step=epoch)
        except Exception:
            pass


# ──────────────────────────────────────────────
# WeightSummaryHook — 权重/梯度监控 (SwanLab)
# ──────────────────────────────────────────────

@HOOKS.register_module(force=True)
class WeightSummaryHook(Hook):
    def __init__(self, interval=50, log_norm=True, log_heatmap=True):
        self.interval = interval
        self.log_norm = log_norm
        self.log_heatmap = log_heatmap

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self.every_n_train_iters(runner, self.interval):
            return
        import swanlab
        import numpy as np

        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        layer_norms = {}
        grad_energies = []
        all_grads = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if self.log_norm:
                    layer_norms[f'weights_norm/{name}'] = param.data.norm(2).item()
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    layer_norms[f'grads_norm/{name}'] = grad_norm
                    grad_energies.append(grad_norm)
                    all_grads.append(param.grad.detach().cpu().numpy().ravel())

        if layer_norms:
            swanlab.log(layer_norms, step=runner.iter)

        if self.log_heatmap and grad_energies and runner.rank == 0:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(np.arange(len(grad_energies)), grad_energies, color='skyblue')
                ax.set_yscale('log')
                ax.set_title(f'Gradient Norms (Iter {runner.iter})')
                ax.set_xlabel('Layer Index')
                ax.set_ylabel('Norm (Log Scale)')
                swanlab.log({'gradient_activity': swanlab.Image(fig)}, step=runner.iter)
                plt.close(fig)

                if all_grads:
                    concat = np.concatenate(all_grads)
                    if len(concat) > 10000:
                        concat = np.random.choice(concat, 10000, replace=False)
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.hist(concat, bins=50, color='steelblue', edgecolor='white')
                    ax2.set_title(f'Gradient Distribution (Iter {runner.iter})')
                    ax2.set_xlabel('Gradient Value')
                    ax2.set_ylabel('Count')
                    swanlab.log({'gradient_distribution': swanlab.Image(fig2)}, step=runner.iter)
                    plt.close(fig2)
            except Exception:
                pass


# ──────────────────────────────────────────────
# 诊断注入器 — 将方向特定诊断回调注入到模型
# ──────────────────────────────────────────────

@HOOKS.register_module(force=True)
class CouplingDiagInjector(Hook):
    """方向一诊断注入器: 将 CouplingDiagnosticsCallback 注入到 head.

    在 before_run 时创建 callback, 注入到 head.coupling_diag_callback,
    并将其 collect 方法注册到 TrainingDiagnosticsHook.diagnostics_callback.
    """

    def __init__(self, interval=100):
        self.interval = interval
        self._callback = None

    def before_run(self, runner):
        from ldmdet.diagnostics.coupling_diag import CouplingDiagnosticsCallback
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        head = model.bbox_head

        self._callback = CouplingDiagnosticsCallback(interval=self.interval)
        head.coupling_diag_callback = self._callback

        # 注册到 TrainingDiagnosticsHook
        for hook in runner.hooks:
            if hook.__class__.__name__ == 'TrainingDiagnosticsHook':
                hook.diagnostics_callback = lambda runner, outputs, step: self._callback.collect(step)
                break

        runner.logger.info('CouplingDiagnosticsCallback injected.')


@HOOKS.register_module(force=True)
class CountDiagInjector(Hook):
    """方向二诊断注入器: 将 CountDiagnosticsCallback 注入到 head."""

    def __init__(self, interval=100):
        self.interval = interval
        self._callback = None

    def before_run(self, runner):
        from ldmdet.diagnostics.count_diag import CountDiagnosticsCallback
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        head = model.bbox_head

        self._callback = CountDiagnosticsCallback(interval=self.interval)
        head.count_diag_callback = self._callback

        for hook in runner.hooks:
            if hook.__class__.__name__ == 'TrainingDiagnosticsHook':
                hook.diagnostics_callback = lambda runner, outputs, step: self._callback.collect(step)
                break

        runner.logger.info('CountDiagnosticsCallback injected.')


@HOOKS.register_module(force=True)
class SNRDiagInjector(Hook):
    """方向三诊断注入器: 将 SNRDiagnosticsCallback 注入到 criterion."""

    def __init__(self, interval=100):
        self.interval = interval
        self._callback = None

    def before_run(self, runner):
        from ldmdet.diagnostics.snr_diag import SNRDiagnosticsCallback
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        criterion = model.bbox_head.criterion

        self._callback = SNRDiagnosticsCallback(interval=self.interval)
        criterion.snr_diag_callback = self._callback

        for hook in runner.hooks:
            if hook.__class__.__name__ == 'TrainingDiagnosticsHook':
                hook.diagnostics_callback = lambda runner, outputs, step: self._callback.collect(step)
                break

        runner.logger.info('SNRDiagnosticsCallback injected.')


@HOOKS.register_module(force=True)
class TrajectoryDiagInjector(Hook):
    """方向四诊断注入器: 将 TrajectoryDiagnosticsCallback 注入到 head.

    注入到 head.trajectory_diag_callback, 由 head 在 loss() 中
    调用 update_scale/update_ot/update_curvature 更新数据.
    """

    def __init__(self, interval=100):
        self.interval = interval
        self._callback = None

    def before_run(self, runner):
        from ldmdet.diagnostics.trajectory_diag import TrajectoryDiagnosticsCallback
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        head = model.bbox_head

        self._callback = TrajectoryDiagnosticsCallback(interval=self.interval)
        head.trajectory_diag_callback = self._callback

        for hook in runner.hooks:
            if hook.__class__.__name__ == 'TrainingDiagnosticsHook':
                hook.diagnostics_callback = lambda runner, outputs, step: self._callback.collect(step)
                break

        runner.logger.info('TrajectoryDiagnosticsCallback injected.')


@HOOKS.register_module(force=True)
class HierarchicalDiagInjector(Hook):
    """方向五诊断注入器: 将 HierarchicalDiagnosticsCallback 注入到 head.

    注入到 head.hierarchical_diag_callback, 由 head 在 loss() 中
    调用 update/update_group_embeddings 更新数据.
    """

    def __init__(self, interval=100, class_indices_per_group=None):
        self.interval = interval
        self.class_indices_per_group = class_indices_per_group
        self._callback = None

    def before_run(self, runner):
        from ldmdet.diagnostics.hierarchical_diag import HierarchicalDiagnosticsCallback
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        head = model.bbox_head

        self._callback = HierarchicalDiagnosticsCallback(
            interval=self.interval,
            class_indices_per_group=self.class_indices_per_group,
        )
        head.hierarchical_diag_callback = self._callback

        for hook in runner.hooks:
            if hook.__class__.__name__ == 'TrainingDiagnosticsHook':
                hook.diagnostics_callback = lambda runner, outputs, step: self._callback.collect(step)
                break

        runner.logger.info('HierarchicalDiagnosticsCallback injected.')
