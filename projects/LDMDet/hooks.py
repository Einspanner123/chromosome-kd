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

        for name, param in model.named_parameters():
            if param.requires_grad:
                # 记录权重范数
                if self.log_norm:
                    norm = param.data.norm(2).item()
                    message_hub.update_scalar(f'weights_norm/{name}', norm)

                # 如果有梯度，记录梯度范数
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    message_hub.update_scalar(f'grads_norm/{name}', grad_norm)
                    layer_names.append(name)
                    grad_energies.append(grad_norm)

        # 生成梯度热点图 (Heatmap)
        if self.log_heatmap and len(grad_energies) > 0:
            try:
                plt.figure(figsize=(10, 8))
                # 将梯度归一化或取 log 以便观察
                data = np.array(grad_energies).reshape(-1, 1)
                plt.imshow(data, aspect='auto', cmap='hot')
                plt.colorbar(label='Gradient Norm')
                plt.title(f'Gradient Heatmap at Iter {runner.iter}')
                plt.ylabel('Layer Index')

                # 记录到 SwanLab
                swanlab.log(
                    {'gradient_activity': swanlab.Image(plt)}, step=runner.iter
                )
                plt.close()
            except Exception:
                pass
