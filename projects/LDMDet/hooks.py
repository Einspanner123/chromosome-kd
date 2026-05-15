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
