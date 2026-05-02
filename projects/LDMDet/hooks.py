import os
import shutil

from mmengine.hooks import Hook

from mmdet.registry import HOOKS


def _copytree_safe(src, dst, ignore=None):
    """Robust copytree that ignores OS permission errors."""
    if ignore is None:
        ignore = shutil.ignore_patterns()

    os.makedirs(dst, exist_ok=True)
    errors = []

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if ignore is not None and ignore(src, [item]):
            continue
        if os.path.isdir(s):
            _copytree_safe(s, d, ignore)
        else:
            try:
                shutil.copy2(s, d)
            except OSError:
                try:
                    shutil.copy(s, d)
                except OSError:
                    errors.append((s, d, "copy failed entirely"))

    try:
        shutil.copystat(src, dst)
    except OSError:
        errors.append((src, dst, "metadata copy skipped"))

    return errors


@HOOKS.register_module()
class CopyProjectHook(Hook):
    """每次训练开始前将项目代码备份到 work_dir。

    这样可以确保每个实验对应的代码版本都被记录下来。
    """

    def __init__(self, src_path="projects/LDMDet", dst_name="LDMDet_backup"):
        self.src_path = src_path
        self.dst_name = dst_name

    def before_run(self, runner):
        timestamp = getattr(runner, "timestamp", None)

        if timestamp is None:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        dst_path = os.path.join(runner.work_dir, timestamp, self.dst_name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        abs_src_path = os.path.abspath(self.src_path)

        runner.logger.info(
            f"Backing up project code from {abs_src_path} to {dst_path}..."
        )

        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)

        try:
            ignore = shutil.ignore_patterns(
                "__pycache__", "*.pyc", "work_dirs", "data"
            )
            errors = _copytree_safe(abs_src_path, dst_path, ignore=ignore)
            if errors:
                runner.logger.warning(
                    f"Backup completed with {len(errors)} permission warnings "
                    f"(files copied but metadata skipped)"
                )
            else:
                runner.logger.info("Project code backup completed.")
        except Exception as e:
            runner.logger.error(f"Failed to backup project code: {str(e)}")
