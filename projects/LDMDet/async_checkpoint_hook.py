"""Async checkpoint hook for mmengine 0.10.5 compatibility.

Wraps the checkpoint save in a background thread so the training loop
does not block on disk I/O.  Handles ``async_save`` locally instead of
passing it through to ``Runner.save_checkpoint()`` (which does not
accept that kwarg in mmengine 0.10.5).
"""

import logging
import threading
from typing import Optional

from mmengine.dist import is_main_process
from mmengine.hooks import CheckpointHook
from mmengine.logging import print_log

from mmdet.registry import HOOKS

log = logging.getLogger(__name__)


@HOOKS.register_module()
class AsyncCheckpointHook(CheckpointHook):
    """CheckpointHook that saves in a background thread.

    All parameters are identical to :class:`CheckpointHook`, except that
    ``async_save`` is accepted but **ignored** — it is always asynchronous.
    """

    def __init__(self, *args, **kwargs):
        # Pop async_save so it never reaches self.args / kwargs
        kwargs.pop('async_save', None)
        super().__init__(*args, **kwargs)
        self._save_thread: Optional[threading.Thread] = None
        self._save_error: Optional[Exception] = None

    # ------------------------------------------------------------------
    # Override the two places that call runner.save_checkpoint()
    # ------------------------------------------------------------------

    def _save_checkpoint_with_step(self, runner, step, meta=None):
        """Dispatch save into background thread, cleanup old ckpt inline."""
        self._wait_prev_save(runner)

        # ── Cleanup old checkpoint (from mmengine CheckpointHook) ──
        if self.max_keep_ckpts > 0 and hasattr(self, 'keep_ckpt_ids'):
            # Do not save the same step twice (e.g. best ckpt uses same step)
            if len(self.keep_ckpt_ids) > 0 and self.keep_ckpt_ids[-1] == step:
                pass
            else:
                if len(self.keep_ckpt_ids) == self.max_keep_ckpts:
                    _step = self.keep_ckpt_ids.popleft()
                    if is_main_process():
                        ckpt_path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(_step)
                        )
                        if self.file_backend.isfile(ckpt_path):
                            self.file_backend.remove(ckpt_path)
                        elif self.file_backend.isdir(ckpt_path):
                            self.file_backend.rmtree(ckpt_path)
                self.keep_ckpt_ids.append(step)
                runner.message_hub.update_info(
                    'keep_ckpt_ids', list(self.keep_ckpt_ids)
                )

        ckpt_filename = self.filename_tmpl.format(step)

        self._start_save_thread(
            runner=runner,
            out_dir=self.out_dir,
            filename=ckpt_filename,
            file_client_args=self.file_client_args,
            save_optimizer=self.save_optimizer,
            save_param_scheduler=self.save_param_scheduler,
            meta=meta,
            by_epoch=self.by_epoch,
            backend_args=self.backend_args,
        )

        last_ckpt = self.file_backend.join_path(self.out_dir, ckpt_filename)
        self.last_ckpt = last_ckpt
        runner.message_hub.update_info('last_ckpt', self.last_ckpt)

        # Write last_checkpoint marker (fast)
        import os.path as osp

        save_file = osp.join(runner.work_dir, 'last_checkpoint')
        with open(save_file, 'w') as f:
            f.write(last_ckpt)

    def _save_best_checkpoint(self, runner, metrics) -> None:
        """Save best checkpoint (lightweight, keep sync for simplicity)."""
        # Best-checkpoint logic rarely blocks — let it run inline
        super()._save_best_checkpoint(runner, metrics)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wait_prev_save(self, runner):
        """Wait for previous async save, re-raising any error."""
        if self._save_thread is not None and self._save_thread.is_alive():
            print_log(
                'Previous async checkpoint save still in progress, waiting…',
                logger='current',
            )
            self._save_thread.join()
        if self._save_error is not None:
            err = self._save_error
            self._save_error = None
            raise err

    def _start_save_thread(self, **kwargs):
        """Launch save in a daemon thread."""
        self._save_error = None
        self._save_thread = threading.Thread(
            target=self._do_save,
            args=(kwargs,),
            daemon=True,
        )
        self._save_thread.start()

    def _do_save(self, kw: dict):
        """Actual save, runs in background thread."""
        try:
            runner = kw.pop('runner')
            runner.save_checkpoint(**kw)
        except Exception as e:
            self._save_error = e

    def after_run(self, runner):
        """Wait for in-flight save before exit."""
        self._wait_prev_save(runner)
