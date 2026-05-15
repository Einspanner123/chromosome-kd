import argparse
import os
import subprocess

import optuna
from mmengine.config import Config
from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Optuna hyperparameter tuning for LDMDet')
    parser.add_argument('config', help='base config file path')
    parser.add_argument(
        '--work-dir',
        help='directory to save tuning results',
        default='work_dirs/optuna_study',
    )
    parser.add_argument(
        '--n-trials', type=int, default=20, help='number of trials')
    parser.add_argument(
        '--gpus', type=int, default=1, help='number of gpus to use per trial')
    parser.add_argument(
        '--study-name', default='ldmdet_rf_tuning', help='optuna study name')
    return parser.parse_args()


class HparamManager:
    """超参数搜索空间管理，解耦参数定义与优化逻辑"""

    @staticmethod
    def suggest(trial):
        params = {}
        # Rectified Flow 相关
        params['model.bbox_head.rf_shift'] = trial.suggest_float(
            'rf_shift', 1.0, 5.0)
        params['model.bbox_head.snr_scale'] = trial.suggest_float(
            'snr_scale', 0.5, 3.0)

        # 优化器相关
        params['optim_wrapper.optimizer.lr'] = trial.suggest_float(
            'lr', 1e-5, 1e-4, log=True)
        params['optim_wrapper.optimizer.weight_decay'] = trial.suggest_float(
            'weight_decay', 1e-5, 1e-3, log=True)

        # 采样步数 (可选)
        # params['model.bbox_head.sampling_timesteps'] = trial.suggest_int('sampling_timesteps', 1, 8)

        return params


class MetricExtractor:
    """结果提取模块，支持多指标或自定义提取逻辑"""

    @staticmethod
    def get_best_map(trial_work_dir):
        import glob
        import json

        # 匹配 MMEngine 的标准日志结构
        log_files = glob.glob(
            os.path.join(trial_work_dir, '*', 'vis_data', 'scalars.json'))
        if not log_files:
            return 0.0

        best_map = 0.0
        # 总是取修改时间最近的日志
        latest_log = max(log_files, key=os.path.getmtime)

        try:
            with open(latest_log, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if 'coco/bbox_mAP' in data:
                        best_map = max(best_map, data['coco/bbox_mAP'])
        except Exception as e:
            print(f'Error extracting metric: {e}')
            return 0.0

        return best_map


class LDMDetObjective:

    def __init__(self, base_config_path, work_dir, gpus):
        self.base_config_path = base_config_path
        self.work_dir = work_dir
        self.gpus = gpus
        mkdir_or_exist(work_dir)

    def __call__(self, trial):
        # 1. 获取建议参数
        params = HparamManager.suggest(trial)

        # 2. 构造并保存配置文件
        cfg = Config.fromfile(self.base_config_path)

        # 使用嵌套键值对更新配置
        for key, value in params.items():
            self._set_cfg_value(cfg, key, value)

        trial_work_dir = os.path.join(self.work_dir, f'trial_{trial.number}')
        cfg.work_dir = trial_work_dir
        mkdir_or_exist(trial_work_dir)

        temp_config_path = os.path.join(trial_work_dir, 'config.py')
        cfg.dump(temp_config_path)

        # 3. 执行训练 (通过命令行隔离进程)
        print(f'\n[Trial {trial.number}] Running training...')
        train_cmd = self._build_cmd(temp_config_path)

        try:
            # check=True 会在命令失败时抛出异常
            subprocess.run(train_cmd, shell=True, check=True)
            # 4. 提取结果
            return MetricExtractor.get_best_map(trial_work_dir)
        except subprocess.CalledProcessError:
            return 0.0

    def _set_cfg_value(self, cfg, key_path, value):
        """支持 'a.b.c' 格式的属性设置"""
        parts = key_path.split('.')
        target = cfg
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value

    def _build_cmd(self, config_path):
        if self.gpus > 1:
            return f'bash tools/dist_train.sh {config_path} {self.gpus}'
        return f'python tools/train.py {config_path}'


def main():
    args = parse_args()

    # 确保工作目录存在，否则 SQLite 无法创建数据库文件
    mkdir_or_exist(args.work_dir)

    # 使用绝对路径避免 URI 解析歧义
    abs_work_dir = os.path.abspath(args.work_dir)
    storage_name = f"sqlite:///{os.path.join(abs_work_dir, 'optuna.db')}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_name,
        direction='maximize',
        load_if_exists=True,
    )

    objective = LDMDetObjective(args.config, args.work_dir, args.gpus)

    study.optimize(objective, n_trials=args.n_trials)

    print('\n' + '=' * 30)
    print('Tuning Finished!')
    print(f'Best Trial: {study.best_trial.number}')
    print(f'Best Value: {study.best_value}')
    print(f'Best Params: {study.best_params}')
    print('=' * 30)


if __name__ == '__main__':
    main()
