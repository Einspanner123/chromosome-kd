#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import time

import psutil


def get_gpu_memory_usage(gpu_id):
    """
    获取指定 GPU 的显存占用情况 (单位: MiB)
    """
    try:
        cmd = f'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_id}'
        output = (
            subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        )
        return int(output)
    except Exception as e:
        print(f'获取 GPU {gpu_id} 显存失败: {e}')
        return 999999  # 返回一个极大值防止误启动


def wait_for_process_and_memory(pid, gpu_ids, mem_threshold=500):
    """
    等待进程结束且显存释放
    """
    print(f'\033[1;33m开始监控进程 PID: {pid}，等待其结束...\033[0m')

    # 1. 等待进程结束
    try:
        process = psutil.Process(pid)
        while process.is_running():
            time.sleep(10)
    except psutil.NoSuchProcess:
        print(f'\033[1;31m进程 {pid} 已经结束或不存在。\033[0m')

    print(
        f'\033[1;33m进程 {pid} 已结束，正在检查 GPU {gpu_ids} 显存是否释放 (阈值: {mem_threshold} MiB)...\033[0m'
    )

    # 2. 等待显存释放
    while True:
        all_released = True
        for gid in gpu_ids:
            used_mem = get_gpu_memory_usage(gid)
            if used_mem > mem_threshold:
                print(
                    f'\033[1;35mGPU {gid} 仍有较多显存占用: {used_mem} MiB，等待中...\033[0m'
                )
                all_released = False
                break

        if all_released:
            print('\033[1;32m显存已释放，准备开始训练！\033[0m')
            break

        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(
        description='训练任务启动脚本（带进程监控功能）'
    )
    parser.add_argument(
        'config',
        help='训练配置文件路径 (例如: projects/LDMDet/configs/xxx.py)',
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default='0',
        help='指定 GPU ID，多个用逗号分隔 (默认: 0)',
    )
    parser.add_argument('--pid', type=int, help='要监控的进程 PID (可选)')
    parser.add_argument(
        '--mem-threshold',
        type=int,
        default=1000,
        help='显存释放阈值 MiB (默认: 1000)',
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default=None,
        help='工作目录 (例如: work_dirs/stability/ema)',
    )

    args = parser.parse_args()

    # 1. 校验配置文件
    if not os.path.exists(args.config):
        print(f'错误: 找不到配置文件 {args.config}')
        sys.exit(1)

    gpu_ids = args.gpu.split(',')

    # 2. 监控逻辑 (如果指定了 PID)
    if args.pid:
        wait_for_process_and_memory(args.pid, gpu_ids, args.mem_threshold)

    # 3. 构造训练命令
    # 模仿 train.sh 的环境变量设置
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu

    train_cmd = [sys.executable, 'tools/train.py', args.config]
    if args.work_dir:
        train_cmd.extend(['--work-dir', args.work_dir])

    print(f'\033[1;32m正在启动训练任务: {" ".join(train_cmd)}\033[0m')
    print(f'\033[1;34m环境变量: CUDA_VISIBLE_DEVICES={args.gpu}\033[0m')

    try:
        # 使用 subprocess.run 启动训练，并继承当前终端的输出
        subprocess.run(train_cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f'训练任务执行出错: {e}')
        sys.exit(1)
    except KeyboardInterrupt:
        print('\n收到停止信号，正在退出...')
        sys.exit(0)


if __name__ == '__main__':
    main()
