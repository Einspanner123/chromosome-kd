#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import psutil


LOG_FILE = "train_monitor_queue.json"


def get_gpu_memory_usage(gpu_id):
    try:
        cmd = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_id}"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
        return int(output)
    except Exception as e:
        print(f"获取 GPU {gpu_id} 显存失败: {e}")
        return 999999


def wait_for_process_and_memory(pid, gpu_ids, mem_threshold=1000):
    print(f"\033[1;33m[Monitor] 开始监控进程 PID: {pid}，等待其结束...\033[0m")

    try:
        process = psutil.Process(pid)
        while process.is_running():
            time.sleep(10)
    except psutil.NoSuchProcess:
        print(f"\033[1;31m[Monitor] 进程 {pid} 已经结束或不存在。\033[0m")

    print(f"\033[1;33m[Monitor] 进程 {pid} 已结束，正在检查 GPU {gpu_ids} 显存是否释放 (阈值: {mem_threshold} MiB)...\033[0m")

    while True:
        all_released = True
        for gid in gpu_ids:
            used_mem = get_gpu_memory_usage(gid)
            if used_mem > mem_threshold:
                print(f"\033[1;35m[Monitor] GPU {gid} 仍有较多显存占用: {used_mem} MiB，等待中...\033[0m")
                all_released = False
                break

        if all_released:
            print("\033[1;32m[Monitor] 显存已释放，准备开始训练！\033[0m")
            break

        time.sleep(10)


def load_queue():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return []


def save_queue(queue):
    with open(LOG_FILE, "w") as f:
        json.dump(queue, f, indent=2)


def add_task(args):
    queue = load_queue()
    task = {
        "config": args.config,
        "gpu": args.gpu,
        "work_dir": args.work_dir,
        "wait_pid": args.wait_pid,
        "conda_env": args.conda_env,
        "status": "pending",
        "added_at": datetime.now().isoformat(),
        "pid": None,
    }
    queue.append(task)
    save_queue(queue)
    print(f"\033[1;32m[Queue] 已添加任务到队列:\033[0m")
    print(f"  Config:   {task['config']}")
    print(f"  GPU:      {task['gpu']}")
    print(f"  WorkDir:  {task['work_dir']}")
    print(f"  WaitPID:  {task['wait_pid']}")
    print(f"  CondaEnv: {task['conda_env']}")
    print(f"  队列位置: {len(queue)}")


def show_queue():
    queue = load_queue()
    if not queue:
        print("\033[1;33m[Queue] 队列为空\033[0m")
        return
    print(f"\033[1;36m{'='*90}\033[0m")
    print(f"\033[1;36m{'#':<4} {'Status':<12} {'GPU':<6} {'PID':<8} {'WaitPID':<10} {'Config'}\033[0m")
    print(f"\033[1;36m{'='*90}\033[0m")
    for i, task in enumerate(queue, 1):
        status_color = {
            "pending": "\033[1;33m",
            "running": "\033[1;32m",
            "done": "\033[1;34m",
            "failed": "\033[1;31m",
        }.get(task["status"], "")
        wait_pid = str(task.get("wait_pid") or "")
        print(f"{status_color}{i:<4} {task['status']:<12} {task['gpu']:<6} {str(task.get('pid', '')):<8} {wait_pid:<10} {task['config']}\033[0m")
    print(f"\033[1;36m{'='*90}\033[0m")


def execute_task(task_idx, task, mem_threshold):
    queue = load_queue()
    task = queue[task_idx]

    gpu_ids = task["gpu"].split(",")

    if task.get("wait_pid"):
        wait_for_process_and_memory(task["wait_pid"], gpu_ids, mem_threshold)

    conda_env = task.get("conda_env", "chromo")
    work_dir = task.get("work_dir", "")

    train_cmd_parts = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python", "tools/train.py", task["config"],
    ]
    if work_dir:
        train_cmd_parts.extend(["--work-dir", work_dir])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = task["gpu"]
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    log_path = os.path.join(work_dir, "train_monitor.log") if work_dir else f"train_monitor_{task_idx}.log"
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)

    tag = f"GPU{task['gpu']}"
    print(f"\033[1;32m[{tag}] 启动命令: {' '.join(train_cmd_parts)}\033[0m")
    print(f"\033[1;34m[{tag}] CUDA_VISIBLE_DEVICES={task['gpu']}\033[0m")
    print(f"\033[1;34m[{tag}] 日志输出: {log_path}\033[0m")

    task["status"] = "running"
    task["started_at"] = datetime.now().isoformat()
    save_queue(queue)

    try:
        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            train_cmd_parts,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        task["pid"] = proc.pid
        save_queue(queue)

        print(f"\033[1;32m[{tag}] 训练进程已启动, PID: {proc.pid}\033[0m")
        print(f"\033[1;33m[{tag}] 等待训练完成 (tail -f {log_path} 查看实时日志)...\033[0m")

        ret = proc.wait()
        log_f.close()

        if ret == 0:
            task["status"] = "done"
            print(f"\033[1;32m[{tag}] 任务完成！\033[0m")
        else:
            task["status"] = "failed"
            print(f"\033[1;31m[{tag}] 任务失败 (exit code: {ret})\033[0m")

    except Exception as e:
        task["status"] = "failed"
        print(f"\033[1;31m[{tag}] 任务异常: {e}\033[0m")

    task["finished_at"] = datetime.now().isoformat()
    save_queue(queue)


def run_queue(args):
    queue = load_queue()
    if not queue:
        print("\033[1;33m[Queue] 队列为空，无任务可执行\033[0m")
        return

    pending = [(i, t) for i, t in enumerate(queue) if t["status"] not in ("done", "running")]
    if not pending:
        print("\033[1;33m[Queue] 没有待执行的任务\033[0m")
        show_queue()
        return

    print(f"\033[1;36m[Queue] 开始执行队列，共 {len(pending)} 个待执行任务\033[0m")
    print(f"\033[1;36m[Queue] 按 GPU 分组并行执行，同 GPU 内串行\033[0m")

    gpu_groups = {}
    for i, task in pending:
        gpu_key = task["gpu"]
        if gpu_key not in gpu_groups:
            gpu_groups[gpu_key] = []
        gpu_groups[gpu_key].append((i, task))

    for gpu_key, tasks in gpu_groups.items():
        print(f"\033[1;36m[Queue] GPU {gpu_key}: {len(tasks)} 个任务\033[0m")
        for _, t in tasks:
            print(f"  - {t['config']}")

    threads = []
    for gpu_key, tasks in gpu_groups.items():
        def run_gpu_group(gpu_tasks, gpu_id):
            for idx, task in gpu_tasks:
                tag = f"GPU{gpu_id}"
                print(f"\n\033[1;36m[{tag}] 开始执行: {task['config']}\033[0m")
                execute_task(idx, task, args.mem_threshold)

        t = threading.Thread(target=run_gpu_group, args=(tasks, gpu_key), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"\n\033[1;36m[Queue] 所有任务执行完毕\033[0m")
    show_queue()


def run_single(args):
    if not os.path.exists(args.config):
        print(f"错误: 找不到配置文件 {args.config}")
        sys.exit(1)

    gpu_ids = args.gpu.split(",")

    if args.wait_pid:
        wait_for_process_and_memory(args.wait_pid, gpu_ids, args.mem_threshold)

    conda_env = args.conda_env
    train_cmd = [
        "conda", "run", "--no-capture-output", "-n", conda_env,
        "python", "tools/train.py", args.config,
    ]
    if args.work_dir:
        train_cmd.extend(["--work-dir", args.work_dir])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    log_path = os.path.join(args.work_dir, "train_monitor.log") if args.work_dir else "train_monitor.log"
    os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)

    print(f"\033[1;32m[Single] 启动命令: {' '.join(train_cmd)}\033[0m")
    print(f"\033[1;34m[Single] CUDA_VISIBLE_DEVICES={args.gpu}\033[0m")
    print(f"\033[1;34m[Single] 日志输出: {log_path}\033[0m")

    try:
        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            train_cmd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        print(f"\033[1;32m[Single] 训练进程已启动, PID: {proc.pid}\033[0m")
        print(f"\033[1;33m[Single] 等待训练完成 (tail -f {log_path} 查看实时日志)...\033[0m")

        ret = proc.wait()
        log_f.close()

        if ret == 0:
            print(f"\033[1;32m[Single] 训练完成！\033[0m")
        else:
            print(f"\033[1;31m[Single] 训练失败 (exit code: {ret})\033[0m")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n收到停止信号，正在退出...")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="训练任务监控与队列管理脚本")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    subparsers.add_parser("show", help="查看当前队列")

    add_parser = subparsers.add_parser("add", help="添加任务到队列")
    add_parser.add_argument("config", help="训练配置文件路径")
    add_parser.add_argument("--gpu", type=str, default="0", help="GPU ID (默认: 0)")
    add_parser.add_argument("--work-dir", type=str, default="", help="工作目录")
    add_parser.add_argument("--wait-pid", type=int, default=None, help="等待的进程 PID")
    add_parser.add_argument("--conda-env", type=str, default="chromo", help="Conda 环境名 (默认: chromo)")

    run_parser = subparsers.add_parser("run", help="执行队列中的所有任务 (按GPU并行)")
    run_parser.add_argument("--mem-threshold", type=int, default=1000, help="显存释放阈值 MiB (默认: 1000)")

    single_parser = subparsers.add_parser("single", help="直接执行单个训练任务 (兼容旧模式)")
    single_parser.add_argument("config", help="训练配置文件路径")
    single_parser.add_argument("--gpu", type=str, default="0", help="GPU ID (默认: 0)")
    single_parser.add_argument("--work-dir", type=str, default="", help="工作目录")
    single_parser.add_argument("--wait-pid", type=int, default=None, help="等待的进程 PID")
    single_parser.add_argument("--conda-env", type=str, default="chromo", help="Conda 环境名 (默认: chromo)")
    single_parser.add_argument("--mem-threshold", type=int, default=1000, help="显存释放阈值 MiB (默认: 1000)")

    args = parser.parse_args()

    if args.command == "show":
        show_queue()
    elif args.command == "add":
        add_task(args)
    elif args.command == "run":
        run_queue(args)
    elif args.command == "single":
        run_single(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
