#!/usr/bin/env bash
# 系统资源监控脚本 - 记录训练过程中的各项指标
# 用法: bash monitor.sh [持续时间秒] [采样间隔秒]
# 示例: bash monitor.sh 3600 5  (监控1小时，每5秒采样一次)

DURATION=${1:-3600}       # 默认监控1小时
INTERVAL=${2:-5}          # 默认5秒采样一次
OUTPUT_DIR="$(dirname "$0")/monitor_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/monitor_${TIMESTAMP}.csv"
SUMMARY_FILE="${OUTPUT_DIR}/summary_${TIMESTAMP}.txt"

mkdir -p "$OUTPUT_DIR"

# CSV 表头
echo "timestamp,cpu_usage_pct,mem_used_gb,mem_available_gb,mem_cached_gb,gpu_util_pct,gpu_mem_used_mib,gpu_mem_total_mib,gpu_temp_c,gpu_power_w,gpu_power_cap_w,disk_root_used_pct,disk_root_avail_gb,io_read_mb_s,io_write_mb_s,process_rss_gb,process_cpu_pct" > "$OUTPUT_FILE"

echo "============================================"
echo "  训练资源监控脚本"
echo "============================================"
echo "  持续时间: ${DURATION}s ($(( DURATION / 60 ))min)"
echo "  采样间隔: ${INTERVAL}s"
echo "  输出文件: ${OUTPUT_FILE}"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo ""

# 获取训练进程 PID
TRAIN_PID=$(ps aux | grep "python.*train.py" | grep -v grep | head -1 | awk '{print $2}')
if [ -z "$TRAIN_PID" ]; then
    echo "[警告] 未找到训练进程，进程相关指标将为空"
fi
echo "训练进程 PID: ${TRAIN_PID:-未找到}"

# 记录上一次的磁盘IO计数
PREV_IO_READ=0
PREV_IO_WRITE=0
PREV_IO_TIME=$(date +%s%N)

SAMPLES=0
START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION))

while [ "$(date +%s)" -lt "$END_TIME" ]; do
    CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    TIMESTAMP_EPOCH=$(date +%s)

    # --- CPU 使用率 (所有核心平均) ---
    CPU_IDLE=$(top -bn1 | grep "Cpu(s)" | awk '{print $8}')
    CPU_USAGE=$(echo "100 - $CPU_IDLE" | bc 2>/dev/null || echo "0")

    # --- 内存 ---
    MEM_INFO=$(free -m | grep "内存：" || free -m | grep "Mem:")
    MEM_USED_GB=$(echo "$MEM_INFO" | awk '{printf "%.1f", $3/1024}')
    MEM_AVAIL_GB=$(echo "$MEM_INFO" | awk '{printf "%.1f", $7/1024}')
    MEM_CACHED_GB=$(echo "$MEM_INFO" | awk '{printf "%.1f", $6/1024}')

    # --- GPU ---
    GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null)
    if [ -n "$GPU_INFO" ]; then
        GPU_UTIL=$(echo "$GPU_INFO" | awk -F', ' '{print $1}')
        GPU_MEM_USED=$(echo "$GPU_INFO" | awk -F', ' '{print $2}')
        GPU_MEM_TOTAL=$(echo "$GPU_INFO" | awk -F', ' '{print $3}')
        GPU_TEMP=$(echo "$GPU_INFO" | awk -F', ' '{print $4}')
        GPU_POWER=$(echo "$GPU_INFO" | awk -F', ' '{print $5}' | cut -d'.' -f1)
        GPU_POWER_CAP=$(echo "$GPU_INFO" | awk -F', ' '{print $6}' | cut -d'.' -f1)
    else
        GPU_UTIL=0; GPU_MEM_USED=0; GPU_MEM_TOTAL=0; GPU_TEMP=0; GPU_POWER=0; GPU_POWER_CAP=0
    fi

    # --- 磁盘 ---
    DISK_INFO=$(df -m / | tail -1)
    DISK_USED_PCT=$(echo "$DISK_INFO" | awk '{print $5}' | tr -d '%')
    DISK_AVAIL_GB=$(echo "$DISK_INFO" | awk '{printf "%.1f", $4/1024}')

    # --- 磁盘 IO (基于 /proc/diskstats for nvme0n1) ---
    DISK_STATS=$(cat /proc/diskstats | grep "nvme0n1 " | head -1)
    if [ -n "$DISK_STATS" ]; then
        IO_READ_SECTORS=$(echo "$DISK_STATS" | awk '{print $6}')
        IO_WRITE_SECTORS=$(echo "$DISK_STATS" | awk '{print $10}')
        CURRENT_IO_TIME=$(date +%s%N)
        TIME_DIFF_SEC=$(echo "scale=3; ($CURRENT_IO_TIME - $PREV_IO_TIME) / 1000000000" | bc 2>/dev/null || echo "1")
        if [ "$TIME_DIFF_SEC" = "0" ] || [ -z "$TIME_DIFF_SEC" ]; then
            TIME_DIFF_SEC=1
        fi
        # 扇区大小 512 bytes -> MB
        IO_READ_MB_S=$(echo "scale=1; ($IO_READ_SECTORS - $PREV_IO_READ) * 512 / 1048576 / $TIME_DIFF_SEC" | bc 2>/dev/null || echo "0")
        IO_WRITE_MB_S=$(echo "scale=1; ($IO_WRITE_SECTORS - $PREV_IO_WRITE) * 512 / 1048576 / $TIME_DIFF_SEC" | bc 2>/dev/null || echo "0")
        PREV_IO_READ=$IO_READ_SECTORS
        PREV_IO_WRITE=$IO_WRITE_SECTORS
        PREV_IO_TIME=$CURRENT_IO_TIME
    else
        IO_READ_MB_S=0; IO_WRITE_MB_S=0
    fi

    # --- 训练进程 ---
    if [ -n "$TRAIN_PID" ] && [ -d "/proc/$TRAIN_PID" ]; then
        PROC_INFO=$(ps -p "$TRAIN_PID" -o rss=,pcpu= 2>/dev/null)
        PROC_RSS_GB=$(echo "$PROC_INFO" | awk '{printf "%.1f", $1/1048576}')
        PROC_CPU_PCT=$(echo "$PROC_INFO" | awk '{printf "%.1f", $2}')
    else
        PROC_RSS_GB=0; PROC_CPU_PCT=0
    fi

    # 写入 CSV
    echo "${CURRENT_TIME},${CPU_USAGE},${MEM_USED_GB},${MEM_AVAIL_GB},${MEM_CACHED_GB},${GPU_UTIL},${GPU_MEM_USED},${GPU_MEM_TOTAL},${GPU_TEMP},${GPU_POWER},${GPU_POWER_CAP},${DISK_USED_PCT},${DISK_AVAIL_GB},${IO_READ_MB_S},${IO_WRITE_MB_S},${PROC_RSS_GB},${PROC_CPU_PCT}" >> "$OUTPUT_FILE"

    SAMPLES=$((SAMPLES + 1))

    # 每30秒打印一次实时状态
    if [ $((SAMPLES % 6)) -eq 0 ] || [ "$SAMPLES" -eq 1 ]; then
        echo "[${CURRENT_TIME}] CPU:${CPU_USAGE}% | MEM:${MEM_USED_GB}/${MEM_AVAIL_GB}GB(avail) | GPU:${GPU_UTIL}% ${GPU_MEM_USED}/${GPU_MEM_TOTAL}MiB ${GPU_TEMP}°C ${GPU_POWER}W | Disk:${DISK_USED_PCT}% | IO R:${IO_READ_MB_S} W:${IO_WRITE_MB_S} MB/s | Proc:${PROC_RSS_GB}GB ${PROC_CPU_PCT}%"
    fi

    sleep "$INTERVAL"
done

# --- 生成汇总报告 ---
echo ""
echo "============================================"
echo "  监控结束，生成汇总报告"
echo "============================================"

# 使用 awk 计算统计值
awk -F',' '
NR > 1 {
    n++
    cpu_sum += $2; cpu_arr[n] = $2
    mem_sum += $3; mem_arr[n] = $3
    gpu_util_sum += $6; gpu_util_arr[n] = $6
    gpu_mem_sum += $7; gpu_mem_arr[n] = $7
    gpu_temp_sum += $9; gpu_temp_arr[n] = $9
    gpu_power_sum += $10; gpu_power_arr[n] = $10
    io_r_sum += $14; io_w_sum += $15
    proc_rss_sum += $16; proc_cpu_sum += $17
}
END {
    # 排序函数 (简单冒泡)
    for (i = 1; i <= n; i++) {
        for (j = i+1; j <= n; j++) {
            if (cpu_arr[i] > cpu_arr[j]) { t=cpu_arr[i]; cpu_arr[i]=cpu_arr[j]; cpu_arr[j]=t }
            if (gpu_util_arr[i] > gpu_util_arr[j]) { t=gpu_util_arr[i]; gpu_util_arr[i]=gpu_util_arr[j]; gpu_util_arr[j]=t }
            if (gpu_mem_arr[i] > gpu_mem_arr[j]) { t=gpu_mem_arr[i]; gpu_mem_arr[i]=gpu_mem_arr[j]; gpu_mem_arr[j]=t }
            if (gpu_temp_arr[i] > gpu_temp_arr[j]) { t=gpu_temp_arr[i]; gpu_temp_arr[i]=gpu_temp_arr[j]; gpu_temp_arr[j]=t }
            if (gpu_power_arr[i] > gpu_power_arr[j]) { t=gpu_power_arr[i]; gpu_power_arr[i]=gpu_power_arr[j]; gpu_power_arr[j]=t }
        }
    }

    printf "总采样数: %d\n\n", n

    printf "%-20s %8s %8s %8s %8s\n", "指标", "平均", "最小", "最大", "P95"
    printf "%-20s %8s %8s %8s %8s\n", "--------------------", "--------", "--------", "--------", "--------"

    p95_idx = int(n * 0.95) + 1
    if (p95_idx > n) p95_idx = n

    printf "%-20s %7.1f%% %7.1f%% %7.1f%% %7.1f%%\n", "CPU使用率", cpu_sum/n, cpu_arr[1], cpu_arr[n], cpu_arr[p95_idx]
    printf "%-20s %7.1fG %7.1fG %7.1fG %7.1fG\n", "内存使用", mem_sum/n, mem_arr[1], mem_arr[n], mem_arr[p95_idx]
    printf "%-20s %7.1f%% %7.1f%% %7.1f%% %7.1f%%\n", "GPU利用率", gpu_util_sum/n, gpu_util_arr[1], gpu_util_arr[n], gpu_util_arr[p95_idx]
    printf "%-20s %7.0fM %7.0fM %7.0fM %7.0fM\n", "GPU显存使用", gpu_mem_sum/n, gpu_mem_arr[1], gpu_mem_arr[n], gpu_mem_arr[p95_idx]
    printf "%-20s %7.1f°C %7.1f°C %7.1f°C %7.1f°C\n", "GPU温度", gpu_temp_sum/n, gpu_temp_arr[1], gpu_temp_arr[n], gpu_temp_arr[p95_idx]
    printf "%-20s %7.0fW %7.0fW %7.0fW %7.0fW\n", "GPU功耗", gpu_power_sum/n, gpu_power_arr[1], gpu_power_arr[n], gpu_power_arr[p95_idx]
    printf "%-20s %7.1f   %7.1f   %7.1f   %7.1f\n", "磁盘读(MB/s)", io_r_sum/n, 0, 0, 0
    printf "%-20s %7.1f   %7.1f   %7.1f   %7.1f\n", "磁盘写(MB/s)", io_w_sum/n, 0, 0, 0
    printf "%-20s %7.1fG %7.1fG %7.1fG %7.1fG\n", "进程RSS", proc_rss_sum/n, 0, 0, 0
    printf "%-20s %7.1f%% %7.1f%% %7.1f%% %7.1f%%\n", "进程CPU", proc_cpu_sum/n, 0, 0, 0
}
' "$OUTPUT_FILE" > "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

echo ""
echo "详细数据: ${OUTPUT_FILE}"
echo "汇总报告: ${SUMMARY_FILE}"
echo ""
echo "提示: 可用以下命令快速绘图:"
echo "  python3 -c \""
echo "import pandas as pd, matplotlib.pyplot as plt"
echo "df = pd.read_csv('${OUTPUT_FILE}')"
echo "fig, axes = plt.subplots(3, 2, figsize=(14, 10))"
echo "df.plot(y='gpu_util_pct', ax=axes[0,0], title='GPU Util'); df.plot(y='gpu_mem_used_mib', ax=axes[0,1], title='GPU Mem')"
echo "df.plot(y='cpu_usage_pct', ax=axes[1,0], title='CPU'); df.plot(y='mem_used_gb', ax=axes[1,1], title='Memory')"
echo "df.plot(y='gpu_temp_c', ax=axes[2,0], title='GPU Temp'); df.plot(y='gpu_power_w', ax=axes[2,1], title='GPU Power')"
echo "plt.tight_layout(); plt.savefig('${OUTPUT_DIR}/plot_${TIMESTAMP}.png', dpi=150)"
echo "print('Plot saved!')"
echo "\""
