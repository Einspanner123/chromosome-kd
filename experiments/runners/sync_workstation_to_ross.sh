#!/bin/bash
# 串行 rsync 同步 workstation 独有 checkpoint 到 ross (本地=ross)
# 优化: 只同步 best_coco_bbox_mAP_*.pth + 配置(.py) + 日志(.log), 排除冗余 epoch_*.pth
# rsync 增量同步: 跳过已存在且完整的文件
set -u

WORKSTATION_USER="linkst"
WORKSTATION_IP="100.99.131.26"
WORKSTATION_PASS="000928"
WORKSTATION_ROOT="/home/linkst/workplace/chromo/chromosome-kd/work_dirs"

# 本地就是 ross, 直接写入本地路径
LOCAL_ROOT="/media/ross/8TB/linkst/chromo/chromosome-kd/work_dirs"

LOG_DIR="/tmp/sync_ws_to_ross"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOG_DIR/rsync_summary_${TS}.log"

echo "=== Workstation -> Ross (本地) rsync 串行同步 ===" | tee "$SUMMARY"
echo "策略: 只同步 best checkpoint + 配置 + 日志 (排除 epoch_*.pth)" | tee -a "$SUMMARY"
echo "开始: $(date)" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# 按大小排序 (best checkpoint ~466M each)
DIRS=(
    "reproduce_0751_stochot_eps5_v2"
    "reproduce_0751_stochot_eps5"
    "nonlinear_trajectory_e42"
    "s1_h3_s4_24obj"
    "r3_vpred_24obj_seed42"
    "s1_h6_s2_24obj"
    "r3_vpred_chr2024_seed42"
    "r3_vpred_24obj_seed123"
    "r3_vpred_24obj_seed789"
    "cross_dataset/d2_0753_stochot_eps5_seed2016452323"
)

TOTAL=${#DIRS[@]}
SUCCESS=0
FAIL=0

for DIR in "${DIRS[@]}"; do
    IDX=$((SUCCESS + FAIL + 1))
    echo "[$(date +%H:%M:%S)] >>> [$IDX/$TOTAL] $DIR" | tee -a "$SUMMARY"

    # 确保本地目标目录存在
    mkdir -p "$LOCAL_ROOT/$DIR"

    START=$(date +%s)

    # rsync: 只同步 best checkpoint + 配置 + 日志, 排除其他大文件
    # --partial: 保留部分传输的文件 (断点续传)
    # -z: 压缩传输
    # -h: 人类可读进度
    sshpass -p "$WORKSTATION_PASS" rsync -avzh --partial \
        --include='*/' \
        --include='best_coco_bbox_mAP_*.pth' \
        --include='*.py' \
        --include='*.log' \
        --include='last_checkpoint' \
        --exclude='*' \
        "$WORKSTATION_USER@$WORKSTATION_IP:$WORKSTATION_ROOT/$DIR/" \
        "$LOCAL_ROOT/$DIR/" 2>&1 | tee -a "$LOG_DIR/${DIR//\//_}_rsync.log" | tail -5 >> "$SUMMARY"

    RC=${PIPESTATUS[0]}
    END=$(date +%s)
    ELAPSED=$((END - START))

    if [ $RC -eq 0 ]; then
        # 验证 best checkpoint
        BEST=$(ls "$LOCAL_ROOT/$DIR"/best_coco_bbox_mAP_*.pth 2>/dev/null | head -1)
        if [ -n "$BEST" ]; then
            SIZE=$(du -h "$BEST" | cut -f1)
            echo "[$(date +%H:%M:%S)] <<< [OK] $DIR (${ELAPSED}s) best=$(basename "$BEST") [$SIZE]" | tee -a "$SUMMARY"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "[$(date +%H:%M:%S)] <<< [WARN] $DIR (${ELAPSED}s) 传输成功但无 best checkpoint" | tee -a "$SUMMARY"
            SUCCESS=$((SUCCESS + 1))
        fi
    else
        echo "[$(date +%H:%M:%S)] <<< [FAIL] $DIR (${ELAPSED}s) rc=$RC" | tee -a "$SUMMARY"
        FAIL=$((FAIL + 1))
    fi
done

echo "" | tee -a "$SUMMARY"
echo "=== 完成 $(date) ===" | tee -a "$SUMMARY"
echo "成功: $SUCCESS / $TOTAL, 失败: $FAIL" | tee -a "$SUMMARY"
echo "汇总日志: $SUMMARY" | tee -a "$SUMMARY"

# 最终验证: 列出所有同步的 best checkpoint
echo "" | tee -a "$SUMMARY"
echo "=== 已同步 best checkpoint 清单 ===" | tee -a "$SUMMARY"
for DIR in "${DIRS[@]}"; do
    BEST=$(ls "$LOCAL_ROOT/$DIR"/best_coco_bbox_mAP_*.pth 2>/dev/null | head -1)
    if [ -n "$BEST" ]; then
        SIZE=$(du -h "$BEST" | cut -f1)
        echo "  ✓ $DIR: $(basename "$BEST") [$SIZE]" | tee -a "$SUMMARY"
    else
        echo "  ✗ $DIR: 无 best checkpoint" | tee -a "$SUMMARY"
    fi
done
