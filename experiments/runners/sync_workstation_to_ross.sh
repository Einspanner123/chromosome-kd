#!/bin/bash
# 串行拷贝 workstation 独有 checkpoint 到 ross (10 个实验, ~17.2GB)
# 网络拥堵, 串行执行避免带宽争抢
# 数据流: workstation -> 本地管道 -> ross (不占本地存储)
set -u

WORKSTATION_USER="linkst"
WORKSTATION_IP="100.99.131.26"
WORKSTATION_PASS="000928"
WORKSTATION_ROOT="/home/linkst/workplace/chromo/chromosome-kd/work_dirs"

ROSS_USER="linkst"
ROSS_IP="100.122.196.41"
ROSS_PASS="lxt000928"
ROSS_ROOT="/media/ross/8TB/linkst/chromo/chromosome-kd/work_dirs"

LOG_DIR="/tmp/sync_ws_to_ross"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOG_DIR/sync_summary_${TS}.log"

echo "=== Workstation -> Ross 串行 checkpoint 同步 ===" | tee "$SUMMARY"
echo "开始: $(date)" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# 按大小排序 (小 -> 大), 优先完成小目录
DIRS=(
    "reproduce_0751_stochot_eps5_v2|505M"
    "reproduce_0751_stochot_eps5|510M"
    "nonlinear_trajectory_e42|969M"
    "s1_h3_s4_24obj|1.3G"
    "r3_vpred_24obj_seed42|2.0G"
    "s1_h6_s2_24obj|2.1G"
    "r3_vpred_chr2024_seed42|2.5G"
    "r3_vpred_24obj_seed123|2.6G"
    "r3_vpred_24obj_seed789|2.6G"
    "cross_dataset/d2_0753_stochot_eps5_seed2016452323|2.6G"
)

TOTAL=${#DIRS[@]}
SUCCESS=0
FAIL=0

for entry in "${DIRS[@]}"; do
    DIR="${entry%%|*}"
    SIZE="${entry##*|}"
    
    echo "[$(date +%H:%M:%S)] >>> [$((SUCCESS+FAIL+1))/$TOTAL] $DIR ($SIZE)" | tee -a "$SUMMARY"
    
    # 确保 ross 目标目录存在 (处理嵌套路径)
    sshpass -p "$ROSS_PASS" ssh -o StrictHostKeyChecking=no "$ROSS_USER@$ROSS_IP" \
        "mkdir -p '$ROSS_ROOT/$(dirname "$DIR")'" 2>>"$LOG_DIR/${DIR//\//_}_err.log"
    
    # tar 管道: workstation -> 本地 -> ross
    START=$(date +%s)
    
    sshpass -p "$WORKSTATION_PASS" ssh -o StrictHostKeyChecking=no "$WORKSTATION_USER@$WORKSTATION_IP" \
        "cd '$WORKSTATION_ROOT' && tar cf - '$DIR'" 2>>"$LOG_DIR/${DIR//\//_}_err.log" | \
    sshpass -p "$ROSS_PASS" ssh -o StrictHostKeyChecking=no "$ROSS_USER@$ROSS_IP" \
        "cd '$ROSS_ROOT' && tar xf -" 2>>"$LOG_DIR/${DIR//\//_}_err.log"
    
    RC=$?
    END=$(date +%s)
    ELAPSED=$((END - START))
    
    if [ $RC -eq 0 ]; then
        # 验证: 检查 ross 上 best checkpoint 是否存在
        VERIFY=$(sshpass -p "$ROSS_PASS" ssh -o StrictHostKeyChecking=no "$ROSS_USER@$ROSS_IP" \
            "ls '$ROSS_ROOT/$DIR'/best_coco_bbox_mAP_*.pth 2>/dev/null | head -1")
        
        if [ -n "$VERIFY" ]; then
            echo "[$(date +%H:%M:%S)] <<< [OK] $DIR (${ELAPSED}s) checkpoint: $(basename "$VERIFY")" | tee -a "$SUMMARY"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "[$(date +%H:%M:%S)] <<< [WARN] $DIR (${ELAPSED}s) 传输成功但未找到 best checkpoint" | tee -a "$SUMMARY"
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
