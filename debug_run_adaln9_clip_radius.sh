#!/usr/bin/env bash
# Step 11: 第三轮修复 (clip_norm=35 + radius=1.5) — 重启训练
# 用法: 在本地 shell 跑 `bash debug_run_adaln9_clip_radius.sh`
# 前置: 必须在有 mmcv 的 conda env (mm 或 chromo) 里
set -e

ROOT="/home/linkst/workplace/chromo/chromosome-kd"
SESSION="chromosome-kd-dit-zero-map"

cd "$ROOT"

# 1) 杀掉旧的 adaln9_dbg 训练 (PID 2648211)
echo "[+] 杀掉旧 adaln9_dbg 训练..."
pkill -f "ldmdet_dit_r50_shifted3_adaln9_dbg" || true
sleep 2
if pgrep -f "ldmdet_dit_r50_shifted3_adaln9_dbg" > /dev/null; then
  pkill -9 -f "ldmdet_dit_r50_shifted3_adaln9_dbg" || true
  sleep 1
fi

# 2) 杀掉可能残留的 postfix Debug Server
pkill -f "debug-server.py.*chromosome-kd-dit-zero-map" || true
sleep 1

# 3) 清空 .ndjson + 设 runId
mkdir -p .dbg
: > ".dbg/trae-debug-log-${SESSION}.ndjson"
export DEBUG_RUN_ID="clip-radius"

# 4) 启动 Debug Server
python3 /home/linkst/.trae-cn/builtin_skills/TRAE-debugger/tools/debug-server/python/debug-server.py \
  --session "$SESSION" --outdir .dbg --clean --idle 1800 &
SERVER_PID=$!
echo "[+] Debug Server PID=$SERVER_PID (runId=clip-radius)"
sleep 2

if ! curl -s --max-time 3 http://127.0.0.1:7777/health > /dev/null; then
  echo "[!] Debug Server not healthy"
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi
echo "[+] Server healthy"

# 5) 启动训练 (1 epoch, 写到独立 dir)
echo "[+] Launching adaln9 clip-radius training (1 epoch)"
python tools/train.py projects/LDMDet/configs/ldmdet_dit.py \
  --work-dir work_dirs/ldmdet_dit_r50_shifted3_adaln9_clipradius \
  2>&1 | tee ".dbg/train_${SESSION}_clipradius.log"

echo "[+] Done. To compare: pre vs fix-A vs fix-clip-radius"
echo "    curl -s 'http://127.0.0.1:7777/logs?hypothesisId=A&runId=clip-radius' | python3 -m json.tool | less"
