#!/usr/bin/env bash
# Post-fix verification: 重新启动 Debug Server + 重置 .ndjson + runId=post-fix
# 与 debug_run_adaln9.sh 的区别:
#   1. 清空 .dbg/trae-debug-log-*.ndjson（保证 post-fix 日志干净）
#   2. 设置 DEBUG_RUN_ID=post-fix（探针会用此 runId 区分 pre/post）
#   3. 写到不同 work_dir，避免覆盖
set -e

ROOT="/home/linkst/workplace/chromo/chromosome-kd"
SESSION="chromosome-kd-dit-zero-map"

cd "$ROOT"

# 1) 清空 .ndjson，强制 runId=post-fix
mkdir -p .dbg
: > ".dbg/trae-debug-log-${SESSION}.ndjson"
export DEBUG_RUN_ID="post-fix"

# 2) 启动 Debug Server (空闲 30 分钟自动退出)
python3 /home/linkst/.trae-cn/builtin_skills/TRAE-debugger/tools/debug-server/python/debug-server.py \
  --session "$SESSION" --outdir .dbg --clean --idle 1800 &
SERVER_PID=$!
echo "[+] Debug Server PID=$SERVER_PID (runId=post-fix)"
sleep 2

# 3) 健康检查
if ! curl -s --max-time 3 http://127.0.0.1:7777/health > /dev/null; then
  echo "[!] Debug Server not healthy, aborting"
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi
echo "[+] Server healthy: $(curl -s http://127.0.0.1:7777/health)"

# 4) 启动 adaln9 post-fix 训练 (1 epoch, 写到独立 dir)
echo "[+] Launching adaln9 post-fix training (1 epoch)"
python tools/train.py projects/LDMDet/configs/ldmdet_dit.py \
  --work-dir work_dirs/ldmdet_dit_r50_shifted3_adaln9_postfix \
  2>&1 | tee ".dbg/train_${SESSION}_postfix.log"

# 5) 训练结束
echo "[+] Done. To compare pre vs post, fetch both runs:"
echo "    curl -s 'http://127.0.0.1:7777/logs?hypothesisId=A&runId=pre-fix' | python3 -m json.tool | less"
echo "    curl -s 'http://127.0.0.1:7777/logs?hypothesisId=A&runId=post-fix' | python3 -m json.tool | less"
echo "[+] Server PID=$SERVER_PID (kill when done)"
