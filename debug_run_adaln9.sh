#!/usr/bin/env bash
# Debug Server + adaln9 training 启动脚本
# 用法: bash debug_run_adaln9.sh
set -e

ROOT="/home/linkst/workplace/chromo/chromosome-kd"
SESSION="chromosome-kd-dit-zero-map"

cd "$ROOT"

# 1) 清理旧日志，强制 runId=pre-fix
mkdir -p .dbg
: > ".dbg/trae-debug-log-${SESSION}.ndjson"

# 2) 启动 Debug Server (前台后台都行；空闲 30 分钟自动退出)
python3 /home/linkst/.trae-cn/builtin_skills/TRAE-debugger/tools/debug-server/python/debug-server.py \
  --session "$SESSION" --outdir .dbg --clean --idle 1800 &
SERVER_PID=$!
echo "[+] Debug Server PID=$SERVER_PID"
sleep 2

# 3) 健康检查
if ! curl -s --max-time 3 http://127.0.0.1:7777/health > /dev/null; then
  echo "[!] Debug Server not healthy on 127.0.0.1:7777, aborting"
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi
echo "[+] Debug Server healthy: $(curl -s http://127.0.0.1:7777/health)"

# 4) 启动 adaln9 训练 (1 个 epoch)
echo "[+] Launching adaln9 training (1 epoch)"
python tools/train.py projects/LDMDet/configs/ldmdet_dit.py \
  --work-dir work_dirs/ldmdet_dit_r50_shifted3_adaln9_dbg \
  2>&1 | tee ".dbg/train_${SESSION}.log"

# 5) 训练结束后保留 server，方便你拷 log
echo "[+] Training done. To view debug logs:"
echo "    curl -s http://127.0.0.1:7777/logs?hypothesisId=A | python3 -m json.tool | less"
echo "    curl -s 'http://127.0.0.1:7777/logs?hypothesisId=B' | python3 -m json.tool"
echo "[+] Or just paste: .dbg/trae-debug-log-${SESSION}.ndjson"
echo "[+] To stop the server: kill $SERVER_PID"
