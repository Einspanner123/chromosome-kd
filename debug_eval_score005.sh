#!/usr/bin/env bash
# Step 12: 跑 eval 用 score_thr=0.05 看 mAP
# 用 clipradius 已经训好的 epoch_2.pth 复测 (不要重训)
set -e

ROOT="/home/linkst/workplace/chromo/chromosome-kd"
cd "$ROOT"

CKPT_DIR="work_dirs/ldmdet_dit_r50_shifted3_adaln9_clipradius"
# 选最新 epoch_*.pth (跳过 best_* 因为 5 epoch 全是 mAP=0)
CKPT=$(ls -1t "$CKPT_DIR"/epoch_*.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  CKPT="$CKPT_DIR/best_coco_bbox_mAP_epoch_1.pth"
fi
OUT_DIR="${CKPT_DIR}_eval_score005"

if [ ! -f "$CKPT" ]; then
  echo "[!] $CKPT not found"
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "[+] Evaluating $CKPT with score_thr=0.05 (on val set)"
# 显式 override: 用 val_dataloader/val_evaluator, 避免 base config 的
# test_dataloader.ann_file 路径翻倍 bug + 与训练时的 val 对齐
python tools/test.py projects/LDMDet/configs/ldmdet_dit.py "$CKPT" \
  --work-dir "$OUT_DIR" \
  --out "$OUT_DIR/results.pkl" \
  --cfg-options \
    test_dataloader=val_dataloader \
    test_evaluator=val_evaluator \
  2>&1 | tee "$OUT_DIR/eval.log"

echo ""
echo "[+] === mAP SUMMARY ==="
grep -oE "coco/bbox_mAP(_[a-z]+)?: 0\.[0-9]+" "$OUT_DIR/eval.log" | head -10
