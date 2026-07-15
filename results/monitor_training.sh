#!/bin/bash
# 训练监控脚本 - 每5分钟检查一次
while true; do
    echo ""
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="

    echo "[Exp1 A4 seed123 - workstation GPU0]"
    SSHPASS='000928' sshpass -e ssh -o StrictHostKeyChecking=no linkst@workstation \
        "cd /home/linkst/workplace/chromo/chromosome-kd && LOG=work_dirs/multi_seed/a4_dpm_pp_24obj/seed_123/20260715_021127/20260715_021127.log && echo Epoch: \$(grep -oP 'Epoch\(train\)\s+\[\K\d+' \$LOG | tail -1) && echo Recent_mAP: \$(grep -oP 'coco/bbox_mAP: \K[0-9.]+' \$LOG | tail -3 | tr '\n' ' ') && echo Best_mAP: \$(grep -oP 'coco/bbox_mAP: \K[0-9.]+' \$LOG | sort -rn | head -1)" 2>&1

    echo "[Exp4 eps2 seed42 - workstation GPU1]"
    SSHPASS='000928' sshpass -e ssh -o StrictHostKeyChecking=no linkst@workstation \
        "cd /home/linkst/workplace/chromo/chromosome-kd && LOG=work_dirs/ablation_old/stochot_eps2/20260715_084450/20260715_084450.log && echo Epoch: \$(grep -oP 'Epoch\(train\)\s+\[\K\d+' \$LOG | tail -1) && echo Recent_mAP: \$(grep -oP 'coco/bbox_mAP: \K[0-9.]+' \$LOG | tail -3 | tr '\n' ' ') && echo Best_mAP: \$(grep -oP 'coco/bbox_mAP: \K[0-9.]+' \$LOG | sort -rn | head -1)" 2>&1

    echo "[Exp2a eps5 seed42 - ross GPU0 (已早停, best=0.746@ep60)]"
    echo "  -> checkpoint已SCP到本地"

    echo "[Exp2b eps5 seed123 - ross GPU0 (新启动)]"
    SSHPASS='lxt000928' sshpass -e ssh -o StrictHostKeyChecking=no linkst@ross \
        "cd /media/ross/8TB/linkst/chromo/chromosome-kd && LOG=\$(ls -t work_dirs/multi_seed/stochot_eps5_old/seed_123/*/[0-9]*.log 2>/dev/null | head -1) && if [ -n \"\$LOG\" ]; then echo Epoch: \$(grep -oP 'Epoch\(train\)\s+\[\K\d+' \$LOG | tail -1) && echo Recent_mAP: \$(grep -oP 'coco/bbox_mAP: \K[0-9.]+' \$LOG | tail -3 | tr '\n' ' ') && echo Best_mAP: \$(grep -oP 'coco/bbox_mAP: \K[0-9.]+' \$LOG | sort -rn | head -1); else echo 'Log not found yet'; fi" 2>&1

    sleep 300
done
