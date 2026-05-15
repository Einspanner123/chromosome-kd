_base_ = ['./ldmdet_rf_heun_shifted_bs2.py']

model = dict(
    bbox_head=dict(
        box_renewal=True,
        roi_share=True,
        roi_share_iou_thr=0.95,
        torch_compile=True,
    ))
