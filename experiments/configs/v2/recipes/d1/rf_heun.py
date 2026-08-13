_base_ = [
    '../../_base_/datasets/d1_inhouse1700.py',
    '../../_base_/models/rf_heun_r50.py',
    '../../_base_/schedules/detector_adamw_150e.py',
    '../../_base_/runtime/default.py',
]
experiment = dict(
    config_id='v2.d1.rf_heun_r50.train', dataset_id='D1_INHOUSE1700_V1',
    method_id='rf_heun_r50', role='generation_reference',
    replication_unit='independent_training_seed', selection_split='val',
    selection_metric='coco/bbox_mAP', test_tuned=False,
    tracker_project='KaryoFlow-Self1700',
)
