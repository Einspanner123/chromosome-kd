"""ChromoGen Phase 2: 图像+BBox联合训练

enable_bbox_head = True，联合训练图像扩散和bbox扩散。
"""

_base_ = ['./chromogen_base.py']

enable_bbox_head = True
max_epochs = 200
learning_rate = 5e-5  # 更低的学习率用于联合训练
lambda_img = 1.0
lambda_bbox = 0.5
lambda_cls = 0.5
output_dir = 'work_dirs/chromogen_phase2_joint'

# BBox头使用更多proposals
bbox_num_proposals = 200
