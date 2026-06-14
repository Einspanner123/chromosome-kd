"""ChromoGen Phase 1: 仅图像生成

enable_bbox_head = False，只训练图像扩散模型。
"""

_base_ = ['./chromogen_base.py']

enable_bbox_head = False
max_epochs = 200
learning_rate = 5e-5
output_dir = 'work_dirs/chromogen_phase1_v2'
