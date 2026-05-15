_base_ = [
    '../strongsort/yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py'
]

# fp16 settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
