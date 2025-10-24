
_base_ = ['../_base_/default_runtime.py']
''''''
'''###################################################### dataset settings ######################################################'''

num_classes = 256
data_preprocessor = dict(
    num_classes=num_classes,
    # RGB format normalization parameters
    mean=[138.5155792236328, 133.14308166503906, 126.48957061767578],
    std=[80.29670715332031, 78.70006561279297, 81.52549743652344],
    # convert image from BGR to RGB
    to_rgb=True,
)

data_root = 'data/caltech-256'
bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]
train_size = 224
test_resize_size = 256
test_size = 224
batch_size = 256
num_workers = 9
warmup = 20
max_epochs = 400

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=train_size,
        backend='pillow',
        interpolation='bicubic',
        crop_ratio_range=(0.7, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=7,
        magnitude_std=0.5,
        hparams=dict(pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=test_resize_size,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=test_size),
    dict(type='PackInputs')
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=batch_size // 2,
    num_workers=num_workers,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        data_prefix='val',
        pipeline=val_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=batch_size // 2,
    num_workers=num_workers,
    dataset=dict(
            type='CustomDataset',
            data_root=data_root,
            data_prefix='test',
            pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='Accuracy', topk=(1, ))
test_evaluator = dict(type='Accuracy', topk=(1, ))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=512)




'''###################################################### schedules settings ######################################################'''

optim_wrapper = dict(
    type='SAGMOptimWrapper',
    SAGM_cfg=dict(
        alpha=0.001,
        adaptive=False,
        perturb_eps=1e-12,
        grad_reduce='mean'),
    optimizer=dict(
        type='AdamW',
        lr=0.0005,
        weight_decay=0.05,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,),
    clip_grad=dict(max_norm=5.0),
)
if optim_wrapper['type'] == 'OptimWrapper':
    optim_wrapper.pop('loss_scale', None)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=2e-5,
        by_epoch=True,
        end=warmup,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=warmup,
        end=max_epochs,
        eta_min_ratio=2e-3,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=5)
val_cfg = dict()
test_cfg = dict()
adversarial_test = [
    dict(
        attack_type='white',
        ckpt=None,
        method=dict(
            type='L2PGD',
            init_cfg=dict(
                abs_stepsize=0.2 / 5,
                steps=10),
            call_cfg=dict(epsilon=0.2))),
    dict(
        attack_type='white',
        ckpt=None,
        method=dict(
            type='L2PGD',
            init_cfg=dict(
                abs_stepsize=0.1 / 5,
                steps=10),
            call_cfg=dict(epsilon=0.1))),
    dict(
        attack_type='white',
        ckpt=None,
        method=dict(
            type='L2PGD',
            init_cfg=dict(
                abs_stepsize=0.05 / 5,
                steps=10),
            call_cfg=dict(epsilon=0.05))),
]




'''###################################################### model settings ######################################################'''

sdloss_cfg = dict(
    losses=[],
    after_iter=0,
    warmup_iter=0,
    grad_modify=dict(
        invalid=[0, 0, 0],
        clip_grad_ratio=[0, 1.],
        clip_to_zero=True,
))

init_cfg = [
        dict(type='TruncNormal', layer=['Parameter', 'Linear', 'Conv2d', 'Conv3d'], std=0.02, bias=0.02),
        dict(type='Constant', layer=['LayerNorm', 'GroupNorm'], val=1., bias=0.02),
]
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/deit/deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        arch='s',
        img_size=train_size,
        patch_size=16,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone')),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=num_classes,
        in_channels=384,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='multi_label', use_sigmoid=False),
        init_cfg=init_cfg),
    init_cfg=init_cfg,
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]))
