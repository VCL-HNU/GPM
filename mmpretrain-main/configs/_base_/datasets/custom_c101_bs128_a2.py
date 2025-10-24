# dataset settings
data_preprocessor = dict(
    num_classes=101,
    # RGB format normalization parameters
    mean=[138.5155792236328, 133.14308166503906, 126.48957061767578],
    std=[80.29670715332031, 78.70006561279297, 81.52549743652344],
    # convert image from BGR to RGB
    to_rgb=True,
)

data_root = 'data/caltech-101'
bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]
train_size = 224
test_size = 224
batch_size = 128
num_workers = 1
repeat_sample = 2

if test_size < 384:
    test_crop_ratio = 0.95
else:
    test_crop_ratio = 1.
test_resize_size = int(test_size / test_crop_ratio)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=train_size,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=7,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(type='PackInputs'),
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
    dict(type='PackInputs'),
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type='RepeatDataset',
        dataset=dict(
            type='CustomDataset',
            data_root=data_root,
            data_prefix='train',
            pipeline=train_pipeline),
        times=repeat_sample),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
            type='CustomDataset',
            data_root=data_root,
            data_prefix='val',
            pipeline=val_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
            type='CustomDataset',
            data_root=data_root,
            data_prefix='test',
            pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_evaluator = dict(type='Accuracy', topk=(1, 5))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048 * repeat_sample)
