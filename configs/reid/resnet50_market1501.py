# dataset settings
dataset_type = 'Market1501'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

policies = [
    [
        dict(type='Posterize', bits=4, prob=0.1),
        dict(type='Rotate', angle=10., prob=0.5)
    ],
    [
        dict(type='Solarize', thr=256 / 9 * 4, prob=0.6),
        dict(type='AutoContrast', prob=0.5)
    ]
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, 128)),
	dict(type='AutoAugment', policies=policies),
    dict(type='RandomCrop', size=(256,128), padding=(10,10)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, 128)),
    #dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    #triplet_sampler=True,
    train=dict(
        type=dataset_type,
        data_prefix='data/Market-1501-v15.09.15',
        pipeline=train_pipeline,
        triplet_sampler=True),
    val=dict(
        type=dataset_type,
        data_prefix='data/Market-1501-v15.09.15',
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix='data/Market-1501-v15.09.15',
        test_mode=True,
        pipeline=test_pipeline),
    )

# model settings
num_classes = 1502
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        strides=(1,2,2,1),
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='BotHead',
        in_channels=2048,
        num_classes=num_classes,
        #loss_ce=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_ce=dict(type='LabelSmoothLoss', label_smooth_val=0.1, num_classes=num_classes, loss_weight=1.0),
        loss_tri=dict(type='TripletLoss', loss_weight=2.0, margin=0.3, norm_feat=False),
        #loss_tri=None,
    ))

#optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
#optimizer = dict(type='Adam', lr=0.0003, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='SGD', lr=0.065, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='step', step=[150, 225, 300])
runner = dict(type='EpochBasedRunner', max_epochs=350)

#optimizer = dict(type='Adam', lr=0.00035)
#lr_config = dict(policy='step', step=[40, 70])
#runner = dict(type='EpochBasedRunner', max_epochs=120)

optimizer_config = dict(grad_clip=None)
# checkpoint saving
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric=['rank1', 'map'])
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
#load_from = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
resume_from = None
workflow = [('train', 1)]
