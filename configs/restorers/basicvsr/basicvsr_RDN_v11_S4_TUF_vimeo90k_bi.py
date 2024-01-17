exp_name = 'basicvsr_RDN_v11_S4_vimeo90k_bi'

# model settings
model = dict(
    type='BasicVSR_RDN',
    generator=dict(
        type='BasicVSRNet_RDN_v11',
        mid_channels=64,
        num_blocks=30,
        spynet_pretrained='pretrained/spynet_20210409-c6c1bd09.pth',
        SMSR_pretrained='pretrained/SMSR_1000.pt', 
        with_tsa= False
        ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0, convert_to='y')

# dataset settings
train_dataset_type = 'SRVimeo90KMultipleGTDataset'
val_dataset_type = 'SRFolderMultipleGTDataset'
test_dataset_type = 'SRVimeo90KDataset'

train_pipeline = [
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='MirrorSequence', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

val_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

test_pipeline = [
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='MirrorSequence', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

data = dict(
    workers_per_gpu=16,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),  # 2 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='D:/AI_master/SR_datasets/vimeo_90k/vimeo_septuplet/sequences_BIx4',
            gt_folder='D:/AI_master/SR_datasets/vimeo_90k/vimeo_septuplet/sequences',
            ann_file='data/vimeo90k/meta_info_Vimeo90K_train_GT.txt',
            pipeline=train_pipeline,
            scale=4,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='D:/AI_master/SR_datasets/Vid4/BIx4',
        gt_folder='D:/AI_master/SR_datasets/Vid4/GT',
        ann_file='data/Vid4/meta_info_Vid4.txt',
        pipeline=val_pipeline,
        scale=4,
        test_mode=True),
    # test
    test=dict(
        type=test_dataset_type,
        lq_folder='D:/AI_master/SR_datasets/vimeo_90k/vimeo_super_resolution_test/low_resolution',
        gt_folder='D:/AI_master/SR_datasets/vimeo_90k/vimeo_super_resolution_test/target',
        ann_file='data/vimeo_90k/meta_info_Vimeo90K_test_GT_02.txt',
        pipeline=test_pipeline,
        scale=4,
        num_input_frames=7,
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=2e-4,
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)})))

# learning policy
total_iters = 300000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[300000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
# evaluation = dict(interval=200, save_image=False, gpu_collect=True)
evaluation = dict(interval=5000, save_image=False)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
