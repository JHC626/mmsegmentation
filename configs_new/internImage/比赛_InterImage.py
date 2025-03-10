# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------





from mmengine.config import read_base
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from torch.optim import AdamW
from mmseg.models.backbones.InterImage import InternImage
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmseg.evaluation import IoUMetric
from mmengine.optim.optimizer import OptimWrapper
from mmseg.engine.optimizers.customLayerDecay import CustomLayerDecayOptimizerConstructor
from mmengine.optim.scheduler.lr_scheduler import LinearLR, PolyLR
from torch.nn.modules.activation import GELU
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.nn.modules.normalization import GroupNorm as GN

with read_base():
    from .._base_.datasets.crop_gaofen import *
    from .._base_.default_runtime import *
    from .._base_.schedules.schedule_80k import *
    from .._base_.models.mask2former_beit import *

num_classes = 9
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
load_from = '/data/JHC/openmmlab/mmsegmentation/checkpoints/InterImage/mask2former_internimage_h-mmseg2-4chan.pth'
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean=[995.26933225455, 1452.7270343669, 1638.4348408118, 3150.9832206793],
    std=[317.36181350835, 406.4103774175, 546.77043273976, 501.33003719076],
    # bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

model.update(
    dict(
        type=EncoderDecoder,
        pretrained=None,
        data_preprocessor=data_preprocessor,
        backbone=dict(
            type=InternImage,
            in_channels=4,
            core_op='DCNv3',
            channels=320,
            depths=[6, 6, 32, 6],
            groups=[10, 20, 40, 80],
            mlp_ratio=4.,
            drop_path_rate=0.5,
            norm_layer='LN',
            layer_scale=None,
            offset_scale=1.0,
            post_norm=False,
            dw_kernel_size=5, # for InternImage-H/G
            res_post_norm=True, # for InternImage-H/G
            level2_post_norm=True, # for InternImage-H/G
            level2_post_norm_block_ids=[5, 11, 17, 23, 29], # for InternImage-H/G
            center_feature_scale=True, # for InternImage-H/G
            with_cp=False,
            out_indices=(0, 1, 2, 3),
            init_cfg=None
        ),
        decode_head=dict(num_classes=num_classes),
        test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341))))

optimizer = dict(
    type=AdamW,
    lr=1e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
)


# optimizer = dict(type=AdamW, lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=optimizer,
    constructor=CustomLayerDecayOptimizerConstructor,
    paramwise_cfg=dict(num_layers=50, layer_decay_rate=0.95,
                       depths=[6, 6, 32, 6], offset_lr_scale=1.0))

param_scheduler = [
    dict(
        type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]


train_cfg.update(type=IterBasedTrainLoop, max_iters=80000, val_interval=4000)
default_hooks.update(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=2000, max_keep_ckpts=10),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))

val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric,
    iou_metrics=['mIoU', 'mFscore'],
    #format_only=True,
    keep_results=True)