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

num_classes = 18
crop_size = (512, 512)
norm_cfg = dict(type='SyncBN', requires_grad=True)
load_from = '/data/JHC/openmmlab/mmsegmentation/work_dirs/比赛_InterImage_gaofen/iter_24000.pth' #如果resume的话，load_from这里就是resume继续训练的断点
data_preprocessor = dict(
    type=SegDataPreProcessor,
    mean =[454.1608733420, 320.6480230485 , 238.9676917808 , 301.4478970428],
    std =[55.4731833972, 51.5171917858, 62.3875607521, 82.6082214602],
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
        test_cfg=dict(mode='slide', crop_size=crop_size, stride=(314, 314))))

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
    # format_only=True,
    keep_results=True)