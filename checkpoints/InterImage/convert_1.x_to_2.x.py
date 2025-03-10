# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_vit(ckpt):

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():

        if k.startswith('backbone'): # norm.weight-->ln1.weight
            new_k = k
        # patch_embed.proj.weight --> patch_embed.projection.weight
        elif k.startswith('decode_head'):
            if 'pixel_decoder' in k:
                if 'attentions.0.' in k:
                    new_k = k.replace('attentions.0.', 'self_attn.')
                elif 'ffns.0.' in k:
                    new_k = k.replace('ffns.0.', 'ffn.')
                else:
                    new_k = k
            elif 'transformer_decoder' in k:
                if 'attentions.0.' in k:
                    new_k = k.replace('attentions.0.', 'self_attn.')
                elif 'attentions.1.' in k:
                    new_k = k.replace('attentions.1.', 'cross_attn.')
                elif 'ffns.0.' in k:
                    new_k = k.replace('ffns.0.', 'ffn.')
                else:
                    new_k = k
            else:
                new_k = k
        new_ckpt[new_k] = v

    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in timm pretrained vit models to '
        'MMSegmentation style.')
    parser.add_argument('--src', help='src model path or url',
                        default='/data/JHC/openmmlab/mmsegmentation/checkpoints/InterImage/mask2former_internimage_h_512_40k_cocostuff164k_to_10k.pth')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('--dst', help='save path',
                        default='/data/JHC/openmmlab/mmsegmentation/checkpoints/InterImage/mask2former_internimage_h-mmseg2.pth')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_vit(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
