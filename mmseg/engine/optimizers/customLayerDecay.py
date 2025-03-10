# Copyright (c) OpenMMLab. All rights reserved.
import json
import warnings

from mmengine.dist import get_dist_info
from mmengine.logging import print_log
from mmengine.optim import DefaultOptimWrapperConstructor

from mmseg.registry import OPTIM_WRAPPER_CONSTRUCTORS


def get_num_layer_for_swin(var_name, num_max_layer, depths):
    if var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('decode_head.mask_embed'):
        return 0
    elif var_name.startswith('decode_head.cls_embed'):
        return 0
    elif var_name.startswith('decode_head.level_embed'):
        return 0
    elif var_name.startswith('decode_head.query_embed'):
        return 0
    elif var_name.startswith('decode_head.query_feat'):
        return 0
    if var_name.startswith('backbone.cb_modules.0.patch_embed'):
        return 0
    elif 'level_embeds' in var_name:
        return 0
    elif var_name.startswith('backbone.layers') or var_name.startswith(
            'backbone.levels'):
        if var_name.split('.')[3] not in ['downsample', 'norm']:
            stage_id = int(var_name.split('.')[2])
            layer_id = int(var_name.split('.')[4])
            # layers for Swin-Large: [2, 2, 18, 2]
            if stage_id == 0:
                return layer_id + 1
            elif stage_id == 1:
                return layer_id + 1 + depths[0]
            elif stage_id == 2:
                return layer_id + 1 + depths[0] + depths[1]
            else:
                return layer_id + 1 + depths[0] + depths[1] + depths[2]
        else:
            stage_id = int(var_name.split('.')[2])
            if stage_id == 0:
                return 1 + depths[0]
            elif stage_id == 1:
                return 1 + depths[0] + depths[1]
            elif stage_id == 2:
                return 1 + depths[0] + depths[1] + depths[2]
            else:
                return 1 + depths[0] + depths[1] + depths[2]
    else:
        return num_max_layer - 1



def get_layer_id_for_convnext(var_name, max_layer_id):
    """Get the layer id to set the different learning rates in ``layer_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        max_layer_id (int): Maximum number of backbone layers.

    Returns:
        int: The id number corresponding to different learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = max_layer_id
        return layer_id
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = max_layer_id
        return layer_id
    else:
        return max_layer_id + 1


def get_stage_id_for_convnext(var_name, max_stage_id):
    """Get the stage id to set the different learning rates in ``stage_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        max_stage_id (int): Maximum number of backbone layers.

    Returns:
        int: The id number corresponding to different learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        return 0
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        return stage_id + 1
    else:
        return max_stage_id - 1


def get_layer_id_for_vit(var_name, max_layer_id):
    """Get the layer id to set the different learning rates.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.

    Returns:
        int: Returns the layer id of the key.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.layers'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return max_layer_id - 1


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class CustomLayerDecayOptimizerConstructor(DefaultOptimWrapperConstructor):

    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        parameter_groups = {}
        print_log(self.paramwise_cfg)
        backbone_small_lr = self.paramwise_cfg.get('backbone_small_lr', False)
        dino_head = self.paramwise_cfg.get('dino_head', False)
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        layer_decay_rate = self.paramwise_cfg.get('layer_decay_rate')
        depths = self.paramwise_cfg.get('depths')
        offset_lr_scale = self.paramwise_cfg.get('offset_lr_scale', 1.0)

        print_log('Build CustomLayerDecayOptimizerConstructor %f - %d' %
                    (layer_decay_rate, num_layers))
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or \
                    'relative_position' in name or \
                    'norm' in name or\
                    'sampling_offsets' in name:
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            layer_id = get_num_layer_for_swin(name, num_layers, depths)
            if layer_id == num_layers - 1 and dino_head and \
                    ('sampling_offsets' in name or 'reference_points' in name):
                group_name = 'layer_%d_%s_0.1x' % (layer_id, group_name)
            elif ('sampling_offsets' in name or 'reference_points' in name) and 'backbone' in name:
                group_name = 'layer_%d_%s_offset_lr_scale' % (layer_id,
                                                              group_name)
            else:
                group_name = 'layer_%d_%s' % (layer_id, group_name)

            if group_name not in parameter_groups:
                scale = layer_decay_rate ** (num_layers - layer_id - 1)
                if scale < 1 and backbone_small_lr == True:
                    scale = scale * 0.1
                if '0.1x' in group_name:
                    scale = scale * 0.1
                if 'offset_lr_scale' in group_name:
                    scale = scale * offset_lr_scale

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            print_log('Param groups = %s' % json.dumps(to_display, indent=2))

        # state_dict = module.state_dict()
        # for group_name in parameter_groups:
        #     group = parameter_groups[group_name]
        #     for name in group["param_names"]:
        #         group["params"].append(state_dict[name])

        params.extend(parameter_groups.values())
