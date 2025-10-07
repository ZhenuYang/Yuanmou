import torch
from timm.models.registry import register_model
from models import *


def get_deit_rpe_config():
    from irpe import get_rpe_config as _get_rpe_config
    rpe_config = _get_rpe_config(
        ratio=1.9,
        method="product",
        mode='ctx',
        shared_head=True,
        skip=0,
        rpe_on='',   # we close rpe
    )
    return rpe_config


def get_repeated_schedule(depth):
    return {
        'norm1': [[depth], [True]], 
        'norm2': [[depth], [True]], 
        'attn_rpe': [[depth], [True]], 
        'attn_qkv': [[depth], [True]],  
        'attn_proj': [[depth], [True]], 
        'mlp_fc1': [[depth], [True]], 
        'mlp_fc2': [[depth], [True]],
    }


@register_model
def aux_deit_tiny_patch16_224(pretrained=False, **kwargs):
    return deit_tiny_patch16_224(pretrained=pretrained,
                                 rpe_config=get_deit_rpe_config(),
                                 use_cls_token=False,
                                 repeated_times_schedule=get_repeated_schedule(12),
                                 **kwargs)


@register_model
def aux_deit_small_patch16_224(pretrained=False, **kwargs):
    return deit_small_patch16_224(pretrained=pretrained,
                                  rpe_config=get_deit_rpe_config(),
                                  use_cls_token=False,
                                  repeated_times_schedule=get_repeated_schedule(12),
                                  **kwargs)

@register_model
def aux_deit_small_patch16_224_distilled(pretrained=False, **kwargs):
    return deit_small_patch16_224_distilled(pretrained=pretrained,
                                  rpe_config=get_deit_rpe_config(),
                                  use_cls_token=True,
                                  repeated_times_schedule=get_repeated_schedule(12),
                                  **kwargs)

@register_model
def aux_deit_base_patch16_224(pretrained=False, **kwargs):
    return deit_base_patch16_224(pretrained=pretrained,
                                 rpe_config=get_deit_rpe_config(),
                                 use_cls_token=False,
                                 repeated_times_schedule=get_repeated_schedule(12),
                                 **kwargs)


@register_model
def aux_deit_base_patch16_384(pretrained=False, **kwargs):
    return deit_base_patch16_384(pretrained=pretrained,
                                 rpe_config=get_deit_rpe_config(),
                                 use_cls_token=False,
                                 repeated_times_schedule=get_repeated_schedule(12),
                                 **kwargs)


@register_model
def aux_deit_large_patch16_224(pretrained=False, **kwargs):
    return deit_large_patch16_224(pretrained=pretrained,
                                 rpe_config=get_deit_rpe_config(),
                                 use_cls_token=False,
                                 repeated_times_schedule=get_repeated_schedule(12),
                                 **kwargs)


@register_model
def aux_deit_small_patch16_224_L24(pretrained=False, **kwargs):
    return deit_small_patch16_224_L24(pretrained=pretrained,
                                  rpe_config=get_deit_rpe_config(),
                                  use_cls_token=False,
                                  repeated_times_schedule=get_repeated_schedule(depth = 24),
                                  **kwargs)


@register_model
def aux_deit_small_patch16_224_L36(pretrained=False, **kwargs):
    return deit_small_patch16_224_L36(pretrained=pretrained,
                                  rpe_config=get_deit_rpe_config(),
                                  use_cls_token=False,
                                  repeated_times_schedule=get_repeated_schedule(depth = 36),
                                  **kwargs)

@register_model
def aux_deit_small_patch16_224_L48(pretrained=False, **kwargs):
    return deit_small_patch16_224_L48(pretrained=pretrained,
                                  rpe_config=get_deit_rpe_config(),
                                  use_cls_token=False,
                                  repeated_times_schedule=get_repeated_schedule(depth = 48),
                                  **kwargs)

