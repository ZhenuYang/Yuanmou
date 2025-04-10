import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


__all__ = [
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224','deit_base_distilled_patch16_224', 'deit_base_patch16_384', 'deit_base_distilled_patch16_384',
    'deit_tiny_patch16_224_L3', 'deit_tiny_patch16_224_L4', 'deit_tiny_patch16_224_L5','deit_tiny_patch16_224_L6', 'deit_tiny_patch16_224_L7', 'deit_tiny_patch16_224_L8','deit_tiny_patch16_224_L9', 'deit_tiny_patch16_224_L10', 'deit_tiny_patch16_224_L11','deit_tiny_patch16_224_L12', 
    'deit_small_patch16_224_L3', 'deit_small_patch16_224_L4', 'deit_small_patch16_224_L5', 'deit_small_patch16_224_L6', 'deit_small_patch16_224_L7', 'deit_small_patch16_224_L8','deit_small_patch16_224_L9', 'deit_small_patch16_224_L10','deit_small_patch16_224_L11','deit_small_patch16_224_L12', 
    'deit_base_patch16_224_L3', 'deit_base_patch16_224_L4', 'deit_base_patch16_224_L5', 'deit_base_patch16_224_L6', 'deit_base_patch16_224_L7', 'deit_base_patch16_224_L8', 'deit_base_patch16_224_L9', 'deit_base_patch16_224_L10', 'deit_base_patch16_224_L11', 'deit_base_patch16_224_L12',
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2



@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, pretrained_cfg=None, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model



@register_model
def deit_tiny_patch16_224_L3(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 3
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=3, num_heads=3, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained

    return model


@register_model
def deit_tiny_patch16_224_L4(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 4
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=4, num_heads=3, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained

    return model


@register_model
def deit_tiny_patch16_224_L5(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 5
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=5, num_heads=3, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained

    return model


@register_model
def deit_tiny_patch16_224_L6(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 6
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model


@register_model
def deit_tiny_patch16_224_L7(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 7
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=7, num_heads=3, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model


@register_model
def deit_tiny_patch16_224_L8(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 8
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=8, num_heads=3, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model


@register_model
def deit_tiny_patch16_224_L9(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 9
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=9, num_heads=3, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model


@register_model
def deit_tiny_patch16_224_L10(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 10
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=10, num_heads=3, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model


@register_model
def deit_tiny_patch16_224_L11(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 11
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=11, num_heads=3, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model


@register_model
def deit_tiny_patch16_224_L12(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model



@register_model
def deit_small_patch16_224_L3(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 3
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=3, num_heads=6, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model


@register_model
def deit_small_patch16_224_L4(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 4
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model


@register_model
def deit_small_patch16_224_L5(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 5
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=5, num_heads=6, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model


@register_model
def deit_small_patch16_224_L6(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 6
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model


@register_model
def deit_small_patch16_224_L7(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 7
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=7, num_heads=6, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model


@register_model
def deit_small_patch16_224_L8(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 8
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model


@register_model
def deit_small_patch16_224_L9(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 9
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=9, num_heads=6, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model


@register_model
def deit_small_patch16_224_L10(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 10
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=10, num_heads=6, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model


@register_model
def deit_small_patch16_224_L11(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 11
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=11, num_heads=6, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model


@register_model
def deit_small_patch16_224_L12(pretrained=False, pretrained_cfg=None, **kwargs):
    # depth 12 -> 12
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    assert not pretrained
    
    return model



@register_model
def deit_base_patch16_224_L3(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=3, num_heads=12, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    
    return model



@register_model
def deit_base_patch16_224_L4(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    
    return model


@register_model
def deit_base_patch16_224_L5(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=5, num_heads=12, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    
    return model



@register_model
def deit_base_patch16_224_L6(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    return model



@register_model
def deit_base_patch16_224_L7(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=7, num_heads=12, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    
    return model


@register_model
def deit_base_patch16_224_L8(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=8, num_heads=12, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    
    return model



@register_model
def deit_base_patch16_224_L9(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=9, num_heads=12, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    
    return model



@register_model
def deit_base_patch16_224_L10(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=10, num_heads=12, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    return model



@register_model
def deit_base_patch16_224_L11(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    return model



@register_model
def deit_base_patch16_224_L12(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, global_pool='avg', class_token=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()

    return model



