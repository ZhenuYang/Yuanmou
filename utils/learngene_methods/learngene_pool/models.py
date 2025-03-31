import os

import torch
import torch.nn as nn
from functools import partial

from timm.models import register_model
# from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, Block, VisionTransformer


__all__ = [
    'deit_tiny_patch16_224',
    'deit_small_patch16_224',
    'deit_base_patch16_224', 'deit_base3_patch16_224', 'deit_base6_patch16_224', 'deit_base9_patch16_224',
    'deit_small_patch16_224', 'deit_small3_patch16_224', 'deit_small6_patch16_224', 'deit_small9_patch16_224',
]

class Block_OutMidMap(Block):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs):
        super().__init__(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, **kwargs)
        self.embed_dim = dim
    
    def forward_OutMidMap(self, x):
        out_att = self.attn(self.norm1(x))
        x = x + self.drop_path1(self.ls1(out_att))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, out_att


class ViT_OutMidMap(VisionTransformer):
    """Vision Transformer 
    This module has the same function as ViT, but outputing the embedding, attention map, hidden states and output logits of the ViT as TinyBert.
    """
    def __init__(self, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs):
        super().__init__(patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, **kwargs)

        self.block = Block_OutMidMap
        self.blocks = nn.Sequential(*[
            self.block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
            )
            for i in range(depth)])

    def extract_block_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        outs = {}
        outs[-1] = x.detach()
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            outs[i] = x.detach()
        return outs
    
    def forward_patch_embed(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        return x
    
    def forward_block(self, x):
        out_att = []
        out_map = []
        out_embed = x  # get embedding features
        for i, blk in enumerate(self.blocks):
            x, att = blk.forward_OutMidMap(x)
            out_att.append(att)  # get attention maps
            out_map.append(x)  # get hidden features
        return x, out_att, out_map, out_embed
    
    def forward_features(self, x):
        x = self.forward_patch_embed(x)
        x, out_atts, out_maps, out_embed = self.forward_block(x)
        out = self.forward_head(x)
        return out, out_atts, out_maps, out_embed


@register_model
def deit_tiny_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ViT_OutMidMap(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        local_dir = 'pretrained/deit_tiny_patch16_224-a1311bcf.pth'
        if os.path.exists(local_dir):
            checkpoint = torch.load(local_dir, map_location='cpu')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_tiny3_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ViT_OutMidMap(
        patch_size=16, embed_dim=192, depth=3, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        local_dir = 'pretrained/deit_tiny_patch16_224-a1311bcf.pth'
        if os.path.exists(local_dir):
            checkpoint = torch.load(local_dir, map_location='cpu')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_tiny6_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ViT_OutMidMap(
        patch_size=16, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        local_dir = 'pretrained/deit_tiny_patch16_224-a1311bcf.pth'
        if os.path.exists(local_dir):
            checkpoint = torch.load(local_dir, map_location='cpu')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_tiny9_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ViT_OutMidMap(
        patch_size=16, embed_dim=192, depth=9, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        local_dir = 'pretrained/deit_tiny_patch16_224-a1311bcf.pth'
        if os.path.exists(local_dir):
            checkpoint = torch.load(local_dir, map_location='cpu')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ViT_OutMidMap(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        local_dir = 'pretrained/deit_small_patch16_224-cd65a155.pth'
        if os.path.exists(local_dir):
            checkpoint = torch.load(local_dir, map_location='cpu')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_small3_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ViT_OutMidMap(
        patch_size=16, embed_dim=384, depth=3, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        local_dir = 'pretrained/deit_small_patch16_224-cd65a155.pth'
        if os.path.exists(local_dir):
            checkpoint = torch.load(local_dir, map_location='cpu')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_small6_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ViT_OutMidMap(
        patch_size=16, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        local_dir = 'pretrained/deit_small_patch16_224-cd65a155.pth'
        if os.path.exists(local_dir):
            checkpoint = torch.load(local_dir, map_location='cpu')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_small9_patch16_224(pretrained=False, pretrained_cfg=None, **kwargs):
    model = ViT_OutMidMap(
        patch_size=16, embed_dim=384, depth=9, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        local_dir = 'pretrained/deit_small_patch16_224-cd65a155.pth'
        if os.path.exists(local_dir):
            checkpoint = torch.load(local_dir, map_location='cpu')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_base_patch16_224(pretrained=False, pretrained_cfg=None,  **kwargs):
    model = ViT_OutMidMap(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        local_dir = './pretrained/deit_base_patch16_224-b5f2ef4d.pth'
        if os.path.exists(local_dir):
            checkpoint = torch.load(local_dir, map_location='cpu')["model"]
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )["model"]
        model.load_state_dict(checkpoint)
    return model

@register_model
def deit_base3_patch16_224(pretrained=False, pretrained_cfg=None,  **kwargs):
    model = ViT_OutMidMap(
        patch_size=16, embed_dim=768, depth=3, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        local_dir = 'pretrained/deit_base_patch16_224-b5f2ef4d.pth'
        if os.path.exists(local_dir):
            checkpoint = torch.load(local_dir, map_location='cpu')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_base6_patch16_224(pretrained=False, pretrained_cfg=None,  **kwargs):
    model = ViT_OutMidMap(
        patch_size=16, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        local_dir = 'pretrained/deit_base_patch16_224-b5f2ef4d.pth'
        if os.path.exists(local_dir):
            checkpoint = torch.load(local_dir, map_location='cpu')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_base9_patch16_224(pretrained=False, pretrained_cfg=None,  **kwargs):
    model = ViT_OutMidMap(
        patch_size=16, embed_dim=768, depth=9, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        local_dir = 'pretrained/deit_base_patch16_224-b5f2ef4d.pth'
        if os.path.exists(local_dir):
            checkpoint = torch.load(local_dir, map_location='cpu')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
        model.load_state_dict(checkpoint["model"])
    return model


    

    

    

    

