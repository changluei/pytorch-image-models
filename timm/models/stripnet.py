""" StripNet

StripNet adapted for timm ImageNet pretraining while preserving the original
backbone module naming used by downstream detection code.
"""

import math
from functools import partial
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, to_2tuple, trunc_normal_
from ._builder import build_model_with_cfg
from ._manipulate import checkpoint
from ._registry import generate_default_cfgs, register_model

__all__ = ['StripNet', 'stripnet_t', 'stripnet_s']


class DWConv(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x)


class Mlp(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.GELU,
            drop: float = 0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StripBlock(nn.Module):
    def __init__(self, dim: int, k1: int, k2: int):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(
            dim, dim, kernel_size=(k1, k2), stride=1, padding=(k1 // 2, k2 // 2), groups=dim)
        self.conv_spatial2 = nn.Conv2d(
            dim, dim, kernel_size=(k2, k1), stride=1, padding=(k2 // 2, k1 // 2), groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)
        return x * attn


class Attention(nn.Module):
    def __init__(self, dim: int, k1: int, k2: int):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = StripBlock(dim, k1, k2)
        self.proj_2 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 4.,
            k1: int = 1,
            k2: int = 19,
            drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            layer_scale_init_value: float = 1e-2,
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, k1, k2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1[:, None, None] * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2[:, None, None] * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to patch embedding."""

    def __init__(
            self,
            patch_size: int = 7,
            stride: int = 4,
            in_chans: int = 3,
            embed_dim: int = 768,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, h, w = x.shape
        x = self.norm(x)
        return x, h, w


class StripNet(nn.Module):
    def __init__(
            self,
            img_size: int = 224,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dims: Tuple[int, ...] = (64, 128, 256, 512),
            mlp_ratios: Tuple[float, ...] = (8., 8., 4., 4.),
            k1s: Tuple[int, ...] = (1, 1, 1, 1),
            k2s: Tuple[int, ...] = (19, 19, 19, 19),
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Type[nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            depths: Tuple[int, ...] = (3, 4, 6, 3),
            num_stages: int = 4,
            act_layer: Type[nn.Module] = nn.GELU,
            **kwargs,
    ):
        super().__init__()
        assert num_stages == len(embed_dims) == len(depths) == len(mlp_ratios) == len(k1s) == len(k2s)
        assert global_pool in ('avg', '')
        if global_pool == '' and num_classes > 0:
            raise ValueError('StripNet requires global_pool="avg" when num_classes > 0.')

        self.img_size = img_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_stages = num_stages
        self.depths = tuple(depths)
        self.embed_dims = tuple(embed_dims)
        self.embed_dim = self.embed_dims[-1]
        self.num_features = self.head_hidden_size = self.embed_dim
        self.grad_checkpointing = False
        self.feature_info = [
            dict(num_chs=dim, reduction=4 * (2 ** i), module=f'norm{i + 1}')
            for i, dim in enumerate(self.embed_dims)
        ]

        dpr = torch.linspace(0, drop_path_rate, sum(self.depths)).tolist()
        cur = 0
        for i in range(self.num_stages):
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
                embed_dim=self.embed_dims[i],
            )
            block = nn.ModuleList([
                Block(
                    dim=self.embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    k1=k1s[i],
                    k2=k2s[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    act_layer=act_layer,
                )
                for j in range(self.depths[i])
            ])
            norm = norm_layer(self.embed_dims[i])
            cur += self.depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, mean=0, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _forward_stage(self, stage_idx: int, x: torch.Tensor) -> torch.Tensor:
        patch_embed = getattr(self, f'patch_embed{stage_idx + 1}')
        block = getattr(self, f'block{stage_idx + 1}')
        norm = getattr(self, f'norm{stage_idx + 1}')

        x, h, w = patch_embed(x)
        for blk in block:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        b = x.shape[0]
        x = x.flatten(2).transpose(1, 2)
        x = norm(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False):
        return dict(
            stem=r'^patch_embed1',
            blocks=r'^(?:patch_embed|block|norm)(\d+)',
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    def freeze_patch_emb(self):
        for param in self.patch_embed1.parameters():
            param.requires_grad = False

    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('avg', '')
            if global_pool == '' and num_classes > 0:
                raise ValueError('StripNet requires global_pool="avg" when num_classes > 0.')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_intermediates(self, x: torch.Tensor):
        outs = []
        for i in range(self.num_stages):
            x = self._forward_stage(i, x)
            outs.append(x)
        return outs

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_stages):
            x = self._forward_stage(i, x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            x = x[-1]
        if self.global_pool == 'avg':
            x = x.mean(dim=(2, 3))
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_stripnet(variant: str, pretrained: bool = False, **kwargs) -> StripNet:
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for StripNet models.')
    return build_model_with_cfg(StripNet, variant, pretrained, **kwargs)


def _cfg(url: str = '', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': (7, 7),
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed1.proj',
        'classifier': 'head',
        'fixed_input_size': False,
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'stripnet_t': _cfg(),
    'stripnet_s': _cfg(),
})


@register_model
def stripnet_t(pretrained: bool = False, **kwargs) -> StripNet:
    model_args = dict(
        embed_dims=(32, 64, 160, 256),
        depths=(3, 3, 5, 2),
        mlp_ratios=(8., 8., 4., 4.),
        k1s=(1, 1, 1, 1),
        k2s=(19, 19, 19, 19),
    )
    return _create_stripnet('stripnet_t', pretrained=pretrained, **dict(model_args, **kwargs))


@register_model
def stripnet_s(pretrained: bool = False, **kwargs) -> StripNet:
    model_args = dict(
        embed_dims=(64, 128, 320, 512),
        depths=(2, 2, 4, 2),
        mlp_ratios=(8., 8., 4., 4.),
        k1s=(1, 1, 1, 1),
        k2s=(19, 19, 19, 19),
    )
    return _create_stripnet('stripnet_s', pretrained=pretrained, **dict(model_args, **kwargs))
