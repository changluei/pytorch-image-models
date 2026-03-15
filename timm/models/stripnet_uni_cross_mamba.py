import math
import warnings
from functools import partial

import torch
import torch.nn as nn
from mmcv import print_log
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import constant_init, normal_init, trunc_normal_init
from mmcv.runner import BaseModule, get_dist_info
from timm.layers import DropPath
from torch.nn.modules.utils import _pair as to_2tuple

from ._registry import register_model

from mamba_ssm import Mamba



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AxialCrossMambaUni(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.row_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.col_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.diag_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.anti_diag_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        
        self.idx_cache = {}

    def _get_diag_indices(self, h, w, device):
        """生成并缓存主对角线遍历的 1D 索引"""
        key = (h, w)
        if key in self.idx_cache:
            return self.idx_cache[key]

        coords = []
        for offset in range(-(h - 1), w):
            for i in range(h):
                j = i + offset
                if 0 <= j < w:
                    coords.append(i * w + j)
                    
        idx = torch.tensor(coords, dtype=torch.long, device=device)
        inv_idx = torch.argsort(idx)
        
        self.idx_cache[key] = (idx, inv_idx)
        return idx, inv_idx

    def forward(self, x):
        b, c, h, w = x.shape
        idx, inv_idx = self._get_diag_indices(h, w, x.device)

        row_tokens = x.flatten(2, 3).transpose(1, 2) 
        row_out = self.row_mamba(row_tokens)
        row_feat = row_out.transpose(1, 2).reshape(b, c, h, w)

        col_tokens = x.transpose(2, 3).flatten(2, 3).transpose(1, 2)  # [B, H*W, C]
        col_out = self.col_mamba(col_tokens)
        col_feat = col_out.transpose(1, 2).reshape(b, c, w, h).transpose(2, 3)

        diag_tokens = row_tokens[:, idx, :]  
        diag_out = self.diag_mamba(diag_tokens)
        diag_feat = diag_out[:, inv_idx, :].transpose(1, 2).reshape(b, c, h, w)

        x_flipped = torch.flip(x, dims=[3])
        anti_tokens = x_flipped.flatten(2, 3).transpose(1, 2)[:, idx, :]
        anti_out = self.anti_diag_mamba(anti_tokens)
        anti_feat = anti_out[:, inv_idx, :].transpose(1, 2).reshape(b, c, h, w)
        anti_feat = torch.flip(anti_feat, dims=[3])

        logit = (row_feat + col_feat + diag_feat + anti_feat) / 4.0
        gate = torch.sigmoid(logit)
        return x * gate


class StripBlock(nn.Module):
    def __init__(
        self,
        dim,
        k1,
        k2,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
    ):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.axial_mamba = AxialCrossMambaUni(
            dim=dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.conv_spatial1 = nn.Conv2d(
            dim,
            dim,
            kernel_size=(k1, k2),
            stride=1,
            padding=(k1 // 2, k2 // 2),
            groups=dim,
        )
        self.conv_spatial2 = nn.Conv2d(
            dim,
            dim,
            kernel_size=(k2, k1),
            stride=1,
            padding=(k2 // 2, k1 // 2),
            groups=dim,
        )
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.axial_mamba(attn)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)
        return x * attn


class Attention(nn.Module):
    def __init__(
        self,
        d_model,
        k1,
        k2,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
    ):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = StripBlock(
            d_model,
            k1,
            k2,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.,
        k1=1,
        k2=19,
        drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_cfg=None,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
    ):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)

        self.attn = Attention(
            dim,
            k1,
            k2,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


@register_model
class StripSMambaNet(BaseModule):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        embed_dims=[64, 128, 256, 512],
        mlp_ratios=[8, 8, 4, 4],
        k1s=[1, 1, 1, 1],
        k2s=[19, 19, 19, 19],
        drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3],
        num_stages=4,
        pretrained=None,
        init_cfg=None,
        norm_cfg=None,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        debug_backbone_stats=False,
        debug_backbone_interval=200,
        debug_backbone_log_first_n=10,
        num_classes=0,
        global_pool='avg',
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
        cache_dir=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and isinstance(pretrained, str)), 'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif isinstance(pretrained, bool):
            if pretrained:
                warnings.warn(
                    '`pretrained=True` is set, but no timm pretrained weights are registered for StripSMambaNet. '
                    'Using random initialization.')
        elif pretrained is not None:
            raise TypeError('pretrained must be a bool, str, or None')

        self.depths = depths
        self.num_stages = num_stages
        self.embed_dim = embed_dims[-1]
        self.num_features = self.embed_dim
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.debug_backbone_stats = debug_backbone_stats
        self.debug_backbone_interval = max(int(debug_backbone_interval), 1)
        self.debug_backbone_log_first_n = max(int(debug_backbone_log_first_n), 0)
        self._debug_forward_iter = 0

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                norm_cfg=norm_cfg,
            )

            block = nn.ModuleList([
                Block(
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    k1=k1s[i],
                    k2=k2s[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    norm_cfg=norm_cfg,
                    mamba_d_state=mamba_d_state,
                    mamba_d_conv=mamba_d_conv,
                    mamba_expand=mamba_expand,
                )
                for j in range(depths[i])
            ])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    @staticmethod
    def _tensor_stats(x):
        if x.numel() == 0:
            return dict(mean=0.0, min=0.0, max=0.0, absmax=0.0)
        finite = x[torch.isfinite(x)]
        if finite.numel() == 0:
            return dict(mean=0.0, min=0.0, max=0.0, absmax=0.0)
        return dict(
            mean=float(finite.mean().item()),
            min=float(finite.min().item()),
            max=float(finite.max().item()),
            absmax=float(finite.abs().max().item()))

    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            # 1. collect all mamba module
            mamba_submodule_ids = set()
            for m in self.modules():
                if isinstance(m, Mamba):
                    mamba_submodule_ids.update(id(sm) for sm in m.modules())
            
            for m in self.modules():
                if id(m) in mamba_submodule_ids:
                    continue

                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(StripSMambaNet, self).init_weights()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if global_pool:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_head(self, x, pre_logits=False):
        if isinstance(x, (list, tuple)):
            x = x[-1]
        x = x.mean(dim=(2, 3))
        if pre_logits:
            return x
        return self.head(x)

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        self._debug_forward_iter += 1
        rank, _ = get_dist_info()
        log_debug = (
            self.training and self.debug_backbone_stats and rank == 0 and
            (self._debug_forward_iter <= self.debug_backbone_log_first_n or
             self._debug_forward_iter % self.debug_backbone_interval == 0))

        stage_debug_info = [] if log_debug else None
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)

            if log_debug:
                patch_nonfinite = int((~torch.isfinite(x)).sum().item())
                patch_stats = self._tensor_stats(x)
                first_nonfinite_block = -1
                block_nonfinite = 0
                block_absmax = 0.0

            for blk_idx, blk in enumerate(block):
                x = blk(x)
                if log_debug:
                    nonfinite = int((~torch.isfinite(x)).sum().item())
                    if nonfinite > 0 and first_nonfinite_block < 0:
                        first_nonfinite_block = blk_idx
                    block_nonfinite += nonfinite
                    block_absmax = max(block_absmax, self._tensor_stats(x)['absmax'])
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
            if log_debug:
                out_nonfinite = int((~torch.isfinite(x)).sum().item())
                out_stats = self._tensor_stats(x)
                stage_debug_info.append(
                    dict(
                        stage=i + 1,
                        shape=f"{tuple(x.shape)}",
                        patch_nonfinite=patch_nonfinite,
                        patch_stats=patch_stats,
                        block_nonfinite=block_nonfinite,
                        first_nonfinite_block=first_nonfinite_block,
                        block_absmax=block_absmax,
                        out_nonfinite=out_nonfinite,
                        out_stats=out_stats))
        if log_debug:
            for info in stage_debug_info:
                print_log(
                    f"[BackboneDebug][iter={self._debug_forward_iter}] "
                    f"stage={info['stage']} shape={info['shape']} "
                    f"patch_nf={info['patch_nonfinite']} "
                    f"patch(m/n/x/abs)={info['patch_stats']['mean']:.4f}/"
                    f"{info['patch_stats']['min']:.4f}/"
                    f"{info['patch_stats']['max']:.4f}/"
                    f"{info['patch_stats']['absmax']:.4f} "
                    f"blocks_nf={info['block_nonfinite']} "
                    f"first_nf_block={info['first_nonfinite_block']} "
                    f"blocks_absmax={info['block_absmax']:.4f} "
                    f"out_nf={info['out_nonfinite']} "
                    f"out(m/n/x/abs)={info['out_stats']['mean']:.4f}/"
                    f"{info['out_stats']['min']:.4f}/"
                    f"{info['out_stats']['max']:.4f}/"
                    f"{info['out_stats']['absmax']:.4f}",
                    logger=None)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        if self.num_classes > 0:
            x = self.forward_head(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """Convert patch embedding weight from manual patchify + linear proj to conv."""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict
