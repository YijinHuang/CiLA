# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Integrated into solo from
# https://github.com/facebookresearch/moco-v3/blob/main/vits.py

import math
from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models._manipulate import checkpoint_seq


class PatchSampler(object):
    def __init__(self, mask_ratio=0.25):
        self.mask_ratio = mask_ratio

    def __call__(self, pmap):
        B, C, H, W = pmap.shape
        num_sample = int((1 - self.mask_ratio) * H * W)

        feat_idx = pmap.flatten(1).argsort(descending=True)[:,:num_sample]
        feat_idx += 1 # class embedding concat before the image embedding
        cls_idx = torch.zeros((B, 1), dtype=torch.int64, device=pmap.device)
        active_idx = torch.cat([cls_idx, feat_idx], dim=1)
        return active_idx


class VisionTransformerSSiT(VisionTransformer):
    def __init__(self, mask_ratio=0.25, feat_concat=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(
                6.0 / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim)
            )
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)
            
        self.mask_ratio = mask_ratio
        self.mask_pooling = nn.MaxPool2d(kernel_size=self.patch_embed.patch_size, stride=self.patch_embed.patch_size)
        self.patch_sampler = PatchSampler(mask_ratio=mask_ratio)
        
        self.feat_concat = feat_concat
        if feat_concat:
            self.head = nn.Linear(self.embed_dim * 2, kwargs['num_classes']) if kwargs['num_classes'] > 0 else nn.Identity()

    def build_2d_sincos_position_embedding(self, temperature=10000.0):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        # https://pytorch.org/docs/stable/generated/torch.meshgrid.html
        # indexing –
        # (str, optional): the indexing mode, either “xy” or “ij”, defaults to “ij”.
        # If “xy” is selected, the first dimension corresponds to the cardinality of
        # the second input and the second dimension corresponds to the cardinality of the first input.
        # If “ij” is selected, the dimensions are in the same order as the cardinality of the inputs.
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert (
            self.embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1
        )[None, :, :]

        assert self.num_prefix_tokens == 1, "Assuming one and only one token, [cls]"
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False

    def forward_features(self, x: torch.Tensor, saliency_map=None) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        if saliency_map is not None and self.mask_ratio < 1:
            pmap = self.mask_pooling(saliency_map)
            active_idx = self.patch_sampler(pmap)
            active_idx = active_idx.unsqueeze(-1).repeat(1, 1, self.embed_dim)
            x = torch.gather(x, dim=1, index=active_idx)

        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.feat_concat:
            feats = x[:, 1:].mean(dim=1)
            x = torch.cat((x[:, 0], feats), dim=1)        
        else:
            x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)
        
    def forward(self, x: torch.Tensor, saliency_map=None) -> torch.Tensor:
        f = self.forward_features(x, saliency_map)
        x = self.forward_head(f)
        return x, f


# extrapolated to the dimensions of vit_tiny
def vit_tiny(**kwargs):
    # patch_size is 16 by default
    model = VisionTransformerSSiT(
        embed_dim=192,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def vit_small(**kwargs):
    # patch_size is 16 by default
    model = VisionTransformerSSiT(
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def vit_base(**kwargs):
    # patch_size is 16 by default
    model = VisionTransformerSSiT(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


# extrapolated to the dimensions of vit_large
def vit_large(**kwargs):
    # patch_size is 16 by default
    model = VisionTransformerSSiT(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model
