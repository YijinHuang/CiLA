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

from typing import Any, Dict, List, Sequence, Tuple

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from solo.losses.mocov3 import mocov3_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
from solo.utils.metrics import accuracy_at_k


class SSiT(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements MoCo V3 (https://arxiv.org/abs/2104.02057).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of projected features.
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
                temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature
        self.saliency_threshold: float = cfg.method_kwargs.saliency_threshold
        self.segmentation_weight: float = cfg.method_kwargs.segmentation_weight

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim
        patch_size: int = cfg.method_kwargs.patch_size

        if "resnet" in self.backbone_name:
            # projector
            self.projector = self._build_mlp(
                2,
                self.features_dim,
                proj_hidden_dim,
                proj_output_dim,
            )
            # momentum projector
            self.momentum_projector = self._build_mlp(
                2,
                self.features_dim,
                proj_hidden_dim,
                proj_output_dim,
            )

            # predictor
            self.predictor = self._build_mlp(
                2,
                proj_output_dim,
                pred_hidden_dim,
                proj_output_dim,
                last_bn=False,
            )
        else:
            # specifically for ViT but allow all the other backbones
            # projector
            self.projector = self._build_mlp(
                3,
                self.features_dim,
                proj_hidden_dim,
                proj_output_dim,
            )
            # momentum projector
            self.momentum_projector = self._build_mlp(
                3,
                self.features_dim,
                proj_hidden_dim,
                proj_output_dim,
            )

            # predictor
            self.predictor = self._build_mlp(
                2,
                proj_output_dim,
                pred_hidden_dim,
                proj_output_dim,
            )
            
            # segmentor
            self.saliency_segmentor = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.features_dim,
                    out_channels=patch_size ** 2,
                    kernel_size=1,
                ),
                nn.PixelShuffle(upscale_factor=patch_size),
            )

        initialize_momentum_params(self.projector, self.momentum_projector)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(SSiT, SSiT).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.patch_size")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()},
            {"name": "predictor", "params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    def saliency_segmentation_loss(self, f, m):
        f = f[:, 1:]
        m = (m > self.saliency_threshold).float()

        B, L, C = f.shape
        H = W = int(L ** 0.5)
        f = f.permute(0, 2, 1).reshape(B, C, H, W)
        ss = self.saliency_segmentor(f)

        bce = F.binary_cross_entropy_with_logits(ss, m)
        return bce

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats, patch_feats = self.backbone(X)
        logits = self.classifier(feats.detach())

        z = self.projector(feats)
        q = self.predictor(z)
        return {
            "logits": logits,
            "feats": feats,
            "patch_feats": patch_feats,
            "q": q,
            "z": z,
        }

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor, M: torch.Tensor) -> Dict[str, Any]:
        """Momentum forward method. Children methods should call this function,
        modify the ouputs (without deleting anything) and return it.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats, _ = self.momentum_backbone(X, M)
        k = self.momentum_projector(feats)
        return {"feats": feats, "k": k}

    def _shared_step_momentum(self, X: torch.Tensor, targets: torch.Tensor, M: torch.Tensor) -> Dict[str, Any]:
        """Forwards a batch of images X in the momentum backbone and optionally computes the
        classification loss, the logits, the features, acc@1 and acc@5 for of momentum classifier.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict[str, Any]:
                a dict containing the classification loss, logits, features, acc@1 and
                acc@5 of the momentum backbone / classifier.
        """

        out = self.momentum_forward(X, M)

        if self.momentum_classifier is not None:
            feats = out["feats"]
            logits = self.momentum_classifier(feats)

            loss = F.cross_entropy(logits, targets, ignore_index=-1)
            acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, 5))
            out.update({"logits": logits, "loss": loss, "acc1": acc1, "acc5": acc5})

        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for MoCo V3 reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of MoCo V3 and classification loss.
        """
        _, X, M, targets = batch
        outs = super(BaseMomentumMethod, self).training_step((_, X, targets), batch_idx)

        X = [X] if isinstance(X, torch.Tensor) else X

        # remove small crops
        X = X[: self.num_large_crops]

        momentum_outs = [self._shared_step_momentum(x, targets, m) for x, m in zip(X, M)]
        momentum_outs = {
            "momentum_" + k: [out[k] for out in momentum_outs] for k in momentum_outs[0].keys()
        }

        if self.momentum_classifier is not None:
            # momentum loss and stats
            momentum_outs["momentum_loss"] = (
                sum(momentum_outs["momentum_loss"]) / self.num_large_crops
            )
            momentum_outs["momentum_acc1"] = (
                sum(momentum_outs["momentum_acc1"]) / self.num_large_crops
            )
            momentum_outs["momentum_acc5"] = (
                sum(momentum_outs["momentum_acc5"]) / self.num_large_crops
            )

            metrics = {
                "train_momentum_class_loss": momentum_outs["momentum_loss"],
                "train_momentum_acc1": momentum_outs["momentum_acc1"],
                "train_momentum_acc5": momentum_outs["momentum_acc5"],
            }
            self.log_dict(metrics, on_epoch=True, sync_dist=True)

            # adds the momentum classifier loss together with the general loss
            outs["loss"] += momentum_outs["momentum_loss"]

        outs.update(momentum_outs)

        class_loss = outs["loss"]
        Q = outs["q"]
        K = outs["momentum_k"]
        F = outs["patch_feats"]

        contrastive_loss = mocov3_loss_func(
            Q[0], K[1], temperature=self.temperature
        ) + mocov3_loss_func(Q[1], K[0], temperature=self.temperature)

        segmentation_loss = self.saliency_segmentation_loss(
            F[0], M[0]
        ) + self.saliency_segmentation_loss(F[1], M[1])

        metrics = {
            "train_contrastive_loss": contrastive_loss,
            "segmentation_loss": segmentation_loss,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return contrastive_loss + self.segmentation_weight * segmentation_loss + class_loss
