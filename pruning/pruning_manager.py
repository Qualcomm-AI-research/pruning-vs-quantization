# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

from collections import namedtuple

import numpy as np
import torch
from torch import nn

from pruning.pruning_utils import TopKMaskStraightThrough, sparsity_level_cubic
from utils import ClassEnumOptions


class PruningManager(nn.Module):
    def __init__(self, owner, tiler, tile_size, **kwargs):
        super().__init__()
        self.owner = [owner]
        self.tiler = tiler(tile_size)
        self.tile_size = tile_size
        self.freeze_mask = False

    def calculate_regularizer_value(self):
        raise NotImplementedError

    def get_regularizer_value(self):
        return 0.0


class ZhuGuptaPruner(PruningManager):
    def __init__(self, owner, s_i, s_f, t_0, delta_t, n, allow_relearn, **kwargs):
        super(ZhuGuptaPruner, self).__init__(owner, **kwargs)
        self.iter = 0

        self.s_i = s_i
        self.s_f = s_f
        self.t_0 = t_0
        self.delta_t = delta_t
        self.n = n
        self.prev_sparsity = 0
        self.allow_relearn = allow_relearn

    def forward(self, x, bias):
        if not hasattr(self, "mask"):
            self.register_buffer("mask", torch.ones_like(x))

        if self.training and not self.freeze_mask:
            self.iter += 1  # This doesn't work if forward is called more than once per iter
            if self.time_to_update_mask():
                if self.allow_relearn:
                    update_input = x
                else:
                    update_input = x * self.mask
                self.update_mask(update_input)

        return x * self.mask, bias

    def get_sparsity_level(self):
        return sparsity_level_cubic(
            s_f=self.s_f, s_i=self.s_i, t=self.iter, t_0=self.t_0, n=self.n, delta_t=self.delta_t
        )

    def time_to_update_mask(self):
        t, t_0 = self.iter, self.t_0
        return t_0 < t <= self.n * self.delta_t + t_0 and (t - t_0) % self.delta_t == 0

    def update_mask(self, x):
        s_t = self.get_sparsity_level()
        x_tiled, _ = self.tiler(x)  # shape: N x tile_size

        if x_tiled is None:
            self.mask = torch.Tensor([1.0]).to(x.device)
            return

        tmp_mask = torch.ones_like(x_tiled[:, 0])

        values = abs(x_tiled.sum(1))
        indices = torch.argsort(values)
        n_prune = int(s_t * tmp_mask.shape[0])
        tmp_mask[indices[:n_prune]] = 0

        mask_tiled, orig_shape = self.tiler(self.mask)
        mask_tiled = mask_tiled * tmp_mask.view(-1, 1)
        self.mask = self.tiler.inverse(mask_tiled, orig_shape)


class BalancedZhuGuptaPruner(ZhuGuptaPruner):
    def __init__(self, owner, zeros_per_tile, **kwargs):
        kwargs["s_f"] = 1.0
        super(BalancedZhuGuptaPruner, self).__init__(owner, **kwargs)

        self.zeros_per_tile = zeros_per_tile
        self.mask_initialized = False

    def update_mask(self, x):
        s_t = self.get_sparsity_level()

        x_tiled, _ = self.tiler(x)  # shape: N x tile_size

        if x_tiled is None:
            self.mask = torch.Tensor([1.0]).to(x.device)
            return

        mask_tiled, orig_shape = self.tiler(self.mask)
        tmp_mask = torch.ones_like(x_tiled)

        values = self.metric(x_tiled)
        indices = torch.argsort(values)

        num_weights_keep_within_tile = self.tile_size - self.zeros_per_tile
        values, topk_indices = torch.topk(
            x_tiled**2, num_weights_keep_within_tile, dim=1, largest=False
        )
        topk_indices = topk_indices[indices]

        n_prune = int(s_t * tmp_mask.shape[0])
        tmp_mask[indices[:n_prune].unsqueeze(1), topk_indices[:n_prune]] = 0

        mask_tiled = mask_tiled * tmp_mask
        self.mask = self.tiler.inverse(mask_tiled, orig_shape)


class BalancedOneShotPruner(PruningManager):
    def __init__(self, owner, zeros_per_tile, tile_size, tiler, **kwargs):
        super(BalancedOneShotPruner, self).__init__(owner, tiler, tile_size, **kwargs)
        self.zeros_per_tile = zeros_per_tile
        self.tile_size = tile_size
        self.mask_initialized = False

    def forward(self, x, bias):
        if not hasattr(self, "mask"):
            self.register_buffer("mask", torch.ones_like(x))

        if hasattr(self, "mask") and self.training and not self.mask_initialized:
            self.create_mask(x)
            self.mask_initialized = True

        return x * self.mask, bias

    def create_mask(self, x):
        x_tiled, orig_shape = self.tiler(x)

        if x_tiled is None:
            self.mask = torch.Tensor([1.0]).to(x.device)
            return

        num_weights_keep = self.tile_size - self.zeros_per_tile
        values, indices = torch.topk(x_tiled**2, num_weights_keep, dim=1)

        mask = torch.zeros_like(x_tiled).scatter_(1, indices, torch.ones_like(indices).float())

        self.mask = self.tiler.inverse(mask, orig_shape)

    def get_sparsity_level(self):
        return self.zeros_per_tile / self.tile_size


class BalancedBatchShapedPruner(PruningManager):
    def __init__(self, owner, zeros_per_tile, tile_size, tiler, **kwargs):
        super().__init__(owner, tiler, tile_size, **kwargs)

        self.zeros_per_tile = zeros_per_tile
        self.tile_size = tile_size
        self.mask_initialized = False

        ref = torch.ones(1, self.tile_size)
        ref[:, : self.zeros_per_tile] = 0.0
        self.ref = ref

        self.mask_params = None

    def forward(self, x, bias):
        self.create_mask(x)
        return self.mask * x, bias

    def create_mask(self, x):
        x_tiled, orig_shape = self.tiler(x)

        if self.mask_params is None:
            self.mask_params = torch.nn.Parameter(torch.zeros_like(x_tiled)).to(x_tiled.device)

        self.tiled_mask = torch.sigmoid(self.mask_params).to(x_tiled.device)
        scale = 1.0 / self.tiled_mask.max(-1, keepdims=True)[0]
        mask = self.tiler.inverse(scale * self.tiled_mask, orig_shape)

        if not hasattr(self, "mask"):
            self.register_buffer("mask", mask)
        else:
            self.mask = mask

    def get_regularizer_value(self):
        if self.ref.device != self.mask_params.device:
            self.ref = self.ref.to(self.mask_params.device)
        ref = self.ref.expand(self.mask_params.shape[0], -1)
        sorted_mask, _ = torch.sort(self.tiled_mask)
        return torch.norm(sorted_mask - ref)


class MovementPruner(PruningManager):
    def __init__(self, owner, s_i, s_f, t_0, delta_t, n, metric, **kwargs):
        super(MovementPruner, self).__init__(owner, **kwargs)

        self.metric = metric
        self.s_i = s_i
        self.s_f = s_f
        self.t_0 = t_0
        self.delta_t = delta_t
        self.n = n
        self.prev_sparsity = 0
        self.importance_score_mode = kwargs["importance_score_mode"]

        self.iter = 0

    def forward(self, x, bias):
        if self.training:
            self.iter += 1
        x_tiled, orig_shape = self.tiler(x)

        # only unstructured pruning is supported for now
        if self.tiler.tile_size > 1:
            raise NotImplementedError("only unstructured pruning is supported for now")
        if not hasattr(self, "importance_score") and self.importance_score_mode == "learned":
            self.register_parameter("importance_score", nn.Parameter(torch.ones_like(x)))

        sparsity_level = self.get_sparsity_level()

        if self.importance_score_mode == "learned":
            importance_score_tiled, shape = self.tiler(self.importance_score)
            assert shape == orig_shape
            mask = TopKMaskStraightThrough.apply(importance_score_tiled.flatten(), sparsity_level)
        else:
            mask = TopKMaskStraightThrough.apply(torch.abs(x_tiled).flatten(), sparsity_level)
        weight_masked = self.tiler.inverse(mask, orig_shape) * x

        return weight_masked, bias

    def get_sparsity_level(self):
        return sparsity_level_cubic(
            s_f=self.s_f, s_i=self.s_i, t=self.iter, t_0=self.t_0, n=self.n, delta_t=self.delta_t
        )


def reparameterize(mu, logvar, sampling=True):
    # output dim: batch_size * dim
    if sampling:
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(*std.shape, 1).to(mu.device).normal_()
        return torch.unsqueeze(mu, 1) + eps * torch.unsqueeze(std, 1)

    return mu


class VIBPruner(PruningManager):
    def __init__(
        self, owner, mask_thresh=0, init_mag=9, init_var=0.01, kl_mult=1, masking=False, **kwargs
    ):
        super(VIBPruner, self).__init__(owner, **kwargs)
        self.epsilon = 1e-8

        # if masking=True, apply mask directly
        self.masking = masking
        self.init_var = init_var
        self.init_mag = init_mag

        self.mask_thresh = mask_thresh
        self.kl_mult = kl_mult

        self.prev_sparsity = 0

    def forward(self, x, bias):
        x_tiled, orig_shape = self.tiler(x)

        if not hasattr(self, "tiled_input_size"):
            self.tiled_input_size = x_tiled.shape

        if not hasattr(self, "post_z_mu"):
            post_z_mu = torch.FloatTensor(x_tiled.shape[:1]).to(x.device)
            post_z_mu = post_z_mu.normal_(1, self.init_var)
            self.post_z_mu = torch.nn.Parameter(post_z_mu)

        if not hasattr(self, "post_z_logD"):
            post_z_logD = torch.FloatTensor(x_tiled.shape[:1]).to(x.device)
            post_z_logD = post_z_logD.normal_(-self.init_mag, self.init_var)
            self.post_z_logD = torch.nn.Parameter(post_z_logD)

        mask = self.get_mask()
        if len(mask.shape) < 2:
            mask = mask.unsqueeze(1)

        x = x_tiled * mask
        x = self.tiler.inverse(x, orig_shape)

        return x, bias

    def get_logalpha(self):
        return self.post_z_logD.data - torch.log(self.post_z_mu.data.pow(2) + self.epsilon)

    def get_mask(self):
        self.mask = self.get_mask_weighted(self.mask_thresh)

        if self.training:
            mask = reparameterize(self.post_z_mu, self.post_z_logD, sampling=True)
        else:
            mask = self.get_mask_weighted(self.mask_thresh)

        return mask

    def get_mask_weighted(self, threshold=0):
        logalpha = self.get_logalpha()
        mask = (logalpha < threshold).float() * self.post_z_mu.data
        return mask

    def get_regularizer_value(self):
        h_D = torch.exp(self.post_z_logD)
        h_mu = self.post_z_mu

        KLD = torch.sum(torch.log(1 + h_mu.pow(2) / (h_D + self.epsilon)))
        KLD *= np.prod(self.tiled_input_size[1:])

        kld = KLD * 0.5 * self.kl_mult
        return kld

    def get_sparsity_level(self):
        mask = self.get_mask_weighted(self.mask_thresh)
        num_zeros = np.float((mask == 0).sum().cpu().item())
        num_total = np.float(mask.numel())
        sparsity_level = num_zeros / num_total
        return sparsity_level


MethodMap2 = namedtuple("MethodMap2", ["value", "cls"])


class PruneMethods(ClassEnumOptions):
    zhu_gupta = MethodMap2(1, ZhuGuptaPruner)
    balanced_one_shot = MethodMap2(2, BalancedOneShotPruner)
    movement_pruning = MethodMap2(3, MovementPruner)
    vibnet = MethodMap2(4, VIBPruner)
    balanced_zhu_gupta = MethodMap2(5, BalancedZhuGuptaPruner)
    balanced_batch_shaped = MethodMap2(6, BalancedBatchShapedPruner)
