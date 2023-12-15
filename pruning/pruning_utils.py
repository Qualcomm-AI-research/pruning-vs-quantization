# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch

from utils import ClassEnumOptions, MethodMap


def topk_func(x, k, reverse=False):
    return (torch.topk(x, k, dim=1, largest=reverse)[0] ** 2).sum(1)


class Metrics(ClassEnumOptions):
    l2 = MethodMap(lambda x: (x**2).sum(1))
    l1 = MethodMap(lambda x: abs(x).sum(1))
    max = MethodMap(lambda x: abs(x).max(1)[0])
    top2 = MethodMap(lambda x: topk_func(x, 2))
    bottom2 = MethodMap(lambda x: topk_func(x, 2, reverse=True))


class TopKMaskStraightThrough:
    # following the implementation of movement pruning authors in order to reproduce their results
    # https://github.com/huggingface/transformers/blob/master/examples/movement-pruning/emmental/modules/binarizer.py
    @staticmethod
    def forward(ctx, inputs: torch.tensor, sparsity: float):
        # assume flattened input
        assert inputs.ndim == 1
        mask = torch.ones_like(inputs)
        _, idx = inputs.sort(descending=False)
        j = int(sparsity * inputs.numel())
        mask[idx[:j]] = 0
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def sparsity_level_cubic(s_f, s_i, t, t_0, n, delta_t):
    if t < t_0:
        return 0.0
    return s_f + (s_i - s_f) * (1 - ((t - t_0) / (n * delta_t))) ** 3
