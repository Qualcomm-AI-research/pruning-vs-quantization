#!/usr/bin/env python
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
import torch.serialization
from utils import StopForwardException, get_layer_by_name

from quantization.range_estimators import RangeEstimators


def pass_data_for_range_estimation(
    loader, model, act_quant, weight_quant, max_num_batches=20, cross_entropy_layer=None, inp_idx=0
):
    print("\nEstimate quantization ranges on training data")
    model.set_quant_state(weight_quant, act_quant)
    # Put model in eval such that BN EMA does not get updated
    model.eval()

    if cross_entropy_layer is not None:
        layer_xent = get_layer_by_name(model, cross_entropy_layer)
        if layer_xent:
            print('Set cross entropy estimator for layer "{}"'.format(cross_entropy_layer))
            act_quant_mgr = layer_xent.activation_quantizer
            act_quant_mgr.range_estimator = RangeEstimators.cross_entropy.cls(
                per_channel=act_quant_mgr.per_channel,
                quantizer=act_quant_mgr.quantizer,
                **act_quant_mgr.range_estim_params,
            )
        else:
            raise ValueError("Cross-entropy layer not found")

    batches = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for i, data in enumerate(loader):
            try:
                if isinstance(data, (tuple, list)):
                    x = data[inp_idx].to(device=device)
                    batches.append(x.data.cpu().numpy())
                    model(x)
                    print(f"proccesed step={i}")
                else:
                    x = {k: v.to(device=device) for k, v in data.items()}
                    model(**x)
                    print(f"proccesed step={i}")

                if i >= max_num_batches - 1 or not act_quant:
                    break
            except StopForwardException:
                pass
        return batches


def set_range_estimators(config, model):
    print("Make quantizers learnable")
    model.learn_ranges()

    if config.qat.grad_scaling:
        print("Activate gradient scaling")
        model.grad_scaling(True)

    # Ensure we have the desired quant state
    model.set_quant_state(config.quant.weight_quant, config.quant.act_quant)
