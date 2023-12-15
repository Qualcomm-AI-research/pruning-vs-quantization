# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import click
import numpy as np
import torch

from quantization.ptq_sqnr_bounds import compute_quant_sqnr_bounds_data_opt
from utils import get_layer_by_name
from utils.utils import load_model_weights


@click.command()
@click.option("--model-name", type=str, default="resnet50")
@click.option("--layer-name", type=str, default="layer4.1.conv1")
@click.option("--batch-size", type=int, default=64)
@click.option("--num-samples-total", type=int, default=64)
@click.option("--image-net-dir", type=str)
@click.option("--num-output-channels", type=int, default=-1)
@click.option("--opt-dim", type=int, default=-1)
@click.option("--n-bits", type=int, default=4)
@click.option("--range-search-method", type=str, default="mse")
@click.option("--verbose-solvers", type=bool, is_flag=True, default=False)
def quant_bounds_data_opt(
    model_name,
    layer_name,
    batch_size,
    num_samples_total,
    image_net_dir,
    num_output_channels,
    opt_dim,
    n_bits,
    range_search_method,
    verbose_solvers,
):
    if opt_dim > 0:
        tensors_dict, model = load_model_weights(model_name=model_name, fp16_conversion=True)
        # calculate the number of chunks
        layer = get_layer_by_name(model, layer_name)
        print(layer.weight.shape)

        if isinstance(layer, torch.nn.Conv2d) and layer.groups == 1:
            dim = layer.in_channels * layer.weight.shape[2] * layer.weight.shape[3]
            n_chunks = int(np.ceil(dim / opt_dim))
        elif isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:
            n_chunks = 1
        elif isinstance(layer, torch.nn.Linear):
            n_chunks = int(np.ceil(layer.in_features / opt_dim))
    else:
        n_chunks = 1

    rng_chunks = range(0, n_chunks)
    for i_chunk in rng_chunks:
        compute_quant_sqnr_bounds_data_opt(
            model_name,
            layer_name,
            batch_size,
            num_samples_total,
            image_net_dir,
            n_chunks,
            num_output_channels,
            n_bits,
            range_search_method,
            verbose_solvers,
            i_chunk,
        )


@click.group()
def cmds():
    pass


cmds.add_command(quant_bounds_data_opt)

if __name__ == "__main__":
    cmds()
