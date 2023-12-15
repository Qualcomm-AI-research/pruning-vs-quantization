# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import json
from functools import partial

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils import get_layer_by_name, load_model, load_model_weights, to_numpy
from utils.quad_prog import conv2qp, mlp2qp

from quantization.optimizers_sdp import sdp_mixed_integer_qp_bounds
from quantization.quantizers import SymmetricUniformQuantizer
from quantization.range_estimators import estimate_range_line_search


def quant_weights_minmax(W, n_bits):
    quant = SymmetricUniformQuantizer(n_bits=n_bits)
    quant_range_min = W.min().item()
    quant_range_max = W.max().item()

    print("min-max range : ", quant_range_min, quant_range_max)
    quant.set_quant_range(quant_range_min, quant_range_max)
    W_int = quant.to_integer_forward(W)

    delta = quant.delta.item()
    return (W_int, delta, quant, W.min(), W.max())


def quant_weights_mse(W, n_bits):
    quant = SymmetricUniformQuantizer(n_bits=n_bits)
    (mse_range_min, mse_range_max) = estimate_range_line_search(W, quant)
    quant.set_quant_range(mse_range_min, mse_range_max)
    W_int = quant.to_integer_forward(W)

    delta = quant.delta.item()
    return (W_int, delta, quant, mse_range_min, mse_range_max)


def compute_quant_sqnr_bounds(
    layer, acts_inp_tensor, i_out_ch, chunk_idx_f, chunk_idx_l, delta, quant, verbose_solver
):
    acts_inp_tensor = torch.Tensor(acts_inp_tensor)
    # naive rounding
    if isinstance(layer, torch.nn.Conv2d) and layer.groups == 1:
        _, C_in, Y, X = acts_inp_tensor.shape
        W = layer.weight[:, chunk_idx_f:chunk_idx_l, :, :]
        weights_matr = W.view(layer.weight.size(0), -1).t()
        weights_matr_ch = weights_matr[:, i_out_ch]
    elif isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:
        _, __, Y, X = acts_inp_tensor.shape
        sh = layer.weight.shape
        W = layer.weight[i_out_ch, :, :, :].reshape(1, 1, sh[2], sh[3])
        weights_matr_ch = W.flatten()
    elif isinstance(layer, torch.nn.Linear):
        weights_matr_ch = layer.weight[i_out_ch, chunk_idx_f:chunk_idx_l]

    # compute the quadratic form terms and the initial objective
    if isinstance(layer, torch.nn.Conv2d):
        (G, r) = conv2qp(layer, acts_inp_tensor, i_out_ch, chunk_idx_f, chunk_idx_l)
    elif isinstance(layer, torch.nn.Linear):
        (G, r) = mlp2qp(layer, acts_inp_tensor, i_out_ch, chunk_idx_f, chunk_idx_l)

    # SDP solver
    (sdp_lb, sdp_ub, sdp_x) = sdp_mixed_integer_qp_bounds(
        to_numpy(G),
        to_numpy(weights_matr_ch),
        r,
        delta,
        verbose_solver,
        quant.int_min,
        quant.int_max,
    )

    n = G.shape[0]
    int_lower_bound = np.ones((n,)) * quant.int_min
    int_upper_bound = np.ones((n,)) * quant.int_max

    sdp_gap = (sdp_ub - sdp_lb) / abs(sdp_lb)

    # check the SDP solution range
    sol_min, sol_max = np.min(sdp_x), np.max(sdp_x)
    print("Solution range: ", sol_min, sol_max)
    print("Quant INT range: ", int_lower_bound[0], int_upper_bound[0])

    print("SDP Optimal value is between %.10f and %.10f\n" % (sdp_lb, sdp_ub))
    print("SDP gap is %.10f\n" % sdp_gap)
    print("SDP Optimal value (relative) is between %.10f and %.10f\n" % (sdp_lb / r, sdp_ub / r))

    res_dict = {}
    res_dict["sdp_lb"] = sdp_lb
    res_dict["sdp_ub"] = sdp_ub
    res_dict["int_min"] = sol_min
    res_dict["int_max"] = sol_max
    res_dict["zeros"] = int(np.sum(sdp_x == 0.0))
    res_dict["total_elements"] = int(sdp_x.size)
    res_dict["norm_sqr"] = r

    return res_dict


def sample_activations(model_name, layer_name, batch_size, num_samples_total, val_dir, cuda=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        sampler=None,
    )

    model, _, __ = load_model(model_name)
    repr_layers = [layer_name]
    model.eval()
    if cuda:
        model.cuda()

    print(model)
    for name in repr_layers:
        layer = get_layer_by_name(model, name)
        print(name, " : ", layer.weight.shape)

    inp_acts_dict = {}

    def save_activation(layer_name):
        def hook(model, input, output):
            key = layer_name + ":" + str(len(inp_acts_dict))
            print(key)
            inp_acts_dict[key] = [input[0].detach().cpu().numpy()]

        return hook

    num_batches = int(np.ceil(num_samples_total // batch_size))
    for name in repr_layers:
        print("*" * 80)
        inp_acts_dict = {}

        model, _, repr_layers = load_model(model_name)
        model.eval()
        if cuda:
            model.cuda()

        layer = get_layer_by_name(model, name)
        layer.register_forward_hook(save_activation(name))

        for i in range(0, num_batches):
            x, y = next(iter(val_loader))

            if cuda:
                x, y = x.cuda(), y.cuda()
            out = model(x)

    return inp_acts_dict


def compute_quant_sqnr_bounds_data_opt(
    model_name,
    layer_name,
    batch_size,
    num_samples_total,
    imagenet_dir,
    n_chunks,
    num_output_channels,
    n_bits,
    range_search_method,
    verbose_solvers,
    i_chunk,
):
    tensors_dict, model = load_model_weights(model_name=model_name, fp16_conversion=True)
    layer = get_layer_by_name(model, layer_name)

    if isinstance(layer, torch.nn.Linear):
        n_inp_channels = layer.in_features
    elif isinstance(layer, torch.nn.Conv2d):
        n_inp_channels = layer.in_channels

    arr_range_channels = np.array(range(0, n_inp_channels))
    arr_range_channels_split = np.array_split(arr_range_channels, n_chunks)
    chunk_idx_f = arr_range_channels_split[i_chunk][0]
    chunk_idx_l = arr_range_channels_split[i_chunk][-1] + 1

    acts_inp_tensor_list = []
    acts_dict = sample_activations(
        model_name, layer_name, batch_size, num_samples_total, imagenet_dir
    )

    if isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:
        assert n_chunks == 1

    i_batch = 0
    acts_inp_tensor = acts_dict[layer_name + ":" + str(i_batch)][0]
    if n_chunks == 1:
        acts_inp_tensor_list.append(acts_inp_tensor)
    else:
        if isinstance(layer, torch.nn.Conv2d) and layer.groups == 1:
            acts_inp_tensor_list.append(acts_inp_tensor[:, chunk_idx_f:chunk_idx_l, 0:, 0:])
        elif isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:
            acts_inp_tensor_list.append(acts_inp_tensor[:, 0:, 0:, 0:])
            assert n_chunks == 1
        elif isinstance(layer, torch.nn.Linear):
            acts_inp_tensor_list.append(acts_inp_tensor[:, :, chunk_idx_f:chunk_idx_l])

    acts_inp_tensor = acts_inp_tensor_list[0]
    W = tensors_dict[layer_name + ".weight"]
    if num_output_channels == -1:
        num_output_channels = W.shape[0]

    if n_chunks > 1:
        if isinstance(layer, torch.nn.Conv2d) and layer.groups == 1:
            W = W[:, chunk_idx_f:chunk_idx_l, :, :]
        elif isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:
            W = W[chunk_idx_f:chunk_idx_l, :, :, :]
        elif isinstance(layer, torch.nn.Linear):
            W = W[:, chunk_idx_f:chunk_idx_l]

    if isinstance(layer, torch.nn.Conv2d) and layer.groups == 1:
        forward_func = torch.nn.functional.conv2d
    elif isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:
        forward_func = partial(torch.nn.functional.conv2d, groups=1)
    elif isinstance(layer, torch.nn.Linear):
        forward_func = torch.nn.functional.linear

    print("layer name = ", layer_name)
    print("W shape: ", W.shape, ", total", W.numel())
    print("acts shape: ", acts_inp_tensor.shape, ", total", acts_inp_tensor.size)

    if range_search_method == "minmax":
        (W_int, delta, quant, range_min, range_max) = quant_weights_minmax(W, n_bits=n_bits)
    elif range_search_method == "MSE":
        (W_int, delta, quant, range_min, range_max) = quant_weights_mse(W, n_bits=n_bits)

    res_dict_acc = {}
    acts_inp_tensor = torch.Tensor(acts_inp_tensor).contiguous()

    for i_out_ch in range(0, num_output_channels):
        print("*" * 80)
        print("output channel #", i_out_ch)
        if isinstance(layer, torch.nn.Conv2d) and layer.groups == 1:
            _, C_in, h, w = W.shape
            W_single_channel = W[i_out_ch, :, :, :].reshape(1, C_in, h, w).contiguous()
            out_orig_single_ch = forward_func(acts_inp_tensor, W_single_channel)
        elif isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:
            C_in, _, h, w = W.shape
            W_single_channel = W[i_out_ch, :, :, :].reshape(1, 1, h, w).contiguous()
            sh = acts_inp_tensor.shape
            acts_single_channel = acts_inp_tensor[:, i_out_ch, :, :].reshape(sh[0], 1, sh[2], sh[3])
            out_orig_single_ch = forward_func(acts_single_channel, W_single_channel)
        elif isinstance(layer, torch.nn.Linear):
            C_out, C_in = W.shape
            W_single_channel = W[i_out_ch, :].reshape(1, C_in).contiguous()
            out_orig_single_ch = forward_func(acts_inp_tensor, W_single_channel)

        norm_channel = ((out_orig_single_ch) ** 2).mean().item()
        if norm_channel == 0.0:
            print("Skipping a dead channels...")
            continue

        if isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:
            res_dict = compute_quant_sqnr_bounds(
                layer,
                acts_single_channel,
                i_out_ch,
                chunk_idx_f,
                chunk_idx_l,
                delta,
                quant,
                verbose_solvers,
            )
        else:
            res_dict = compute_quant_sqnr_bounds(
                layer,
                acts_inp_tensor,
                i_out_ch,
                chunk_idx_f,
                chunk_idx_l,
                delta,
                quant,
                verbose_solvers,
            )

        if len(res_dict_acc) == 0:
            res_dict_acc = res_dict
        else:
            for key in res_dict.keys():
                if key == "int_min":
                    res_dict_acc[key] = min(res_dict_acc[key], res_dict[key])
                elif key == "int_max":
                    res_dict_acc[key] = max(res_dict_acc[key], res_dict[key])
                else:
                    res_dict_acc[key] += res_dict[key]
        print("Norm: %.2f" % (res_dict["norm_sqr"]))

    res_dict_acc["num_input_channels"] = int(chunk_idx_l - chunk_idx_f)
    res_dict_acc["num_output_channels"] = num_output_channels

    # The error lower bound is SQNR upper bound and vice versa
    SQNR_ub = -10.0 * np.log10(res_dict_acc["sdp_lb"] / res_dict_acc["norm_sqr"])
    SQNR_lb = -10.0 * np.log10(res_dict_acc["sdp_ub"] / res_dict_acc["norm_sqr"])
    print("-" * 80)
    print(f"layer {layer_name} final results:")
    print(f"SQNR lower bound: {SQNR_lb}, upper bound {SQNR_ub}")
