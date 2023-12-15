# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import torch


def weight_single_ch_vec(layer, acts_inp_tensor_smaller, chunk_idx_f, chunk_idx_l, i_out_ch):
    if isinstance(layer, torch.nn.Conv2d) and layer.groups == 1:
        _, C_in, Y, X = acts_inp_tensor_smaller.shape
        W = layer.weight[:, chunk_idx_f:chunk_idx_l, :, :]
        weights_matr = W.view(layer.weight.size(0), -1).t()
        weights_matr_ch = weights_matr[:, i_out_ch]
    elif isinstance(layer, torch.nn.Conv2d) and layer.groups > 1:
        _, __, Y, X = acts_inp_tensor_smaller.shape
        sh = layer.weight.shape
        W = layer.weight[i_out_ch, :, :, :].reshape(1, 1, sh[2], sh[3])
        weights_matr_ch = W.flatten()
    elif isinstance(layer, torch.nn.Linear):
        weights_matr_ch = layer.weight[i_out_ch, chunk_idx_f:chunk_idx_l]

    return weights_matr_ch


def conv_quad_const_term_chunked(acts_inp_tensor_bigger, acts_inp_tensor_smaller, W_single_channel):
    const_term = 0.0
    i_chunk = 0

    if isinstance(acts_inp_tensor_bigger, list):
        rng = acts_inp_tensor_bigger
    else:
        chunk_size = 8
        num_chunks = acts_inp_tensor_bigger.shape[0] // chunk_size
        rng = np.array_split(acts_inp_tensor_bigger, num_chunks, axis=0)

    _, C_in, Y, X = rng[0].shape
    h, w = W_single_channel.shape[2], W_single_channel.shape[3]
    Y_ = Y - h + 1
    X_ = X - w + 1

    for acts_inp_tensor_chunk in rng:
        # G matrix
        acts_inp_tensor_chunk = torch.Tensor(acts_inp_tensor_chunk)
        N_chunk = acts_inp_tensor_chunk.shape[0]
        acts_inp_matr_chunk = torch.nn.functional.unfold(acts_inp_tensor_chunk, (h, w)).transpose(
            1, 2
        )
        acts_inp_matr_chunk = acts_inp_matr_chunk.reshape(
            N_chunk * X_ * Y_, acts_inp_matr_chunk.shape[2]
        )

        A_np = acts_inp_matr_chunk.detach().cpu().numpy().astype("float64")
        if i_chunk == 0:
            G = A_np.T @ A_np
        else:
            G += A_np.T @ A_np

        acts_out_single_ch = torch.nn.functional.conv2d(acts_inp_tensor_chunk, W_single_channel)
        const_term += (acts_out_single_ch.flatten() ** 2).sum()
        i_chunk += 1

    acts_inp_tensor_smaller = torch.Tensor(acts_inp_tensor_smaller)

    return (G, const_term)


def mlp2qp_(
    layer, acts_inp_tensor_smaller, acts_inp_tensor_bigger, i_out_ch, num_input_channels, more_data
):
    if num_input_channels != -1:
        W = layer.weight[:, 0:num_input_channels]
    if more_data:
        (G, A, const_term) = mlp_quad_const_term_chunked(acts_inp_tensor_bigger, W, i_out_ch)
    else:
        sh = acts_inp_tensor_smaller.shape
        A = acts_inp_tensor_smaller.reshape(sh[0] * sh[1], sh[2])
        A_np = A.detach().cpu().numpy().astype("float64")
        G = A_np.T @ A_np
        W_single_ch = W[i_out_ch, :].reshape(1, num_input_channels)
        acts_out_single_ch = torch.nn.functional.linear(acts_inp_tensor_smaller, W_single_ch)
        r = (acts_out_single_ch.flatten() ** 2).sum().detach().numpy().item()
    return (G, A, r)


def mlp_quad_const_term_chunked(
    layer, acts_inp_tensor_bigger, i_out_ch, chunk_idx_f, chunk_idx_l, chunk_size=8
):
    W = layer.weight[:, chunk_idx_f:chunk_idx_l]
    const_term = 0.0
    if isinstance(acts_inp_tensor_bigger, list):
        rng = acts_inp_tensor_bigger
    else:
        num_chunks = acts_inp_tensor_bigger.shape[0] // chunk_size
        rng = np.array_split(acts_inp_tensor_bigger, num_chunks, axis=0)

    i_chunk = 0
    for acts_inp_tensor_chunk in rng:
        sh = acts_inp_tensor_chunk.shape
        A = acts_inp_tensor_chunk.reshape(sh[0] * sh[1], sh[2])
        A_np = A.astype("float64")
        if i_chunk == 0:
            G = A_np.T @ A_np
        else:
            G += A_np.T @ A_np
        W_single_ch = W[i_out_ch, :].reshape(1, chunk_idx_l - chunk_idx_f)
        acts_out_single_ch = torch.nn.functional.linear(
            torch.Tensor(acts_inp_tensor_chunk), W_single_ch
        )
        r = (acts_out_single_ch.flatten() ** 2).sum().detach().numpy().item()
        const_term += r
        i_chunk += 1

    return (G, const_term)


def conv2qp(layer, acts_inp_tensor_smaller, i_out_ch, chunk_idx_f, chunk_idx_l):
    if layer.groups == 1:
        _, C_in, Y, X = acts_inp_tensor_smaller.shape
        W = layer.weight
        W_single_ch = W[i_out_ch, chunk_idx_f:chunk_idx_l, :, :].reshape(
            1, C_in, layer.weight.shape[2], layer.weight.shape[3]
        )
    elif layer.groups > 1:
        sh = layer.weight.shape
        W_single_ch = layer.weight[i_out_ch, :, :, :].reshape(1, 1, sh[2], sh[3])
    (G, const_term) = conv_quad_const_term_chunked(
        acts_inp_tensor_smaller, acts_inp_tensor_smaller, W_single_ch
    )

    r = const_term.detach().numpy().item()
    return (G, r)


def mlp2qp(layer, acts_inp_tensor_smaller, i_out_ch, chunk_idx_f, chunk_idx_l):
    W = layer.weight[:, chunk_idx_f:chunk_idx_l]
    sh = acts_inp_tensor_smaller.shape
    A = acts_inp_tensor_smaller.reshape(sh[0] * sh[1], sh[2])
    A_np = A.detach().cpu().numpy().astype("float64")
    G = A_np.T @ A_np
    W_single_ch = W[i_out_ch, :].reshape(1, chunk_idx_l - chunk_idx_f)
    acts_out_single_ch = torch.nn.functional.linear(acts_inp_tensor_smaller, W_single_ch)
    r = (acts_out_single_ch.flatten() ** 2).sum().detach().numpy().item()

    return (G, r)
