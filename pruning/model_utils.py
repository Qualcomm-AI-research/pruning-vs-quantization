# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import numpy as np
import torch
import transformers

from pruning.pruning_manager import PruningManager
from quantization.autoquant_utils import QuantLayerNorm
from quantization.base_quantized_classes import QuantizedActivation
from quantization.hijacker import QuantizationHijacker
from utils import get_layer_name


def remove_mobilenet_pruning_managers(model):
    def remove_check(mod):
        return (
            isinstance(mod, QuantizationHijacker)
            and hasattr(mod, "prune_manager")
            and mod.prune_manager is not None
            and hasattr(mod, "groups")
            and mod.groups > 1
        )

    for name, module in model.named_modules():
        if remove_check(module):
            print("Removing pruning manager from", name)
            module.prune_manager = None


def remove_vit_pruning_managers(model, skip_layernorm, skip_patch_embedding, skip_embeddings):
    model.head.prune_manager = None
    for name, module in model.named_modules():
        if skip_layernorm and isinstance(module, QuantLayerNorm):
            print("Removing pruning manager from", name)
            module.prune_manager = None
        if skip_embeddings:
            if "embed" in name and hasattr(module, "prune_manager"):
                module.prune_manager = None
    if skip_patch_embedding:
        print("Removing pruning manager from module.patch_embed.proj")
        model.patch_embed.proj.prune_manager = None


def remove_bert_pruning_managers(model, skip_embeddings, skip_value_layers, skip_dense_layers):
    for name, module in model.named_modules():
        if isinstance(module, (QuantLayerNorm, QuantizedActivation)):
            print("Removing pruning manager from", name)
            module.prune_manager = None
        if skip_embeddings:
            if "embed" in name and hasattr(module, "prune_manager"):
                module.prune_manager = None
        if skip_dense_layers:
            if "dense" in name and hasattr(module, "prune_manager"):
                module.prune_manager = None
        if skip_value_layers:
            if "value" in name and hasattr(module, "prune_manager"):
                module.prune_manager = None


def setup_pruning(config, model, task):
    if config.pruning_options.prune_method is not None:
        if config.architecture.architecture == "mobilenet_v2_cirqus_quantized":
            if config.quant.weight_quant:
                config.quant.quant_tricks = "134"
            if config.pruning_options.mnv2_prune_pw_only:
                remove_mobilenet_pruning_managers(model)
        elif "efficientnet" in config.architecture.architecture:
            if config.pruning_options.mnv2_prune_pw_only:
                remove_mobilenet_pruning_managers(model)
        elif config.architecture.architecture.startswith("vit"):
            remove_vit_pruning_managers(
                model,
                config.pruning_options.skip_layernorm,
                config.pruning_options.skip_patch_embedding,
                config.pruning_options.skip_embeddings,
            )
        elif config.architecture.architecture.startswith("bert"):
            remove_bert_pruning_managers(
                model,
                config.pruning_options.skip_embeddings,
                config.pruning_options.skip_value_layers,
                config.pruning_options.skip_dense_layers,
            )
        else:
            config.pruning_options.mnv2_prune_pw_only = False
            config.pruning_options.skip_layernorm = False
            config.pruning_options.skip_patch_embedding = False
            config.pruning_options.skip_embeddings = False

        if isinstance(task, transformers.trainer.Trainer):
            orig_loss = task.compute_loss

            def combined_loss(model, inputs):
                prune_loss = 0.0
                for _, module in model.named_modules():
                    if isinstance(module, PruningManager):
                        prune_loss += module.get_regularizer_value()
                return (
                    orig_loss(model, inputs) + config.pruning_options.prune_reg_lambda * prune_loss
                )

            task.compute_loss = combined_loss
        else:
            orig_loss = task._loss_fn

            def combined_loss(*args):
                prune_loss = 0.0
                for _, module in model.named_modules():
                    if isinstance(module, PruningManager):
                        prune_loss += module.get_regularizer_value()
                return orig_loss(*args) + config.pruning_options.prune_reg_lambda * prune_loss

            if config.pruning_options.prune_reg_lambda > 0:
                task._loss_fn = combined_loss


def load_pruned_model(config, model, dataloader):
    initialize_sparse_model(model, dataloader)
    load_sparse_model_state_dict(config, model, config.pruning_options.pruned_model_path)
    sparsify_model(model)

    print(model)

    total_pruned_ratio, pruned_ratios_str = get_model_mask_sparsity(model)
    print("Ratio of pruned weights in pre-trained network: {}".format(total_pruned_ratio))


def initialize_sparse_model(model, dataloader):
    model.train()
    for i, data in enumerate(dataloader):
        device = next(model.parameters()).device
        x = data[0].to(device=device)
        model(x)
        break
    model.eval()


def sparsify_model(model):
    def _sparsify_layer(layer):
        if any([isinstance(ch, PruningManager) for ch in layer.children()]) and isinstance(
            layer, QuantizationHijacker
        ):
            prune_manager = layer.prune_manager
            prune_manager.freeze_mask = True

    model.eval()
    model.apply(_sparsify_layer)

    return model


def load_sparse_model_state_dict(config, model, model_path):
    try:
        model.load_state_dict(torch.load(model_path))
    except RuntimeError as e:
        error_str = str(e)
        if "Unexpected key(s) in state_dict" in error_str:
            print(e)
            raise e
        elif (
            "Missing key(s) in state_dict" in error_str
            and config.quant.act_quant
            and config.quant.n_bits_act == 8
        ):
            model.load_state_dict(
                torch.load(config.pruning_options.pruned_model_path), strict=False
            )

    total_pruned_ratio, pruned_ratios_str = get_model_mask_sparsity(model)
    print("Ratio of pruned weights in pre-trained network: {}".format(total_pruned_ratio))


def get_model_mask_sparsity(model):
    sparsity_str = []
    total_params_per_layer = []
    zero_params_per_layer = []

    def _get_layer_sparsity(layer):
        if isinstance(layer, PruningManager):
            if not hasattr(layer, "mask"):
                print("Layer has no mask defined - skipping calculation.")
                return 0

            num_zeros = np.float((layer.mask == 0).sum().cpu().item())
            num_total = np.float(layer.mask.numel())
            sparsity_level = num_zeros / num_total

            sparsity_str.append(
                "{}% pruned for layer {}".format(sparsity_level * 100, get_layer_name(model, layer))
            )
            total_params_per_layer.append(num_total)
            zero_params_per_layer.append(num_zeros)

    model.apply(_get_layer_sparsity)
    zero_params_per_net = sum(zero_params_per_layer)
    params_per_net = sum(total_params_per_layer)

    try:
        total_sparsity = zero_params_per_net / params_per_net
    except ZeroDivisionError:
        return 0, ""

    return total_sparsity, " | ".join(sparsity_str)


def run_model_and_get_weights_sparsity(model):
    sparsity_str = []
    total_params_per_layer = []
    zero_params_per_layer = []

    def _get_layer_sparsity(layer):
        if any([isinstance(ch, PruningManager) for ch in layer.children()]) and isinstance(
            layer, QuantizationHijacker
        ):
            weight, bias = layer.get_params()
            num_zeros = np.float((weight == 0).sum().cpu().item())
            num_total = np.float(weight.numel())
            sparsity_level = num_zeros / num_total

            sparsity_str.append(
                "{}% pruned for layer {}".format(sparsity_level * 100, get_layer_name(model, layer))
            )
            total_params_per_layer.append(num_total)
            zero_params_per_layer.append(num_zeros)

    model.apply(_get_layer_sparsity)
    total_sparsity = sum(zero_params_per_layer) / sum(total_params_per_layer)

    return total_sparsity, " | ".join(sparsity_str)
