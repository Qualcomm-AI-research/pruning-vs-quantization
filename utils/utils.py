#!/usr/bin/env python
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import collections
import os
import random
from collections import namedtuple
from enum import Flag, auto
from functools import partial
from typing import Tuple

import click
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.models as models


class DotDict(dict):
    """
    A dictionary that allows attribute-style access.
    Examples
    --------
    >>> config = DotDict(a=None)
    >>> config.a = 42
    >>> config.b = 'egg'
    >>> config  # can be used as dict
    {'a': 42, 'b': 'egg'}
    """

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, key):
        self.__delitem__(key)

    def __getattr__(self, key):
        if key in self:
            return self.__getitem__(key)
        raise AttributeError(f"DotDict instance has no key '{key}' ({self.keys()})")


def relu(x):
    x = np.array(x)
    return x * (x > 0)


def get_all_layer_names(model, subtypes=None):
    if subtypes is None:
        return [name for name, module in model.named_modules()][1:]
    return [name for name, module in model.named_modules() if isinstance(module, subtypes)]


def get_layer_name_to_module_dict(model):
    return {name: module for name, module in model.named_modules() if name}


def get_module_to_layer_name_dict(model):
    modules_to_names = collections.OrderedDict()
    for name, module in model.named_modules():
        modules_to_names[module] = name
    return modules_to_names


def get_layer_name(model, layer):
    for name, module in model.named_modules():
        if module == layer:
            return name
    return None


def get_layer_by_name(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    return None


def create_conv_layer_list(cls, model: nn.Module) -> list:
    """
    Function finds all prunable layers in the provided model

    Parameters
    ----------
    cls: SVD class
    model : torch.nn.Module
    A pytorch model.

    Returns
    -------
    conv_layer_list : list
    List of all prunable layers in the given model.

    """
    conv_layer_list = []

    def fill_list(mod):
        if isinstance(mod, tuple(cls.supported_layer_types)):
            conv_layer_list.append(mod)

    model.apply(fill_list)
    return conv_layer_list


def create_linear_layer_list(cls, model: nn.Module) -> list:
    """
    Function finds all prunable layers in the provided model

    Parameters
    ----------
    model : torch.nn.Module
        A pytorch model.

    Returns
    -------
    conv_layer_list : list
        List of all prunable layers in the given model.

    """
    conv_layer_list = []

    def fill_list(mod):
        if isinstance(mod, tuple(cls.supported_layer_types)):
            conv_layer_list.append(mod)

    model.apply(fill_list)
    return conv_layer_list


def to_numpy(tensor):
    """
    Helper function that turns the given tensor into a numpy array

    Parameters
    ----------
    tensor : torch.Tensor

    Returns
    -------
    tensor : float or np.array

    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "is_cuda"):
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()
    if hasattr(tensor, "detach"):
        return tensor.detach().numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()

    return np.array(tensor)


def set_module_attr(model, layer_name, value):
    split = layer_name.split(".")

    this_module = model
    for mod_name in split[:-1]:
        if mod_name.isdigit():
            this_module = this_module[int(mod_name)]
        else:
            this_module = getattr(this_module, mod_name)

    last_mod_name = split[-1]
    if last_mod_name.isdigit():
        this_module[int(last_mod_name)] = value
    else:
        setattr(this_module, last_mod_name, value)


def search_for_zero_planes(model: torch.nn.Module):
    """If list of modules to winnow is empty to start with, search through all modules to check
    if any
    planes have been zeroed out. Update self._list_of_modules_to_winnow with any findings.
    :param model: torch model to search through modules for zeroed parameters
    """

    list_of_modules_to_winnow = []
    for _, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.modules.conv.Conv2d)):
            in_channels_to_winnow = _assess_weight_and_bias(module.weight, module.bias)
            if in_channels_to_winnow:
                list_of_modules_to_winnow.append((module, in_channels_to_winnow))
    return list_of_modules_to_winnow


def _assess_weight_and_bias(weight: torch.nn.Parameter, _bias: torch.nn.Parameter):
    """4-dim weights [CH-out, CH-in, H, W] and 1-dim bias [CH-out]"""
    if len(weight.shape) > 2:
        input_channels_to_ignore = (weight.sum((0, 2, 3)) == 0).nonzero().squeeze().tolist()
    else:
        input_channels_to_ignore = (weight.sum(0) == 0).nonzero().squeeze().tolist()

    if type(input_channels_to_ignore) != list:
        input_channels_to_ignore = [input_channels_to_ignore]

    return input_channels_to_ignore


def seed_all(seed: int = 1029, deterministic: bool = False):
    """
    This is our attempt to make experiments reproducible by seeding all known RNGs and setting
    appropriate torch directives.
    For a general discussion of reproducibility in Pytorch and CUDA and a documentation of the
    options we are using see, e.g.,
    https://pytorch.org/docs/1.7.1/notes/randomness.html
    https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

    As of today (July 2021), even after seeding and setting some directives,
    there remain unfortunate contradictions:
    1. CUDNN
    - having CUDNN enabled leads to
      - non-determinism in Pytorch when using the GPU, cf. MORPH-10999.
    - having CUDNN disabled leads to
      - most regression tests in Qrunchy failing, cf. MORPH-11103
      - significantly increased execution time in some cases
      - performance degradation in some cases
    2. torch.set_deterministic(d)
    - setting d = True leads to errors for Pytorch algorithms that do not (yet) have a deterministic
      counterpart, e.g., the layer `adaptive_avg_pool2d_backward_cuda` in vgg16__torchvision.

    Thus, we leave the choice of enforcing determinism by disabling CUDNN and non-deterministic
    algorithms to the user. To keep it simple, we only have one switch for both.
    This situation could be re-evaluated upon updates of Pytorch, CUDA, CUDNN.
    """

    assert isinstance(seed, int), f"RNG seed must be an integer ({seed})"
    assert seed >= 0, f"RNG seed must be a positive integer ({seed})"

    # Builtin RNGs
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Numpy RNG
    np.random.seed(seed)

    # CUDNN determinism (setting those has not lead to errors so far)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Torch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Problematic settings, see docstring. Precaution: We do not mutate unless asked to do so
    if deterministic is True:
        torch.backends.cudnn.enabled = False

        torch.set_deterministic(True)  # Use torch.use_deterministic_algorithms(True) in torch 1.8.1
        # When using torch.set_deterministic(True), it is advised by Pytorch to set the
        # CUBLAS_WORKSPACE_CONFIG variable as follows, see
        # https://pytorch.org/docs/1.7.1/notes/randomness.html#avoiding-nondeterministic-algorithms
        # and the link to the CUDA homepage on that website.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def assert_allclose(actual, desired, *args, **kwargs):
    """A more beautiful version of torch.all_close."""
    np.testing.assert_allclose(to_numpy(actual), to_numpy(desired), *args, **kwargs)


def count_params(module):
    return len(nn.utils.parameters_to_vector(module.parameters()))


class StopForwardException(Exception):
    """Used to throw and catch an exception to stop traversing the graph."""

    pass


class StopForwardHook:
    def __call__(self, module, *args):
        raise StopForwardException


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class CosineTempDecay:
    def __init__(self, t_max, temp_range=(20.0, 2.0), rel_decay_start=0):
        self.t_max = t_max
        self.start_temp, self.end_temp = temp_range
        self.decay_start = rel_decay_start * t_max

    def __call__(self, t):
        if t < self.decay_start:
            return self.start_temp

        rel_t = (t - self.decay_start) / (self.t_max - self.decay_start)
        return self.end_temp + 0.5 * (self.start_temp - self.end_temp) * (1 + np.cos(rel_t * np.pi))


class BaseEnumOptions(Flag):
    def __str__(self):
        return self.name

    @classmethod
    def list_names(cls):
        return [m.name for m in cls]


class ClassEnumOptions(BaseEnumOptions):
    @property
    def cls(self):
        return self.value.cls

    def __call__(self, *args, **kwargs):
        return self.value.cls(*args, **kwargs)


MethodMap = partial(namedtuple("MethodMap", ["value", "cls"]), auto())


def split_dict(src: dict, include=(), remove_prefix: str = ""):
    """
    Splits dictionary into a DotDict and a remainder.
    The arguments to be placed in the first DotDict are those listed in `include`.
    Parameters
    ----------
    src: dict
        The source dictionary.
    include:
        List of keys to be returned in the first DotDict.
    remove_suffix:
        remove prefix from key
    """
    result = DotDict()

    for arg in include:
        if remove_prefix:
            key = arg.replace(f"{remove_prefix}_", "", 1)
        else:
            key = arg
        result[key] = src[arg]
    remainder = {key: val for key, val in src.items() if key not in include}
    return result, remainder


class ClickEnumOption(click.Choice):
    """
    Adjusted click.Choice type for BaseOption which is based on Enum
    """

    def __init__(self, enum_options, case_sensitive=True):
        assert issubclass(enum_options, BaseEnumOptions)
        self.base_option = enum_options
        super().__init__(self.base_option.list_names(), case_sensitive)

    def convert(self, value, param, ctx):
        # Exact match
        if value in self.choices:
            return self.base_option[value]

        # Match through normalization and case sensitivity
        # first do token_normalize_func, then lowercase
        # preserve original `value` to produce an accurate message in
        # `self.fail`
        normed_value = value
        normed_choices = self.choices

        if ctx is not None and ctx.token_normalize_func is not None:
            normed_value = ctx.token_normalize_func(value)
            normed_choices = [ctx.token_normalize_func(choice) for choice in self.choices]

        if not self.case_sensitive:
            normed_value = normed_value.lower()
            normed_choices = [choice.lower() for choice in normed_choices]

        if normed_value in normed_choices:
            return self.base_option[normed_value]

        self.fail(
            "invalid choice: %s. (choose from %s)" % (value, ", ".join(self.choices)), param, ctx
        )


def pytorch_model_zoo_list():
    list_models = []
    list_names = []
    # classification models
    list_names_classification = [
        "alexnet",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2",
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19_bn",
        "vgg19",
        "squeezenet1_0",
        "squeezenet1_1",
        "inception_v3",
        "densenet121",
        "densenet169",
        "densenet201",
        "densenet161",
        "googlenet",
        "mobilenet_v2",
        "mobilenet_v3_large",
        "mobilenet_v3_small",
        "mnasnet0_5",
        "mnasnet1_0",
        "shufflenet_v2_x0_5",
        "shufflenet_v2_x1_0",
    ]
    for name in list_names_classification:
        model = getattr(models, name)
        list_models.append(model)
        list_names.append(name)
    # detection models
    list_names_detection = [
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "maskrcnn_resnet50_fpn",
        "keypointrcnn_resnet50_fpn",
        "retinanet_resnet50_fpn",
        "ssd300_vgg16",
        "ssdlite320_mobilenet_v3_large",
    ]
    for name in list_names_detection:
        model = getattr(models.detection, name)
        list_models.append(model)
        list_names.append(name)
    list_names_segmentation = ["lraspp_mobilenet_v3_large"]
    for name in list_names_segmentation:
        model = getattr(models.segmentation, name)
        list_models.append(model)
        list_names.append(name)
    list_names_video = ["r3d_18", "mc3_18", "r2plus1d_18"]
    for name in list_names_video:
        model = getattr(models.video, name)
        list_models.append(model)
        list_names.append(name)

    return (list_models, list_names)


def model_weights_dict(model, fp16_conversion=True, dev="cuda"):
    res_dict = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            res_dict[n] = m.weight.to(device=dev)
        elif isinstance(m, torch.nn.Linear):
            res_dict[n] = m.weight.to(device=dev)

    if fp16_conversion:
        res_dict_half = {k: v.half() for k, v in res_dict.items()}
        return res_dict_half
    else:
        return res_dict


def remove_pytorch_checkpoints(checkpoints_dir="/tmp/hub/checkpoints/"):
    for f in os.listdir(checkpoints_dir):
        os.remove(os.path.join(checkpoints_dir, f))


def load_model(model_name, fp16_conversion=True):
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        include = ["weight"]
        exclude = ["bn", "norm"]
        repr_layers = [
            "features.8.conv.0.0",
            "features.8.conv.1.0",
            "features.11.conv.0.0",
            "features.11.conv.1.0",
        ]
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        include = ["weight"]
        exclude = ["bn", "norm"]
        repr_layers = [
            "layer1.0.conv2",
            "layer1.2.conv3",
            "layer2.0.conv2",
            "layer2.1.conv1",
            "layer2.2.conv2",
            "layer3.0.conv1",
            "layer3.0.conv2",
            "layer3.4.conv3",
            "layer4.1.conv1",
            "layer4.2.conv2",
        ]
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        include = ["weight"]
        exclude = ["bn", "norm"]
        repr_layers = [
            "layer1.0.conv2",
            "layer1.1.conv1",
            "layer2.0.downsample.0",
            "layer3.0.downsample.0",
        ]
    elif model_name == "vit":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        include = ["weight"]
        exclude = ["bn", "norm"]
        repr_layers = [
            "blocks.2.attn.qkv",
            "blocks.2.attn.proj",
            "blocks.2.mlp.fc1",
            "blocks.2.mlp.fc2",
            "blocks.11.attn.qkv",
            "blocks.11.attn.proj",
            "blocks.11.mlp.fc1",
            "blocks.11.mlp.fc2",
        ]
    elif model_name == "efficientnet":
        model = timm.create_model("tf_efficientnet_b0", pretrained=True)
        include = ["weight"]
        exclude = ["bn", "norm"]
        repr_layers = [
            "blocks.1.1.conv_pw",
            "blocks.1.1.conv_dw",
            "blocks.1.1.se.conv_reduce",
            "blocks.1.1.se.conv_expand",
            "blocks.1.1.conv_pwl",
            "blocks.6.0.conv_pw",
            "blocks.6.0.conv_dw",
            "blocks.6.0.se.conv_reduce",
            "blocks.6.0.se.conv_expand",
            "blocks.6.0.conv_pwl",
        ]
    else:
        raise NotImplementedError("")

    if fp16_conversion:
        weights_dict = {
            name: torch.tensor(param.half(), dtype=torch.float32)
            for name, param in model.named_parameters()
            if (all([i in name for i in include]) and not any([e in name for e in exclude]))
        }
    else:
        weights_dict = {name: param for name, param in model.named_parameters()}

    return model, weights_dict, repr_layers


def load_model_weights(model_name, fp16_conversion=True):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        include = ["weight"]
        exclude = ["bn", "norm"]
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        include = ["weight"]
        exclude = ["bn", "norm"]
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        include = ["weight"]
        exclude = ["bn", "norm"]
    elif model_name == "vit":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        include = ["weight"]
        exclude = ["bn", "norm"]
    elif model_name == "efficientnet":
        model = timm.create_model("tf_efficientnet_b0", pretrained=True)
        include = ["weight"]
        exclude = ["bn", "norm"]
    else:
        raise NotImplementedError("")

    if fp16_conversion:
        dict = {
            name: torch.tensor(param.half(), dtype=torch.float32)
            for name, param in model.named_parameters()
            if (all([i in name for i in include]) and not any([e in name for e in exclude]))
        }
    else:
        dict = {name: param for name, param in model.named_parameters()}
    return dict, model


def split_dict(src: dict, include=(), remove_prefix: str = ""):
    """
    Splits dictionary into a DotDict and a remainder.
    The arguments to be placed in the first DotDict are those listed in `include`.
    Parameters
    ----------
    src: dict
        The source dictionary.
    include:
        List of keys to be returned in the first DotDict.
    remove_suffix:
        remove prefix from key
    """
    result = DotDict()

    for arg in include:
        if remove_prefix:
            key = arg.replace(f"{remove_prefix}_", "", 1)
        else:
            key = arg
        result[key] = src[arg]
    remainder = {key: val for key, val in src.items() if key not in include}
    return result, remainder
