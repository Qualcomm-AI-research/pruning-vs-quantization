#!/usr/bin/env python
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os

import torch
from quantization.autoquant_utils import QuantizedActivationWrapper, quantize_sequential
from quantization.base_quantized_classes import FP32Acts, QuantizedActivation
from quantization.base_quantized_model import QuantizedModel
from torch import nn

from models.mobilenetv3_small import Block, MobileNetV3Small


class QuantizedSEModule(QuantizedActivation):
    def __init__(self, se_orig, **quant_params):
        super().__init__(**quant_params)

        assert len(se_orig.se) == 7
        self.se = quantize_sequential(se_orig.se, **quant_params)

    def forward(self, x):
        y = x * self.se(x)
        y = self.quantize_activations(y)
        return y


class QuantizedMobileBottleneck(QuantizedActivation):
    def __init__(self, mb_orig, **quant_params):
        super().__init__(**quant_params)

        conv_list = nn.Sequential(
            mb_orig.conv1,
            mb_orig.bn1,
            mb_orig.nolinear1,
            mb_orig.conv2,
            mb_orig.bn2,
            mb_orig.nolinear2,
            mb_orig.conv3,
            mb_orig.bn3,
        )
        self.conv = quantize_sequential(conv_list, **quant_params)
        self.se = mb_orig.se
        if len(mb_orig.shortcut) > 0:
            self.shortcut = quantize_sequential(mb_orig.shortcut, **quant_params)
        else:
            self.shortcut = nn.Identity()

        self.se = None
        if mb_orig.se:
            self.se = QuantizedSEModule(mb_orig.se, **quant_params)

        self.use_res_connect = mb_orig.stride == 1

    def forward(self, x):
        y = self.conv(x)
        if self.se:
            y = self.se(y)
        if self.use_res_connect:
            y = y + self.shortcut(x)
            return self.quantize_activations(y)
        else:
            return y


class QuantizedMobileNetV3Small(QuantizedModel):
    def __init__(self, base_model, input_size=(1, 3, 224, 224), quant_setup=None, **quant_params):
        super().__init__(input_size)
        ## Quantize parts
        specials = {Block: QuantizedMobileBottleneck}
        assert len(base_model.bneck) == 11
        features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.hs1,
            *base_model.bneck,
            base_model.conv2,
            base_model.bn2,
            base_model.hs2,
        )
        self.features = quantize_sequential(features, specials=specials, **quant_params)

        global_pool = nn.AdaptiveAvgPool2d(1)
        if quant_setup == "LSQ_paper":
            # Keep global_pool intact as we quantize the input the last layer
            self.global_pool = global_pool
        else:
            self.global_pool = QuantizedActivationWrapper(
                global_pool,
                tie_activation_quantizers=True,
                input_quantizer=self.features[-1].activation_quantizer,
                **quant_params,
            )

        classifier = nn.Sequential(
            base_model.linear3, base_model.bn3, base_model.hs3, base_model.linear4
        )
        self.classifier = quantize_sequential(classifier, **quant_params)

        ## setups
        if quant_setup == "FP_logits":
            print("Do not quantize output of FC layer")
            # no activation quantization of logits:
            self.classifier[-1].activation_quantizer = FP32Acts()

        elif quant_setup == "LSQ":
            print("Set quantization to LSQ (first + last layer in 8 bits)")
            # Weights of the first layer
            self.features[0].weight_quantizer.quantizer.n_bits = 8
            # The quantizer of the last conv_layer layer (input to global)
            self.classifier[0].activation_quantizer.quantizer.n_bits = 8
            # Weights of the last layer
            self.classifier[-1].weight_quantizer.quantizer.n_bits = 8
            # no activation quantization of logits
            self.classifier[-1].activation_quantizer = FP32Acts()

        elif quant_setup == "LSQ_paper":
            # Weights of the first layer
            self.features[0].activation_quantizer = FP32Acts()
            self.features[0].weight_quantizer.quantizer.n_bits = 8

            # Weights of the last layer
            self.classifier[-1].activation_quantizer.quantizer.n_bits = 8
            self.classifier[-1].weight_quantizer.quantizer.n_bits = 8

            # Set all QuantizedActivations to FP32
            for layer in [*self.features.modules(), *self.classifier.modules()]:
                if isinstance(layer, QuantizedActivation):
                    layer.activation_quantizer = FP32Acts()

        elif quant_setup is not None and quant_setup != "all":
            raise ValueError(f"Quantization setup '{quant_setup}' not supported for MobileNetV3")

    def forward(self, x):
        # features
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def mobilenetv3_small_100_quantized(pretrained=True, model_dir=None, load_type="fp32", **qparams):
    fp_model = MobileNetV3Small()
    if pretrained and load_type == "fp32":
        # Load model from pretrained FP32 weights
        assert os.path.exists(model_dir)
        print(f"Loading pretrained weights from {model_dir}")
        state_dict = torch.load(model_dir)["state_dict"]

        # Clean state_dict
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module"):
                prefix, layer_name = key.split(".", 1)
                clean_state_dict[layer_name] = value

        del state_dict

        fp_model.load_state_dict(clean_state_dict, strict=True)
        quant_model = QuantizedMobileNetV3Small(fp_model, **qparams)
    elif load_type == "quantized":
        # Load pretrained QuantizedModel
        print(f"Loading pretrained quantized model from {model_dir}")
        state_dict = torch.load(model_dir)
        quant_model = QuantizedMobileNetV3Small(fp_model, **qparams)
        quant_model.load_state_dict(state_dict)

    return quant_model
