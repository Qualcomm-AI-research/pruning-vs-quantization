#!/usr/bin/env python
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

from utils import ClassEnumOptions, MethodMap

from models.efficientnet_lite_quantized import efficientnet_lite0_quantized
from models.mobilenet_v2_quantized import mobilenetv2_quantized
from models.mobilenetv3_small_quantized import mobilenetv3_small_100_quantized
from models.resnet_quantized import resnet18_quantized, resnet50_quantized


class QuantArchitectures(ClassEnumOptions):
    mobilenet_v2_quantized = MethodMap(mobilenetv2_quantized)
    resnet18_quantized = MethodMap(resnet18_quantized)
    resnet50_quantized = MethodMap(resnet50_quantized)
    efficientnet_lite0_quantized = MethodMap(efficientnet_lite0_quantized)
    mobilenetv3_small_100_quantized = MethodMap(mobilenetv3_small_100_quantized)
