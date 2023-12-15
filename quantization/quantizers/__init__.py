#!/usr/bin/env python
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

from quantization.quantizers.base_quantizers import QuantizerBase
from quantization.quantizers.uniform_quantizers import (
    AsymmetricUniformQuantizer,
    SymmetricUniformQuantizer,
)

from utils import ClassEnumOptions, MethodMap


class QMethods(ClassEnumOptions):
    symmetric_uniform = MethodMap(SymmetricUniformQuantizer)
    asymmetric_uniform = MethodMap(AsymmetricUniformQuantizer)
