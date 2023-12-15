# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import json

import numpy as np
import torch
from pruning.pruning_utils import TopKMaskStraightThrough
from quantization.quantizers.uniform_quantizers import SymmetricUniformQuantizer
from quantization.range_estimators import estimate_range_line_search
from scipy.stats import kurtosis

from utils.utils import to_numpy


class MomentEstimator:
    def __init__(self, *args, **kwargs):
        return

    def estimate(self, **kwargs):
        return


class EstimatorSampleKurtosis(MomentEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def estimate(self, **kwargs):
        moments_dict = kwargs["moments_dict"]
        arr_flat = to_numpy(kwargs["input_tensor"]).flatten()
        k = kurtosis(arr_flat, axis=0, fisher=True, bias=True)
        moments_dict["sample_kurtosis"] = float(k)


class EstimatorSamplePruningSQNR(MomentEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compression_ratio = kwargs["compression_ratio"]
        self.n_bits = 16.0 * (1 - self.compression_ratio)

    def estimate_sqnr(self, t, sparsity):
        ste = TopKMaskStraightThrough()
        t_s = ste.forward(None, torch.abs(t), sparsity) * t
        mse = torch.mean((t - t_s) ** 2)
        tensor_norm = torch.mean(t**2)
        if mse.item() > 0.0:
            pruning_sqnr = 10 * np.log10(tensor_norm.item() / mse.item())
        else:
            pruning_sqnr = np.Inf
        return (mse, pruning_sqnr)

    def estimate(self, **kwargs):
        moments_dict = kwargs["moments_dict"]
        quant_sparsity = moments_dict["quant_sparsity"]
        t = kwargs["input_tensor"].flatten()
        sparsity = self.compression_ratio
        (mse, pruning_sqnr) = self.estimate_sqnr(t, sparsity)
        keep_ratio = (1.0 - quant_sparsity) * self.n_bits / 16.0
        (mse_corrected, pruning_sqnr_corrected) = self.estimate_sqnr(t, 1.0 - keep_ratio)
        # pruning SQNR for compression ratio equal to n_bits / 16 (assuming no sparsity in quantized values)
        moments_dict["pruning_sqnr"] = float(pruning_sqnr)
        moments_dict["pruning_mse"] = float(mse)
        # pruning SQNR for compression ratio equal to that of quantization with natural sparsity taken into account
        moments_dict["pruning_sqnr_corrected"] = float(pruning_sqnr_corrected)
        print(f"Pruning SQNR: {float(pruning_sqnr_corrected)}")
        moments_dict["pruning_mse_corrected"] = float(mse_corrected)


class EstimatorSampleQuantSQNR(MomentEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bits = kwargs["n_bits"]
        self.range_estimator = kwargs["range_estimator"]

    def estimate(self, **kwargs):
        moments_dict = kwargs["moments_dict"]
        t = kwargs["input_tensor"].flatten()

        quant = SymmetricUniformQuantizer(n_bits=self.n_bits)
        if self.range_estimator == "minmax":
            quant_range_min = t.min()
            quant_range_max = t.max()
        elif self.range_estimator == "mse":
            (quant_range_min, quant_range_max) = estimate_range_line_search(
                t.clone(), quant, num_candidates=100
            )

        quant.set_quant_range(quant_range_min, quant_range_max)
        t_q = quant.forward(t)
        nz = torch.sum(t_q == 0)
        sparsity = float(nz / t.numel())

        mse_quant = ((t - t_q) ** 2).mean().item()
        tensor_norm = torch.mean(t**2).item()
        sqnr_quant = float(10 * np.log10(tensor_norm / mse_quant))
        moments_dict["quant_sqnr"] = sqnr_quant
        print(f"Quant SQNR: {sqnr_quant}")
        moments_dict["quant_mse"] = float(mse_quant)
        moments_dict["quant_sparsity"] = float(sparsity)


class TensorDataEvaluator:
    def eval_moments(self, **kwargs):
        input_data, estimators = kwargs["input_data"], kwargs["estimators"]
        self.dict_outputs = {}
        for name, input_tensor in input_data.items():
            print(f"{name}: weight tensor of shape {tuple(input_tensor.shape)}")
            self.dict_outputs[name] = {}
            for estim in estimators:
                estim.estimate(input_tensor=input_tensor, moments_dict=self.dict_outputs[name])
        return self.dict_outputs

    def save(self, fname):
        if hasattr(self, "dict_outputs"):
            a_file = open(fname, "w")
            json.dump(self.dict_outputs, a_file)
            a_file.close()
