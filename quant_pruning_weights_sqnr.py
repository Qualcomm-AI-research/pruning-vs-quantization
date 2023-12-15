# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os

from utils.moment_estimators import (
    EstimatorSampleKurtosis,
    EstimatorSamplePruningSQNR,
    EstimatorSampleQuantSQNR,
    TensorDataEvaluator,
)
from utils.utils import (
    model_weights_dict,
    pytorch_model_zoo_list,
    remove_pytorch_checkpoints,
)


def eval_pytorch_model_zoo_weights_sqnr(range_bitwidth=[8, 7, 6, 5, 4, 3, 2], tmp_dir="/tmp"):
    (models_list, names_list) = pytorch_model_zoo_list()
    print("Full model list:")
    print(*names_list, sep="\n")
    print(len(models_list))
    os.environ["TORCH_HOME"] = tmp_dir

    results_dir = os.getcwd() + "/plots_data/weights_sqnr/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    range_estimator = "mse"
    for i_model in range(0, len(models_list)):
        print("*" * 100)
        print(f"Processing model {names_list[i_model]}:")
        model = models_list[i_model](pretrained=True)
        model_name = names_list[i_model]
        print(model)

        weights_dict = model_weights_dict(model)

        for n_bits in range_bitwidth:
            print("bitwidth ", n_bits)
            compression_ratio = 1.0 - n_bits / 16.0
            fname_json = (
                results_dir + model_name + "_" + str(n_bits) + "_bits_" + range_estimator + ".json"
            )

            evaluator = TensorDataEvaluator()
            sqnr_estimators = [
                EstimatorSampleKurtosis(),
                EstimatorSampleQuantSQNR(n_bits=n_bits, range_estimator=range_estimator),
                EstimatorSamplePruningSQNR(compression_ratio=compression_ratio),
            ]

            evaluator.eval_moments(input_data=weights_dict, estimators=sqnr_estimators)
            evaluator.save(fname=fname_json)
        remove_pytorch_checkpoints()


if __name__ == "__main__":
    eval_pytorch_model_zoo_weights_sqnr()
