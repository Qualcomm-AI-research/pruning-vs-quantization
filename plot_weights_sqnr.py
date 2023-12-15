# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from utils import pytorch_model_zoo_list

results_dir = os.getcwd() + "/plots_data/weights_sqnr/"
fig, ax = plt.subplots(1, 1)

cmap = {}

palette = ["#dd4a4c", "#f98e52", "#fed481", "yellowgreen", "#d6ee9b", "#86cfa5", "#3d95b8"]
sparsity = {i: 1.0 - i / 16.0 for i in range(2, 8 + 1)}

for idx, c in enumerate(palette):
    cmap[idx + 2] = c
bits2color = cmap

colors_used = {k: False for k in bits2color.keys()}
_, list_model_names = pytorch_model_zoo_list()

lim = 0.0
x_data = []
y_data = []
alpha = 0.05

for n_bits in [8, 7, 6, 5, 4, 3, 2]:
    for model_name in list_model_names:
        range_estimator = "mse"
        fname_json = results_dir + model_name + "_" + str(n_bits) + "_bits_" + range_estimator + ".json"

        json_file = open(fname_json, "r")
        json_dict = json.load(json_file)

        x_title_str = "quant_sqnr"
        y_title_str = "pruning_sqnr"

        for layer_name, moments_dict in json_dict.items():
            x = moments_dict[x_title_str]
            y = moments_dict[y_title_str]
            k = moments_dict["sample_kurtosis"]

            if not math.isnan(x) and not math.isnan(y) and not math.isinf(x) and not math.isinf(y):
                lim = max(lim, max(x, y))
                x_data.append(x)
                y_data.append(y)
            color = bits2color[n_bits]

            if not colors_used[n_bits]:
                label = (
                    "INT" + str(n_bits) + " vs " + str((1.0 - n_bits / 16.0) * 100.0) + "% sparsity"
                )
                ax.scatter(-1.0 * x, -1.0 * y, color=color, alpha=1.0, label=label)
                ax.scatter(x, y, color=color, alpha=alpha)
                colors_used[n_bits] = True
            else:
                ax.scatter(x, y, color=color, alpha=alpha)

lim = 60.0
sep_line = np.linspace(0, lim, 1000)

ax.plot(sep_line, sep_line, linestyle="--", color="darkgrey")
ax.set_xlabel("Quantization SNR", fontsize=14)
ax.set_ylabel("Pruning SNR", fontsize=14)
ax.set_xlim([0.0, lim])
ax.set_ylim([0.0, lim])
ax.set_title("PyTorch model zoo weights (FP16)", fontsize=14)

ax.legend(fontsize=14)

max_sqnr = 40.0
sep_line = np.linspace(0, max_sqnr, 1000)
plt.plot(sep_line, sep_line, linestyle="dotted", color="gray")

cmap = {}

fig = plt.gcf()
fig.set_size_inches(5 * 1.5, 5 * 1.5)
plt.show()

plt.savefig("pytorch_model_zoo_sqnr.png")
