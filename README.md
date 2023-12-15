# Pruning versus quantization: Which is better?
This repository contains the implementation and experiments for the paper presented in

**Andrey Kuzmin, Markus Nagel, Mart van Baalen, Arash Behboodi, Tijmen Blankevoort<sup>1</sup>", NeurIPS 2023.** 
[[ArXiv]](https://arxiv.org/pdf/2307.02973.pdf)

<sup>1</sup> Qualcomm AI Research (Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc.)

You can use this code to recreate the results in the paper.

## Abstract
Neural network pruning and quantization techniques are almost as old as neural networks themselves. However, to date only ad-hoc comparisons between the two have been published. In this paper, we set out to answer the question on which is better: neural network quantization or pruning? By answering this question, we hope to inform design decisions made on neural network hardware going forward. 
We provide an extensive comparison between the two techniques for compressing deep neural networks. 
First, we give an analytical comparison of expected quantization and pruning error for general data distributions.
Then, we provide lower bounds for the per-layer pruning and quantization error in trained networks, and compare these to empirical error after optimization.
Finally, we provide an extensive experimental comparison for training 9 large-scale models on 4 tasks.
Our results show that in most cases quantization outperforms pruning. Only in some scenarios with very high compression ratio, pruning might be beneficial from an accuracy standpoint.

## Method and Results

In this repository we share the code to reproduce the results for computing MSE error for pruning/quantization of PyTorch model zoo tensors, the code for obtaining the error lower bound for post-training pruning and quantization, and finally fine-tuning pipelines for QAT and pruning of 4 ImageNet pre-trained ImageNet models including  Resnet18, Resnet50, MobileNet-V2, EfficientNet-lite.

## How to install

### Using Docker

You can build and run the container with the following commands

```bash
docker build -f docker/Dockerfile --tag pruning_vs_quantization:latest .
docker run -ti pruning_vs_quantization:latest
```

You might need to mount some directories (ImageNet path) inside the docker. You can add the option `docker run  ... -v /local/path/to/imagenet_raw:/app/imagenet_raw`.

## Running experiments
### Pruning and quantization SQNR for PyTorch model zoo tensors:
The main run file to compute the SQNR (signal-to-noise ratio) for the weight tensors is 
`quant_pruning_weights_sqnr.py`. The script takes no input arguments and computes the SQNR for all the weight tensors model by model:
```bash
python quant_pruning_weights_sqnr.py
```
The output is then saved to the directory `/plots_data/weights_sqnr/`. The following command can be used to plot the results (generating the plot takes a few minutes):
```bash
python plot_weights_sqnr.py
```

<p align="center">
    <img src="img/pytorch_model_zoo_sqnr.png " width="425"/>
</p>

### Post-training quantization SQNR bounds:
The main run file to compute the upper bound for SQNR for output activations is `post_training_quant_acts_sqnr.py`. The formulation of Park, J., and S. Boyd. "A semidefinite programming method for integer convex quadratic minimization." is used. The lower bound is given by the baseline method (the SDP solver from the same paper). 
```bash
python post_training_quant_acts_sqnr.py quant-bounds-data-opt --model-name mobilenet_v2 --range-search-method MSE --num-output-channels 8 --layer-name features.8.conv.0.0 --n-bits 4 --opt-dim 32 --verbose-solvers --image-net-dir </PATH/TO/IMAGENET>
```
The command above gives an example of computing SQNR for MobileNet-V2, a subset of the layer `features.8.conv.0.0` for 8 output channels (use `--num-output-channels -1` to run the computation for all the output channels) at 4 bits. The parameter `opt-dim` is used to determine the maximal quadratic problem size to solve. The total number of input channels is decomposed according to this dimensionality.

### Post-training pruning SQNR bounds:
An example for computing the SQNR for post-training quantization and pruning is given in the matlab script `post_training_pruning_acts_sqnr.m`. The script requires CVX package installed with mosek or gurobi solver support.   

### ImageNet experiments
The main run file to reproduce the ImageNet experiments is `main.py`. 
It contains commands for fine-tuning quantized and pruned ImageNet-pretrained models`.

To reproduce the quantization experiments at 4 bits:
```bash
python main.py train-compressed --architecture <ARCHITECTURE_NAME> 
#--model-dir </PATH/TO/PRETRAINED/MODEL> # only needed for MobileNet-V2
--batch-size <BATCH_SIZE>  --n-bits 4 --images-dir </PATH/TO/IMAGENET>
--learning-rate <LEARNING_RATE> --learning-rate-schedule cosine:<LEARNING_RATE_COSINE>   
--no-act-quant --weight-quant-method MSE --optimizer SGD --weight-decay <WEIGHT_DECAY> --sep-quant-optimizer 
--quant-optimizer Adam --quant-learning-rate 1e-5 --num-workers 8 --cuda --num-est-batches 1 
--max-epochs 20 --quant-weight-decay 0.0 --no-per-channel --reestimate-bn-stats
 ```

To reproduce the pruning experiments and 0.75 sparsity:
```bash
python main.py train-compressed --architecture <ARCHITECTURE_NAME> 
#--model-dir </PATH/TO/PRETRAINED/MODEL> # only needed for MobileNet-V2 
--batch-size <BATCH_SIZE>  --images-dir </PATH/TO/IMAGENET>
--learning-rate <LEARNING_RATE> --learning-rate-schedule cosine:<LEARNING_RATE_COSINE> 
--no-act-quant --no-weight-quant --optimizer SGD --weight-decay <WEIGHT_DECAY> --no-sep-quant-optimizer 
--num-workers 8 --cuda --num-est-batches 1 --no-per-channel --reestimate-bn-stats
--max-epochs 20  --prune-method zhu_gupta --s-f 0.75 --s-i 0.0 --prune-epoch-start 0 --prune-last-epoch 15 
 ```

where <ARCHITECTURE_NAME> can be resnet18_quantized, resnet50_quantized, mobilenet_v2_quantized. 
Please note that only MobileNet-V2 requires pre-trained weights that can be downloaded here (the tar file is used as it is without a need to untar):
- [MobileNetV2](https://drive.google.com/open?id=1jlto6HRVD3ipNkAl1lNhDbkBp7HylaqR)

To download it, use the following command:
```
wget -O mobile_net_weights -r 'https://drive.google.com/uc?export=download&id=1jlto6HRVD3ipNkAl1lNhDbkBp7HylaqR'
```

The list of weight decay and learning rate values used to reproduce our experiments is the following.
The value of <LEARNING_RATE_COSINE> in our experiments was always set to 1.0e-2 * <LEARNING_RATE>.

| Model             | Batch size | Weight decay | LR 75% pruning | LR INT4 quantization |
|-------------------|------------|--------------|----------------|----------------------|
| Resnet-18         | 256        | 1.0e-4       | 1.0e-2         | 1.0e-2               |
| Resnet-50         | 128        | 1.0e-4       | 3.3e-2         | 1.0e-5               |
| MobileNet-V2      | 128        | 5.0e-5       | 3.3e-4         | 3.3e-4               |
| EfficientNet-lite | 128        | 5.0e-5       | 3.3e-4         | 3.3e-3               |

## Reference
If you find our work useful, please cite
```
@article{kuzmin2023pruning,
  title={Pruning vs Quantization: Which is Better?},
  author={Kuzmin, Andrey and Nagel, Markus and Van Baalen, Mart and Behboodi, Arash and Blankevoort, Tijmen},
  booktitle={Neural Information Processing Systems},  
  year={2023},
}
```
