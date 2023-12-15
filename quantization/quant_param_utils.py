from quantization.quantizers import QuantizerBase
from quantization.quantizers.rounding_utils import ParametrizedGradEstimatorBase


def separate_quantized_model_params(quant_model):
    """
    This method separates the parameters of the quantized model to 4 categories.
    Parameters
    ----------
    quant_model:      (QuantizedModel)

    Returns
    -------
    quant_params:       (list)
        Quantization parameters, e.g. delta and zero_float
    model_params:    (list)
        The model parameters of the base model without any quantization operations
    grad_params:        (list)
        Parameters found in the gradient estimators (ParametrizedGradEstimatorBase)
    -------

    """
    quant_params, grad_params = [], []
    quant_params_names, grad_params_names = [], []
    for mod_name, module in quant_model.named_modules():
        if isinstance(module, QuantizerBase):
            for name, param in module.named_parameters(recurse=False):
                quant_params.append(param)
                quant_params_names.append(".".join((mod_name, name)))
        if isinstance(module, ParametrizedGradEstimatorBase):
            # gradient estimator params
            for name, param in module.named_parameters(recurse=False):
                grad_params.append(param)
                grad_params_names.append(".".join((mod_name, name)))

    def tensor_in_list(tensor, lst):
        return any([e is tensor for e in lst])

    found_params = quant_params + grad_params

    model_params = [p for p in quant_model.parameters() if not tensor_in_list(p, found_params)]
    model_param_names = [
        n for n, p in quant_model.named_parameters() if not tensor_in_list(p, found_params)
    ]

    print("Quantization parameters ({}):".format(len(quant_params_names)))
    print(quant_params_names)

    print("Gradient estimator parameters ({}):".format(len(grad_params_names)))
    print(grad_params_names)

    print("Other model parameters ({}):".format(len(model_param_names)))
    print(model_param_names)

    assert len(model_params + quant_params + grad_params) == len(
        list(quant_model.parameters())
    ), "{}; {}; {} -- {}".format(
        len(model_params), len(quant_params), len(grad_params), len(list(quant_model.parameters()))
    )

    return quant_params, model_params, grad_params
