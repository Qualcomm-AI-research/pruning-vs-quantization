import os
from functools import partial

import timm


def load_timm_model(model_name, pretrained=False, torch_cache_dirpath=None, **kwargs):
    del kwargs  # unused

    if torch_cache_dirpath is not None:
        os.environ["TORCH_HOME"] = torch_cache_dirpath

    pretrained = True
    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval()
    return model


vit_small_patch16_224 = partial(load_timm_model, "vit_small_patch16_224")
