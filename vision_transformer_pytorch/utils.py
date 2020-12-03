# MODIFIED FROM
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py

import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

################################################################################
### Help functions for model architecture
################################################################################

# Params: namedtuple
# get_width_and_height_from_size and calculate_output_image_size

# Parameters for the entire model (stem, all blocks, and head)
Params = collections.namedtuple('Params', [
    'image_size', 'patch_size', 'emb_dim', 'mlp_dim', 'num_heads', 'num_layers', 'num_classes', 'attn_dropout_rate',
    'dropout_rate'
])

# Set Params and BlockArgs's defaults
Params.__new__.__defaults__ = (None, ) * len(Params._fields)


def get_width_and_height_from_size(x):
    """Obtain height and width from x.
    Args:
        x (int, tuple or list): Data size.
    Returns:
        size: A tuple or list (H,W).
    """
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


################################################################################
### Helper functions for loading model params
################################################################################

# get_model_params and efficientnet:
#     Functions to get BlockArgs and GlobalParams for efficientnet
# url_map and url_map_advprop: Dicts of url_map for pretrained weights
# load_pretrained_weights: A function to load pretrained weights


def vision_transformer(model_name):
    """Create Params for vision transformer model.
    Args:
        model_name (str): Model name to be queried.
    Returns:
        Params(params_dict[model_name])
    """

    params_dict = {
        'ViT-B_16': (384, 16, 768, 3072, 12, 12, 1000, 0.0, 0.1),
        'ViT-B_32': (384, 32, 768, 3072, 12, 12, 1000, 0.0, 0.1),
        'ViT-L_16': (384, 16, 1024, 4096, 16, 24, 1000, 0.0, 0.1),
        'ViT-L_32': (384, 32, 1024, 4096, 16, 24, 1000, 0.0, 0.1)
    }
    image_size, patch_size, emb_dim, mlp_dim, num_heads, num_layers, num_classes, attn_dropout_rate, dropout_rate = params_dict[model_name]
    params = Params(
        image_size=image_size,
        patch_size=patch_size,
        emb_dim=emb_dim,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        attn_dropout_rate=attn_dropout_rate,
        dropout_rate=dropout_rate
    )

    return params


def get_model_params(model_name, override_params):
    """Get the block args and global params for a given model name.
    Args:
        model_name (str): Model's name.
        override_params (dict): A dict to modify params.
    Returns:
        params
    """
    params = vision_transformer(model_name)

    if override_params:
        # ValueError will be raised here if override_params has fields not included in params.
        params = params._replace(**override_params)
    return params


# train with Standard methods
# check more details in paper(EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks)
url_map = {
    'ViT-B_16':
    'https://github.com/tczhangzhi/VisionTransformer-PyTorch/releases/download/1.0.1/ViT-B_16_imagenet21k_imagenet2012.pth',
    'ViT-B_32':
    'https://github.com/tczhangzhi/VisionTransformer-PyTorch/releases/download/1.0.1/ViT-B_32_imagenet21k_imagenet2012.pth',
    'ViT-L_16':
    'https://github.com/tczhangzhi/VisionTransformer-PyTorch/releases/download/1.0.1/ViT-L_16_imagenet21k_imagenet2012.pth',
    'ViT-L_32':
    'https://github.com/tczhangzhi/VisionTransformer-PyTorch/releases/download/1.0.1/ViT-L_32_imagenet21k_imagenet2012.pth',
}


def load_pretrained_weights(model, model_name, weights_path=None, load_fc=True, advprop=False):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): The whole model of vision transformer.
        model_name (str): Model name of vision transformer.
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
    """
    if isinstance(weights_path, str):
        state_dict = torch.load(weights_path)
    else:
        state_dict = model_zoo.load_url(url_map[model_name])

    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        assert not ret.missing_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    else:
        state_dict.pop('classifier.weight')
        state_dict.pop('classifier.bias')
        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == set(
            ['classifier.weight', 'classifier.bias']), 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    assert not ret.unexpected_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)

    print('Loaded pretrained weights for {}'.format(model_name))