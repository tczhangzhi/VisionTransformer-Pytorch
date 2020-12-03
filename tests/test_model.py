# MODIFIED FROM
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/tests/test_model.py

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from vision_transformer_pytorch import VisionTransformer

# -- fixtures -------------------------------------------------------------------------------------


@pytest.fixture(scope='module', params=['ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32'])
def model(request):
    return request.param


@pytest.fixture(scope='module', params=[True, False])
def pretrained(request):
    return request.param


@pytest.fixture(scope='function')
def net(model, pretrained):
    return VisionTransformer.from_pretrained(model) if pretrained else VisionTransformer.from_name(model)


# -- tests ----------------------------------------------------------------------------------------


def test_forward(net):
    """Test `.forward()` doesn't throw an error"""
    data = torch.zeros((1, 3, *net.image_size))
    output = net(data)
    assert not torch.isnan(output).any()

@pytest.mark.parametrize('img_size', [224, 256, 512])
def test_hyper_params(model, img_size):
    """Test `.forward()` doesn't throw an error with different input size"""
    data = torch.zeros((1, 3, img_size, img_size))
    net = VisionTransformer.from_name(model, image_size=img_size)
    output = net(data)
    assert not torch.isnan(output).any()


def test_modify_classifier(net):
    """Test ability to modify fc modules of network"""
    classifier = nn.Linear(net._params.emb_dim, net._params.num_classes)

    net.classifier = classifier

    data = torch.zeros((2, 3, *net.image_size))
    output = net(data)
    assert not torch.isnan(output).any()