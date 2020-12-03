# MODIFIED FROM
# https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
# https://github.com/asyml/vision-transformer-pytorch/blob/main/src/model.py

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils import (get_width_and_height_from_size, load_pretrained_weights, get_model_params)

VALID_MODELS = ('ViT-B_16', 'ViT-B_32', 'ViT-L_16', 'ViT-L_32')


class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding

        if self.dropout:
            out = self.dropout(out)

        return out


class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out


class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768, ), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim**0.5

        self.query = LinearGeneral((in_dim, ), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim, ), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim, ), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim, ))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, n, _ = x.shape

        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3)

        out = self.out(out, dims=([2, 3], [0, 1]))

        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out


class Encoder(nn.Module):
    def __init__(self,
                 num_patches,
                 emb_dim,
                 mlp_dim,
                 num_layers=12,
                 num_heads=12,
                 dropout_rate=0.1,
                 attn_dropout_rate=0.0):
        super(Encoder, self).__init__()

        # positional embedding
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)

        # encoder blocks
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):

        out = self.pos_embedding(x)

        for layer in self.encoder_layers:
            out = layer(out)

        out = self.norm(out)
        return out


class VisionTransformer(nn.Module):
    """ Vision Transformer.
        Most easily loaded with the .from_name or .from_pretrained methods.
        Args:
            params (namedtuple): A set of Params.
        References:
            [1] https://arxiv.org/abs/2010.11929 (An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)
        Example:
            
            
            import torch
            >>> from vision_transformer_pytorch import VisionTransformer
            >>> inputs = torch.rand(1, 3, 256, 256)
            >>> model = VisionTransformer.from_pretrained('ViT-B_16')
            >>> model.eval()
            >>> outputs = model(inputs)
    """
    def __init__(self, params=None):
        super(VisionTransformer, self).__init__()
        self._params = params

        self.embedding = nn.Conv2d(3, self._params.emb_dim, kernel_size=self.patch_size, stride=self.patch_size)
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self._params.emb_dim))

        # transformer
        self.transformer = Encoder(num_patches=self.num_patches,
                                   emb_dim=self._params.emb_dim,
                                   mlp_dim=self._params.mlp_dim,
                                   num_layers=self._params.num_layers,
                                   num_heads=self._params.num_heads,
                                   dropout_rate=self._params.dropout_rate,
                                   attn_dropout_rate=self._params.attn_dropout_rate)

        # classfier
        self.classifier = nn.Linear(self._params.emb_dim, self._params.num_classes)

    @property
    def image_size(self):
        return get_width_and_height_from_size(self._params.image_size)

    @property
    def patch_size(self):
        return get_width_and_height_from_size(self._params.patch_size)

    @property
    def num_patches(self):
        h, w = self.image_size
        fh, fw = self.patch_size
        gh, gw = h // fh, w // fw
        return gh * gw

    def extract_features(self, x):
        emb = self.embedding(x)  # (n, c, gh, gw)
        emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)

        # prepend class token
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, emb], dim=1)

        # transformer
        feat = self.transformer(emb)
        return feat

    def forward(self, x):
        feat = self.extract_features(x)

        # classifier
        logits = self.classifier(feat[:, 0])
        return logits

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """create an vision transformer model according to name.
        Args:
            model_name (str): Name for vision transformer.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'image_size', 'patch_size',
                    'emb_dim', 'mlp_dim',
                    'num_heads', 'num_layers',
                    'num_classes', 'attn_dropout_rate',
                    'dropout_rate'
        Returns:
            An vision transformer model.
        """
        cls._check_model_name_is_valid(model_name)
        params = get_model_params(model_name, override_params)
        model = cls(params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, in_channels=3, num_classes=1000, **override_params):
        """create an vision transformer model according to name.
        Args:
            model_name (str): Name for vision transformer.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'image_size', 'patch_size',
                    'emb_dim', 'mlp_dim',
                    'num_heads', 'num_layers',
                    'num_classes', 'attn_dropout_rate',
                    'dropout_rate'
        Returns:
            A pretrained vision transformer model.
        """
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path, load_fc=(num_classes == 1000))
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.
        Args:
            model_name (str): Name for vision transformer.
        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.
        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            self.embedding = nn.Conv2d(in_channels,
                                       self._params.emb_dim,
                                       kernel_size=self.patch_size,
                                       stride=self.patch_size)
