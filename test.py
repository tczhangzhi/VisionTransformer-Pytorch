import torch
import torch.nn as nn

from vision_transformer_pytorch import VisionTransformer

net = VisionTransformer.from_pretrained('R50+ViT-B_16')
