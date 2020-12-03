# Vision Transformer Pytorch
This project is modified from [lukemelas](https://github.com/lukemelas)/[EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) and [asyml](https://github.com/asyml)/[vision-transformer-pytorch](https://github.com/asyml/vision-transformer-pytorch) to provide out-of-box API for you to utilize VisionTransformer as easy as EfficientNet.

### Quickstart

Install with `pip install vision_transformer_pytorch` and load a pretrained VisionTransformer with:

```
from vision_transformer_pytorch import VisionTransformer
model = VisionTransformer.from_pretrained('ViT-B_16')
```

### About Vision Transformer PyTorch

Vision Transformer Pytorch is a PyTorch re-implementation of Vision Transformer based on one of the best practice of commonly utilized deep learning libraries, [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch), and an elegant implement of VisionTransformer, [vision-transformer-pytorch](https://github.com/asyml/vision-transformer-pytorch). In this project, we aim to make our PyTorch implementation as simple, flexible, and extensible as possible.

If you have any feature requests or questions, feel free to leave them as GitHub issues!

### Installation

Install via pip:

```
pip install vision_transformer_pytorch
```

Or install from source:

```
git clone https://github.com/tczhangzhi/VisionTransformer-Pytorch
cd VisionTransformer-Pytorch
pip install -e .
```

### Usage

#### Loading pretrained models

Load an EfficientNet:

```
from vision_transformer_pytorch import VisionTransformer
model = VisionTransformer.from_name('ViT-B_16')
```

Load a pretrained EfficientNet:

```
from vision_transformer_pytorch import VisionTransformer
model = VisionTransformer.from_pretrained('ViT-B_16')
# inputs = torch.randn(1, 3, *model.image_size)
# model(inputs)
# model.extract_features(inputs)
```

Default hyper parameters:

| Param\Model       | ViT-B_16 | ViT-B_32 | ViT-L_16 | ViT-L_32 |
| ----------------- | -------- | -------- | -------- | -------- |
| image_size        | 384      | 384      | 384      | 384      |
| patch_size        | 16       | 32       | 16       | 32       |
| emb_dim           | 768      | 768      | 1024     | 1024     |
| mlp_dim           | 3072     | 3072     | 4096     | 4096     |
| num_heads         | 12       | 12       | 16       | 16       |
| num_layers        | 12       | 12       | 24       | 24       |
| num_classes       | 1000     | 1000     | 1000     | 1000     |
| attn_dropout_rate | 0.0      | 0.0      | 0.0      | 0.0      |
| dropout_rate      | 0.1      | 0.1      | 0.1      | 0.1      |

If you need to modify these hyper parameters, please use:

```
from vision_transformer_pytorch import VisionTransformer
model = VisionTransformer.from_name('ViT-B_16', image_size=256, patch_size=64, ...)
```

#### ImageNet

See `examples/imagenet` for details about evaluating on ImageNet.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!