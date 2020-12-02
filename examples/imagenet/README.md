### Imagenet

This is a preliminary directory for evaluating the model on ImageNet. It is adapted from the standard PyTorch Imagenet script.

For now, only evaluation is supported, but I am currently building scripts to assist with training new models on Imagenet.

To run on Imagenet, place your `train` and `val` directories in `data`.

Example commands:

```
# Evaluate small VisionTransformer on CPU
python main.py data -e -a 'ViT-B_16' --pretrained 
# Evaluate large VisionTransformer on GPU
python main.py data -e -a 'ViT-L_32' --pretrained --gpu 0 --batch-size 128
# Evaluate ResNet-50 for comparison
python main.py data -e -a 'resnet50' --pretrained --gpu 0
```