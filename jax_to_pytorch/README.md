### JAX to PyTorch Conversion

This directory is used to convert JAX weights to PyTorch. It was hacked together fairly quickly, so the code is not the most beautiful (just a warning!), but it does the job. I will be refactoring it soon.

I should also emphasize that you do *not* need to run any of this code to load pretrained weights. Simply use `VisionTransformer.from_pretrained(...)`.

That being said, the main script here is `convert_to_jax/load_jax_weights.py`. In order to use it, you should first download the pre-trained JAX weights following the description official repository.

>You can find all these models in the following storage bucket:
>
>https://console.cloud.google.com/storage/vit_models/
>
>For example, if you would like to download the ViT-B/16 pre-trained on imagenet21k run the following command:

```
mkdir pretrained_jax
cd pretrained_jax
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
cd ..
```

Then

```
mkdir pretrained_pytorch
cd convert_jax_to_pt
python load_jax_weights.py \
    --jax_checkpoint ../pretrained_jax/ViT-B_16.npz \
    --output_file ../pretrained_pytorch/ViT-B_16.pth
```