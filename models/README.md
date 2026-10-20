# models/

Written by step 9 of `notebooks/Proj.ipynb`.

| File | What it is |
|---|---|
| `cnn_seed0.pt`, `cnn_seed1.pt`, `cnn_seed2.pt` | CNN on pixels, one file per seed |
| `resnet_seed0.pt` to `resnet_seed2.pt` | ResNet on pixels |
| `diffusion_cnn_seed0.pt` to `diffusion_cnn_seed2.pt` | CNN on the diffusion residual |

Each file is the `state_dict` of a network (the weights of the epoch with the lowest validation loss). To use one, build the network with the classes from step 6 of the notebook (`CNN()` or `ResNet()`) and call `model.load_state_dict(torch.load(path))`. The diffusion CNN expects the standardised residual images from step 7 as input.

The pretrained diffusion model (`google/ddpm-cifar10-32`) is downloaded from Hugging Face when needed and is not stored here.
