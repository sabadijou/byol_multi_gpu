# BYOL: Bootstrap Your Own Latent - Multi GPUs Support

PyTorch implementation of [BYOL](https://arxiv.org/abs/2006.07733): a new approach to self-supervised image representation learning.

2. (Optionally) Automatically trains a linear classifier, and logs its accuracy after each epoch.
3. All functions and classes are fully type annotated for better usability/hackability with Python>=3.6.


## TO DO
* Enable mixed-precision training in PyTorch Lightning.  `kornia.augmentation.RandomResizedCrop` currently doesn't support this.  I'll need to ensure that our implementation is sufficiently performant, so it doesn't inadvertently slow down training.
