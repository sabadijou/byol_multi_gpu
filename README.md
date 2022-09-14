# BYOL: Bootstrap Your Own Latent - Multi GPU Support

PyTorch implementation of [BYOL](https://arxiv.org/abs/2006.07733): a new approach to self-supervised image representation learning.
1. This repository enables standard multi GPU training.
2. Both instance loss and PixPro are provided for image classification and segmentation.
3. Benefits from gather-layer that collects all batches from all devices to calculate the loss.

# Paper Abstract
BYOL relies on two neural networks, referred to as online and target networks, that interact and learn from each other. From an augmented view of an image, we train the online network to predict the target network representation of the same image under a different augmented view. At the same time, we update the target network with a slow-moving average of the online network. While state-of-the art methods rely on negative pairs, BYOL achieves a new state of the art without them. BYOL reaches 74.3% top-1 classification accuracy on ImageNet using a linear evaluation with a ResNet-50 architecture and 79.6% with a larger ResNet. We show that BYOL performs on par or better than the current state of the art on both transfer and semi-supervised benchmarks.

## Get started
1. Clone the RESA repository
    ```
    git clone https://github.com/zjulearning/resa.git
    ```
    We call this directory as `$RESA_ROOT`

2. Create a conda virtual environment and activate it (conda is optional)

    ```Shell
    conda create -n resa python=3.8 -y
    conda activate resa
    ```

3. Install dependencies

    ```Shell
    # Install pytorch firstly, the cudatoolkit version should be same in your system. (you can also use pip to install pytorch and torchvision)
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

    # Or you can install via pip
    pip install torch torchvision

    # Install python packages
    pip install -r requirements.txt
    ```
