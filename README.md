# BYOL: Bootstrap Your Own Latent - Multi GPU Support

PyTorch implementation of [BYOL](https://arxiv.org/abs/2006.07733): a new approach to self-supervised image representation learning.
1. This repository enables standard multi GPU training.
2. Both instance loss and PixPro are provided for image classification and segmentation.
3. Benefits from gather-layer that collects all batches from all devices to calculate the loss.

## Paper Abstract
BYOL relies on two neural networks, referred to as online and target networks, that interact and learn from each other. From an augmented view of an image, we train the online network to predict the target network representation of the same image under a different augmented view. At the same time, we update the target network with a slow-moving average of the online network. While state-of-the art methods rely on negative pairs, BYOL achieves a new state of the art without them. BYOL reaches 74.3% top-1 classification accuracy on ImageNet using a linear evaluation with a ResNet-50 architecture and 79.6% with a larger ResNet. We show that BYOL performs on par or better than the current state of the art on both transfer and semi-supervised benchmarks.

## Get started
1. Clone the repository
    ```
    git clone https://github.com/sabadijou/byol_multi_gpu.git
    ```
    We call this directory as `$RESA_ROOT`

2. Create an environment and activate it (We've used conda. but it is optional)

    ```Shell
    conda create -n byol python=3.9 -y
    conda activate 
    ```

3. Install dependencies

    ```Shell
    # Install pytorch firstly, the cudatoolkit version should be same in your system. (you can also use pip to install pytorch and torchvision)
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    
    # Install pytorch_lightning
    pip install pytorch_lightning
  
    # Install kornia and einops
    pip install kornia
    pip install einops
    ```
  ## Get started
  1. Simply open train.py in a python editor and customize the hyperparameters section.
  ```Shell
  batch_size = 32
  num_workers = 32
  using_pixpro = True   # True if using SSL for using backbone to implement segmentation tasks,
                        # False if using SSL for using backbone to implement classification tasks,
  num_of_gpus = torch.cuda.device_count()
  image_folder_path = r'image_folder_path'
  # Define the backbone
  backbone = models.resnet34(pretrained=True)
  hidden_layer_pixel = 'layer4'
  '''
