import torch
import pytorch_lightning as pl
from torchvision import models
from models.byol import PixelCL
from dataset.dataloader_multi_gpu import BYOLDataloader

###########################################################################
''' Hyper parameters'''
batch_size = 32
num_workers = 32
using_pixpro = True # True if using SSL for using backbone to implement segmentation tasks,
                    # False if using SSL for using backbone to implement classification tasks,
num_of_gpus = torch.cuda.device_count()
image_folder_path = r'image_folder_path'
# Define the backbone
backbone = models.resnet34(pretrained=True)
hidden_layer_pixel = 'layer4'
########################################################################
'''
* You can replace other dataloaders with following 

* You can use the following dataloader to train on an Image folder
'''


dataset = BYOLDataloader(data_path=image_folder_path)

trainloader = torch.utils.data.DataLoader(dataset,
                                          shuffle=True,
                                          batch_size=batch_size,
                                          pin_memory=False,
                                          num_workers=num_workers)


#########################################################################


model = PixelCL(
    backbone,
    image_size=256,
    hidden_layer_pixel=hidden_layer_pixel,  # leads to output of 8x8 feature map for pixel-level learning
    hidden_layer_instance=-2,     # leads to output for instance-level learning
    projection_size=256,          # size of projection output, 256 was used in the paper
    projection_hidden_size=2048,  # size of projection hidden dimension, paper used 2048
    moving_average_decay=0.99,    # exponential moving average decay of target encoder
    ppm_num_layers=1,             # number of layers for transform function in the pixel propagation module, 1 was optimal
    ppm_gamma=2,                  # sharpness of the similarity in the pixel propagation module, already at optimal value of 2
    distance_thres=0.7,           # ideal value is 0.7, as indicated in the paper, which makes the assumption of each feature map's pixel diagonal distance to be 1 (still unclear)
    similarity_temperature=0.3,   # temperature for the cosine similarity for the pixel contrastive loss
    alpha=1.,                     # weight of the pixel propagation loss (pixpro) vs pixel CL loss
    use_pixpro=using_pixpro,      # do pixel pro instead of pixel contrast loss, defaults to pixpro, since it is the best one
    cutout_ratio_range=(0.6, 0.8),# a random ratio is selected from this range for the random cutout
    len_dataloader=len(trainloader)
    )

trainer = pl.Trainer(accelerator="gpu",
                     gpus=[i for i in range(num_of_gpus)],
                     auto_select_gpus=True,
                     strategy='ddp',
                     sync_batchnorm=True,
                     max_epochs=250)
trainer.fit(model, trainloader)
