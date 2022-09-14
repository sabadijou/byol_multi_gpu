from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import torch
import os


def img_transformer(h, w):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(size=(h, w))])
    return transform


class BYOLDataloader(torch.utils.data.Dataset):
    def __init__(self, data_path, img_h=224, img_w=224):
        super(BYOLDataloader, self).__init__()
        self.data_path = data_path
        self.sample_path = os.listdir(self.data_path)
        self.img_h = img_h
        self.img_w = img_w
        self.transformer = img_transformer(img_h, img_w)

    def __getitem__(self, idx):

        # Load Sample #########################################
        try:
            sample = Image.open(os.path.join(self.data_path,
                                             self.sample_path[idx]), mode='r', formats=None)
            sample = np.array(sample)
            sample = np.resize(sample, new_shape=(self.img_h, self.img_w, 3))
        except:
            sample = Image.open(os.path.join (self.data_path,
                                               self.sample_path[0]), mode='r', formats=None)
            sample = np.array(sample)
            sample = np.resize(sample, new_shape=(self.img_h, self.img_w, 3))
        sample = self.transformer(sample)
        return sample

    def __len__(self):
        return len(self.sample_path)
