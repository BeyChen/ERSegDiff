# Tool: PyCharm
# coding: utf-8
"""=========================================
# Project_Name: light_seg
# Author: WenDong
# Date: 2022/4/8 15:33
# Function: 
# Description: 
========================================="""

import os
import random

from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import torch


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=256, mode='train', augmentation_prob=0.4):

        # self.root = root
        # self.GT_paths = root[:-1] + '_GT/'
        self.GT_paths = os.path.join(root, 'masks')
        self.root = os.path.join(root, 'imgs')
        self.image_paths = list(map(lambda x: os.path.join(self.root, x), os.listdir(self.root)))
        self.image_size = image_size
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        filename = image_path.split('_')[-1][:-len(".jpg")]
        GT_path = os.path.join(self.GT_paths, 'ISIC_' + filename + '_segmentation.png')

        image = Image.open(image_path)
        GT = Image.open(GT_path)
        Transform = []
        # 随机设置大小
        Transform.append(T.Resize((self.image_size, self.image_size)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        image = Transform(image)
        GT = Transform(GT)
        Norm_ = T.Normalize((0.2197, 0.2197, 0.2197), (0.6025, 0.6025, 0.6025))
        image = Norm_(image)
        return {
            'image': image.type(torch.FloatTensor),
            'mask': GT.type(torch.FloatTensor),
            'image_name': filename,
            'mask_name': filename + '_mask'
        }
    def __len__(self):
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=0, mode='train', augmentation_prob=0.4, shuffle=True):
    dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=True)
    return data_loader



