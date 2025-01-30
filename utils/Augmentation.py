# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

from datasets.transforms_ss import *
from randaugment import RandAugment
# @title Load libraries and variables
import os
import json
import torch
from PIL import Image
from omegaconf import OmegaConf
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from captum.attr import visualization
from torchvision import transforms
from argparse import Namespace
import torch
import torchvision.transforms as transforms
import sys
import numpy as np
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import math
from pathlib import Path
import sys
from IPython import display
from base64 import b64encode
from PIL import Image
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm
import kornia.augmentation as K
import numpy as np
import imageio
from urllib.request import urlopen

from helpers import *

import torchvision
from enum import Enum

class DATASET(Enum):
    MNIST = 1
    FLOWERS = 2
    CIFAR = 3
    OXFORD_PET = 4


class AddNoise(object):

    def __init__(self, noise_fac, cutn):
        self.noise_fac = 0.1
        self.cutn = cutn

    def __call__(self, data):
        batch, augmented_masks = data
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch, augmented_masks

# class Stack(object):

#     def __init__(self):
#         pass

#     def __call__(self, data):
#         cutouts, augmented_masks = data
#         if type(cutouts) != list:
#             batch = torch.stack([cutouts], dim=0)
#         else:
#             batch = torch.stack(cutouts, dim=0)
#         augmented_masks = torch.stack(augmented_masks, dim=0)
#         return batch, augmented_masks

class ImageAugmentations(object):

    def __init__(self):
        self.img_augs = nn.Sequential(
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1/.3), p=0.7),
        )

    def __call__(self, data):
        batch, augmented_masks = data
        batch = self.img_augs(batch)
        return batch, augmented_masks

# class GroupNormalize(object):
#     def __init__(self):
#         self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
#                                     std=[0.26862954, 0.26130258, 0.27577711])

#     def __call__(self, data):
#         batch, augmented_masks = data
#         batch, augmented_masks = self.normalize(batch), augmented_masks
#         return batch, augmented_masks

class GroupAug(object):
    """Randomly Grayscale flips the given PIL.Image with a probability
    """
    def __init__(self, cut_size, cutn, cut_pow, noise_fac=0.1):
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border', same_on_batch=True),
            K.RandomPerspective(0.7,p=0.7, same_on_batch=True)
        )
        # self.img_augs = ImageAugmentations(is_classifier)
        self.add_noise = AddNoise(noise_fac=noise_fac, cutn=cutn)
        self.frame_len = 8
        self.av_pool = nn.AdaptiveAvgPool3d((self.frame_len, self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool3d((self.frame_len, self.cut_size, self.cut_size))

    def __call__(self, data):
        input_img, input_masks = data['videos'], data['masks']
        cutouts = []
        augmented_masks = []

        masks = []

        permuted_input_img = input_img.permute(0, 2, 1, 3, 4) # switch time and channel
        for mask_i in range(len(input_masks)):
            curr_mask = input_masks[mask_i].permute(1, 2, 0, 3, 4)
            masks.append((self.av_pool(curr_mask) + self.max_pool(curr_mask))/2)
        cutout = (self.av_pool(permuted_input_img) + self.max_pool(permuted_input_img))/2

        sample = torch.cat([cutout] + masks, dim=0)
        
        beg_neutral_masks = len([cutout] + masks)

        sample = sample.permute(0, 2, 1, 3, 4) # switch time and channel back
        b, t, c, h, w = sample.size()
        sample = sample.reshape(-1, c, h, w)
        aug_sample = self.augs(sample)
        # (128+128)x3x224x224
        aug_sample = aug_sample.reshape(b, t, c, h, w)
        cutouts.append(aug_sample[0:input_img.shape[0]])

        curr_augmented_masks = aug_sample[input_img.shape[0]::]
        curr_augmented_masks = curr_augmented_masks[:, :, 0:1, ...]
        curr_augmented_masks = torch.round(curr_augmented_masks)
        augmented_masks.append(curr_augmented_masks)

        batch = torch.stack(cutouts, dim=0)
        batch = batch.reshape(-1, c, h, w)
        batch = batch.squeeze()
        augmented_masks = torch.stack(augmented_masks, dim=0)
        augmented_masks = augmented_masks.reshape(-1, 1, h, w)
        augmented_masks = augmented_masks.squeeze(dim=0)
        
        batch, augmented_masks = self.add_noise((batch, augmented_masks))
        batch = batch.reshape(input_img.shape[0], t, c, h, w)
        augmented_masks = augmented_masks.reshape(input_img.shape[0], t, 1, h, w)
        return batch, augmented_masks

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

def get_mask_augmentation(cut_size, cutn, cut_pow=1., noise_fac = 0.1, data_set=None):
    # Isn't this just a bunch of transforms?
    unique = torchvision.transforms.Compose([
        GroupAug(cut_size, cutn, cut_pow)
        #ImageAugmentations(),
    ])
    
    return torchvision.transforms.Compose([unique])

def get_augmentation(training, config):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = config.data.input_size * 256 // 224
    bounding_boxes = config.bounding_boxes
    if training:
        # FIXME move bounding box cropping here
        unique = torchvision.transforms.Compose([
                                                # torchvision.transforms.Resize((224, 224)),
                                                #  GroupMultiScaleCrop(config.data.input_size, [1, .875, .75, .66], bounding_boxes=bounding_boxes),
                                                #  GroupRandomHorizontalFlip(is_sth='some' in config.data.dataset),
                                                 GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
                                                                        saturation=0.2, hue=0.1),
                                                 GroupRandomGrayscale(p=0.2),
                                                 GroupGaussianBlur(p=0.2),
                                                 GroupSolarization(p=0.2)]
                                                )
    else:
        unique = torchvision.transforms.Compose([
                                                #  GroupScale(scale_size),
                                                #  GroupCenterCrop(config.data.input_size)
                                                 ])

    common = torchvision.transforms.Compose([Stack(roll=False),
                                             ToTorchFormatTensor(div=True),
                                             GroupNormalize(input_mean,
                                                            input_std)])
    return torchvision.transforms.Compose([unique, common])

def randAugment(transform_train,config):
    print('Using RandAugment!')
    import torchvision.transforms as transforms
    transform_train.transforms.insert(0, GroupTransform(transforms.RandAugment(config.data.randaug.N, config.data.randaug.M)))
    return transform_train
