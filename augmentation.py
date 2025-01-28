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
# from IPython.display import display
# sys.path.insert(0, "./external/tamingtransformers")
# sys.path.append("./external/TransformerMMExplainability")
# import taming.modules
from urllib.request import urlopen
# import taming.models.cond_transformer as cond_transformer
# import taming.models.vqgan as vqgan
# import external.TransformerMMExplainability
# import CLIP.clip as clip

from helpers import *
from prompt import *

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

class Stack(object):

    def __init__(self):
        pass

    def __call__(self, data):
        cutouts, augmented_masks = data
        if type(cutouts) != list:
            batch = torch.stack([cutouts], dim=0)
        else:
            batch = torch.stack(cutouts, dim=0)
        augmented_masks = torch.stack(augmented_masks, dim=0)
        return batch, augmented_masks

class ImageAugmentations(object):

    def __init__(self, is_classifier=False):
        self.img_augs = nn.Sequential(
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1/.3), p=0.7),
        )
        self.is_classifier = is_classifier

    def __call__(self, data):
        batch, augmented_masks = data
        if self.is_classifier:
            batch = batch.squeeze()
            augmented_masks = augmented_masks.squeeze(dim=0)
        batch = self.img_augs(batch)
        return batch, augmented_masks

class GroupNormalize(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

    def __call__(self, data):
        batch, augmented_masks = data
        batch, augmented_masks = self.normalize(batch), augmented_masks
        return batch, augmented_masks

class GroupAug(object):
    """Randomly Grayscale flips the given PIL.Image with a probability
    """
    def __init__(self, cut_size, cutn, cut_pow, is_classifier, noise_fac=0.1):
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border', same_on_batch=True),
            K.RandomPerspective(0.7,p=0.7, same_on_batch=True)
        )
        self.img_augs = ImageAugmentations(is_classifier)
        self.add_noise = AddNoise(noise_fac=noise_fac, cutn=cutn)
        self.frame_len = 8
        self.av_pool = nn.AdaptiveAvgPool3d((self.frame_len, self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool3d((self.frame_len, self.cut_size, self.cut_size))

    def __call__(self, data):
        input_img, input_masks = data
        cutouts = []
        augmented_masks = []

        for _ in range(self.cutn):

            
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
            augmented_masks = torch.stack(augmented_masks, dim=0)
            batch = batch.reshape(-1, c, h, w)
            augmented_masks = augmented_masks.reshape(-1, 1, h, w)
            batch, augmented_masks = self.img_augs((batch, augmented_masks))
            batch, augmented_masks = self.add_noise((batch, augmented_masks))
            batch = batch.reshape(input_img.shape[0], t, c, h, w)
            augmented_masks = augmented_masks.reshape(input_img.shape[0], t, 1, h, w)
            return batch, augmented_masks

def get_mask_augmentation(cut_size, cutn, cut_pow=1., noise_fac = 0.1, data_set=None, is_classifier=False):
    # Isn't this just a bunch of transforms?
    unique = torchvision.transforms.Compose([
        GroupAug(cut_size, cutn, cut_pow, is_classifier=is_classifier)
    ])
    common = torchvision.transforms.Compose([
        # Normalize Batch
        # GroupNormalize()
    ])
    
    return torchvision.transforms.Compose([unique, common])