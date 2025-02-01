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

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, data):
        img_group, img_mask = data['video'], data['mask']
        return {'video': [self.worker(img) for img in img_group], 'mask': img_mask}

def get_mask_augmentation(cut_size, cutn, cut_pow=1., noise_fac = 0.1):
    # Isn't this just a bunch of transforms?
    unique = torchvision.transforms.Compose([
        GroupAug(cut_size, cutn, cut_pow)
        #ImageAugmentations(),
    ])
    
    return torchvision.transforms.Compose([unique])

def get_augmentation(training, config):
    input_mean=[0.5, 0.5, 0.5]
    input_std=[0.5, 0.5, 0.5]
    # input_mean = [0.48145466, 0.4578275, 0.40821073]
    # input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = config.data.input_size * 256 // 224
    if training:
        unique = torchvision.transforms.Compose([
                                                 GroupMultiScaleCrop(config.data.input_size, [1, .875, .75, .66]),
                                                 GroupRandomHorizontalFlip(is_sth='some' in config.data.dataset),
                                                 GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
                                                                        saturation=0.2, hue=0.1),
                                                 GroupRandomGrayscale(p=0.2),
                                                 GroupGaussianBlur(p=0),
                                                 GroupSolarization(p=0)]
                                                )
    else:
        unique = torchvision.transforms.Compose([
                                                 GroupScale(scale_size),
                                                 GroupCenterCrop(config.data.input_size)
                                                 ])

    common = torchvision.transforms.Compose([GroupStack(roll=False),
                                             GroupToTorchFormatTensor(div=True),
                                             GroupNormalize(input_mean,
                                                            input_std)])
    return torchvision.transforms.Compose([unique, common])

def randAugment(transform_train,config):
    print('Using RandAugment!')
    import torchvision.transforms as transforms
    transform_train.transforms.insert(0, GroupTransform(transforms.RandAugment(config.data.randaug.N, config.data.randaug.M)))
    return transform_train
