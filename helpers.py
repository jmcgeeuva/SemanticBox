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
from IPython.display import display
# sys.path.insert(0, "./external/tamingtransformers")
# sys.path.append("./external/TransformerMMExplainability")
# import taming.modules
from urllib.request import urlopen
# import taming.models.cond_transformer as cond_transformer
# import taming.models.vqgan as vqgan
# import external.TransformerMMExplainability
# import CLIP.clip as clip

class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_img, min_val, max_val):
        ctx.min = min_val
        ctx.max = max_val
        ctx.save_for_backward(input_img)
        return input_img.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_in):
        input_img, = ctx.saved_tensors
        return grad_in * (grad_in * (input_img - input_img.clamp(ctx.min, ctx.max)) >= 0), None, None


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)