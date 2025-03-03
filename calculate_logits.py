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
sys.path.insert(0, "./ml-no-token-left-behind/external/tamingtransformers")
sys.path.append("./ml-no-token-left-behind/external/TransformerMMExplainability")
# import taming.modules
from urllib.request import urlopen
# import taming.models.cond_transformer as cond_transformer
# import taming.models.vqgan as vqgan
# import external.TransformerMMExplainability
# import CLIP.clip as clip
from modeling_florence2 import shift_tokens_right

######################################
import os
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
from modules.Visual_Prompt import visual_prompt
from utils.KLLoss import KLLoss
# from test import validate
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import  *
import sys
sys.path.insert(0, "./../ml-no-token-left-behind/explain/mlexternal/tamingtransformers")
sys.path.append("./../ml-no-token-left-behind/external/TransformerMMExplainability")
from helpers import ReplaceGrad
######################################

from helpers import *

def calculate_logits(text_strs, processor, model_image, model_text, fusion_model, texts, videos):
    videos = videos.squeeze(dim=1)
    b,t,c,h,w = videos.size()
    images = videos.view(-1,c,h,w )
    text = []
    for i, v in enumerate(text_strs):
        caption = [f'<CAPTION_TO_PHRASE_GROUNDING>{v}' for _ in range(t)] #
        text.extend(caption)
    
    images = [transforms.functional.to_pil_image(image) for image in images]
    
    inputs = processor(text=text, images=images, padding=True, do_resize=True, return_tensors="pt")
    pixel_values = inputs['pixel_values']
    input_ids = inputs['input_ids']
    
    image_features, attention_mask, inputs_embeds, texts = model_image(input_ids, pixel_values, texts, text_strs)

    i_n, i_c, i_e = inputs_embeds.shape
    b = i_n//t
    inputs_embeds = inputs_embeds.view(b,t,i_c,-1)
    inputs_embeds = inputs_embeds.permute(0, 2, 1, 3)
    inputs_embeds = inputs_embeds.reshape(-1,t,inputs_embeds.shape[-1])
    # if 80-i_c > 0:
    #     inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, 80-i_c))
    #     i_c = 80
    # FIXME play with how this is run and how we get temporal understanding (maybe move to before image embedding)
    inputs_embeds = fusion_model(inputs_embeds)
    inputs_embeds = inputs_embeds.view(b,i_c,-1)
    
    text_embedding, logits = model_text(attention_mask, inputs_embeds, texts)
    flo_loss = logits.loss

    image_features = image_features.view(b, -1, image_features.shape[-1])
    image_features = image_features.mean(dim=1)
    # TODO add embedding to expand from 768 to context length
    image_features = model_text.language_model.lm_head(image_features)
    
    logit_scale = 100.0 #perceptor.logit_scale.exp()
    logits_per_image, logits_per_text = create_logits(image_features, text_embedding, logit_scale)
    return image_features, logits_per_image, logits_per_text, flo_loss