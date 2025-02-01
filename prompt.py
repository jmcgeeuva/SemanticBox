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
from test import validate
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

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self,text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self,image):
        return self.model.encode_image(image)

class PromptLoss(nn.Module):
    def __init__(self, text, perceptor, replace_grad):
        super().__init__()
        text, weight, stop = self.parse_prompt(text)
        self.register_buffer('tokenized_text', clip.tokenize(text))
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))
        self.t = 0.1  # threshold
        self.temperature = 20 # temp

        self.perceptor_model = perceptor
        self.replace_grad = replace_grad

    def parse_prompt(self, prompt):
        vals = prompt.rsplit(':', 2)
        # if only one or neither of the weight/stop are 
        # provided then set to defaults weight = 1 and stop = -inf
        vals = vals + ['', '1', '-inf'][len(vals):]
        return vals[0], float(vals[1]), float(vals[2])

    def spatial_explainability_loss(self, input_image, mask, token_text):
        batch_size = input_image.shape[0]
        print(input_image.shape,token_text.shape)
        text = token_text.repeat(batch_size, 1)
        print(text.shape)
        raise ValueError("test")
        index = [i for i in range(batch_size)]
        clip_c = self.perceptor_model.logit_scale.exp()
        self.perceptor_model.zero_grad()

        with torch.enable_grad():
            logits_per_image, logits_per_text = self.perceptor_model(input_image, text)
            logits_per_image = logits_per_image
            logits_per_image = logits_per_image / clip_c

            one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
            one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.to(input_image.device) * logits_per_image)

            image_attn_blocks = list(dict(self.perceptor_model.visual.transformer.resblocks.named_children()).values())
            # print(dir(image_attn_blocks[0]))
            num_tokens = image_attn_blocks[0].attn.attn_output_weights.shape[-1]
            R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn.attn_output_weights.dtype).to(logits_per_image.device)
            R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)

            for blk_idx, blk in enumerate(image_attn_blocks):
                if blk_idx <= 10:
                    continue

                grad = torch.autograd.grad(one_hot, [blk.attn.attn_output_weights], retain_graph=True, create_graph=True)[0]

                # print("It worked")
                cam = blk.attn.attn_output_weights
                cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
                cam = cam.clamp(min=0).mean(dim=1)
                R = R + torch.bmm(cam, R)

            R[:, 0, 0] = 0
            image_relevance = R[:, 0, 1:]

            image_relevance = image_relevance.reshape(-1, 1, 7, 7)
            image_relevance = torch.nn.functional.interpolate(image_relevance, size=mask.shape[-1], mode='bicubic')
            image_relevance = image_relevance / torch.sum(image_relevance, dim=(-2, -1), keepdim=True)
            max = image_relevance.max(dim=-1, keepdim=True)[0]
            max = max.max(dim=-2, keepdim=True)[0]
            img_expl = image_relevance / max

            img_expl = (img_expl - self.t) * self.temperature
            binarized_expl = torch.sigmoid(img_expl)

            mask = mask.to(img_expl.device)
            intersection = (binarized_expl * mask).float().sum((-2, -1))
            union = (binarized_expl * (1 - mask)).float().sum((-2, -1))
            # import pdb; pdb.set_trace()
            mask_size = 1
            for dim in mask.shape:
                mask_size *= dim
            # mask_size = (mask.shape[0] * mask.shape[1] * mask.shape[2] * mask.shape[3])
            dice_loss = (2 * intersection / (2 * intersection + union))

        self.perceptor_model.zero_grad()
        return (-1) * dice_loss

    def forward(self, input_img, mask, dynamic_lambda):
        masked_input = input_img * mask.to(input_img.device)
        masked_input = torch.cat([input_img, masked_input], dim=0)

        print(self.tokenized_text.shape)
        expl_loss = self.spatial_explainability_loss(input_img, mask, self.tokenized_text)
        expl_loss = expl_loss * dynamic_lambda

        image_embedding = self.perceptor_model.encode_image(masked_input).float()
        # input = input.to(clip_device)
        input_normed = F.normalize(image_embedding.unsqueeze(1), dim=2)

        print(self.tokenized_text.shape)
        embed = self.perceptor_model.encode_text(self.tokenized_text).float()
        raise ValueError("test")
        embed_normed = F.normalize(embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()

        loss = self.replace_grad(dists, torch.maximum(dists, self.stop)).mean() + expl_loss.mean()
        return self.weight.abs() * loss

class PromptLoss2(nn.Module):
    def __init__(self, perceptor, replace_grad, im_emb_type='just_image'): 
        super().__init__()
        weight = float('1')
        stop = float('-inf')
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))
        self.t = 0.1  # threshold
        self.temperature = 20 # temp

        self.perceptor_model = perceptor
        self.replace_grad = replace_grad
        self.im_emb_type= im_emb_type

    def spatial_explainability_loss(self, input_image, mask, token_text):
        # t,c,h,w = input_image.size()
        # input_image= input_image.view(-1,c,h,w )
        batch_size = input_image.shape[0]
        text = token_text.repeat(batch_size, 1)
        # text = token_text.repeat(batch_size, 1)
        index = [i for i in range(batch_size)]
        logit_scale = self.perceptor_model.logit_scale.exp()
        self.perceptor_model.zero_grad()

        with torch.enable_grad():
            text_embedding = self.perceptor_model.encode_text(text)
            image_embedding = self.perceptor_model.encode_image(input_image)
            logits_per_image, logits_per_text = create_logits(image_embedding,text_embedding,logit_scale)
            # No need to do fusion here because it analyzes attention per frame not video
            # logits_per_image, logits_per_text = self.perceptor_model(input_image, text)
            # logits_per_image = logits_per_image
            # logits_per_image = logits_per_image / logit_scale

            one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
            one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.to(input_image.device) * logits_per_image)

            image_attn_blocks = list(dict(self.perceptor_model.visual.transformer.resblocks.named_children()).values())
            # print(dir(image_attn_blocks[0]))
            num_tokens = image_attn_blocks[0].attn.attn_output_weights.shape[-1]
            R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn.attn_output_weights.dtype).to(logits_per_image.device)
            R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)

            for blk_idx, blk in enumerate(image_attn_blocks):
                if blk_idx <= 10:
                    continue

                grad = torch.autograd.grad(one_hot, [blk.attn.attn_output_weights], retain_graph=True, create_graph=True)[0]

                # print("It worked")
                cam = blk.attn.attn_output_weights
                cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
                cam = cam.clamp(min=0).mean(dim=1)
                R = R + torch.bmm(cam, R)

            R[:, 0, 0] = 0
            image_relevance = R[:, 0, 1:]

            image_relevance = image_relevance.reshape(-1, 1, 7, 7)
            image_relevance = torch.nn.functional.interpolate(image_relevance, size=mask.shape[-1], mode='bicubic')
            image_relevance = image_relevance / torch.sum(image_relevance, dim=(-2, -1), keepdim=True)
            image_max = image_relevance.max(dim=-1, keepdim=True)[0]
            image_max = image_max.max(dim=-2, keepdim=True)[0]
            img_expl = image_relevance / image_max

            img_expl = (img_expl - self.t) * self.temperature
            binarized_expl = torch.sigmoid(img_expl)

            mask = mask.to(img_expl.device)
            intersection = (binarized_expl * mask).float().sum((-2, -1))
            union = (binarized_expl * (1 - mask)).float().sum((-2, -1))
            # import pdb; pdb.set_trace()
            mask_size = 1
            for dim in mask.shape:
                mask_size *= dim
            # mask_size = (mask.shape[0] * mask.shape[1] * mask.shape[2] * mask.shape[3])
            dice_loss = (2 * intersection / (2 * intersection + union))


        self.perceptor_model.zero_grad()
        return (-1) * dice_loss

    def forward(self, input_img, mask, dynamic_lambda, tokenized_text):
        masked_input = input_img * mask.to(input_img.device)
        masked_input = torch.cat([input_img, masked_input], dim=0)

        expl_loss = self.spatial_explainability_loss(input_img, mask, tokenized_text)
        expl_loss = expl_loss * dynamic_lambda

        image_embedding = self.perceptor_model.encode_image(masked_input)
        img_embed_float = image_embedding.float()
        input_normed = F.normalize(img_embed_float.unsqueeze(1), dim=2)

        text = tokenized_text.unsqueeze(0)
        text_embedding = self.perceptor_model.encode_text(text)
        embed_float = text_embedding.float()
        embed_normed = F.normalize(embed_float.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()

        loss = self.replace_grad(dists, torch.maximum(dists, self.stop)).mean() + expl_loss.mean()

        final_loss = self.weight.abs() * loss

        # Just the mask embedding
        if self.im_emb_type == 'just_image':
            image_embedding = image_embedding[0:8]
        elif self.im_emb_type == 'just_mask':
            image_embedding = image_embedding[8::]
        elif self.im_emb_type == 'average':
            # Average the masks and image embeddings
            image_embedding = image_embedding.view(8, 2, 512).mean(dim=1)
        
        return final_loss, text_embedding, image_embedding