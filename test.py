# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
import sys
sys.path.insert(0, "./../explainable_bounding_box/ml-no-token-left-behind/external/tamingtransformers/")
sys.path.append("./../explainable_bounding_box/ml-no-token-left-behind/external/TransformerMMExplainability/")
import CLIP.clip as clip
import torch.nn as nn
from datasets import EducationBBDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
import numpy
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import get_augmentation
import torch
from utils.Text_Prompt import *

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from torchvision import transforms
# from TSSTANET.tsstanet import tanet, sanet, stanet, stanet_af
import os
import random
import math
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from florence.configuration_florence2 import *
from florence.florence_attn import *
import florence.modeling_florence2 as flor2
from florence.processor import *

def unnormalize(bb, w, h):
    new_bb = []
    for b in bb:
        if b % 2 != 0:
            new_bb.append(math.ceil(b/1000)*h)
        else:
            new_bb.append((b/1000)*w)
    return new_bb

def normalize(bb, w, h):
    new_bb = []
    for b in bb:
        if b % 2 != 0:
            new_bb.append(math.ceil((b/h)*1000))
        else:
            new_bb.append(math.ceil((b/w)*1000))
    return new_bb

def plot_confusion_matrix(y_true, y_pred, classes, name,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix

    new_pred = []
    for i, (pred, gt) in enumerate(zip(y_pred, y_true)):
        if type(pred) == type(list()):
            if gt in pred:
                new_pred.append(gt)
            else:
                # if not just add the top-1 choice
                new_pred.append(pred[0])
        else:
            new_pred.append(pred)

    y_pred = new_pred
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    with open('confusion.txt', 'w') as f:
        for el in cm:
            for np_entry in el:
                f.write(f'{np_entry},')
            f.write('\n')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(f'{name}.png')
    plt.clf()

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)

def calculate_similarity(logits_per_image, b, num_text_aug):
    similarity = logits_per_image.view(b, num_text_aug, -1).softmax(dim=-1)
    similarity = similarity.mean(dim=1, keepdim=False)
    return similarity


def florence_bb(bbs, ff_images, REGION_TO_DESCRIPTION, processor, flo_model, config, device):
    
    # [16, 8, 4] = [B, T, C]
    prompts_r2d = []
    bbs = bbs[:, 0, :]
    width = ff_images[0].size[0]
    height = ff_images[0].size[1]
    for bb in bbs:
        norm_bb = normalize(bb.tolist(), width, height)
    
        prompt_r2d = f'<{REGION_TO_DESCRIPTION}>'
        for dim in norm_bb:
            prompt_r2d += f'<loc_{dim}>'
        prompts_r2d.append(prompt_r2d)
    
    text_dict = {
        REGION_TO_DESCRIPTION: []
    }

    # if config.data.weights.lambda_bb > 0 and config.data.weights.lambda_ff > 0:
    # tasks = [(CAPTION, prompt_cap, False), (REGION_TO_DESCRIPTION, prompts_r2d, True)]
    # elif config.data.weights.lambda_bb > 0:
    #     tasks = [(REGION_TO_DESCRIPTION, prompts_r2d, True)]
    # elif config.data.weights.lambda_bb > 0:
    #     tasks = [(CAPTION, prompt_cap, False)]

    task, prompt, padding = (REGION_TO_DESCRIPTION, prompts_r2d, True)
    generated_text = call_florence(prompt, ff_images, flo_model, processor, padding)
    
    for this_text, this_prompt in zip(generated_text, prompt):
        parsed_answer = processor.post_process_generation(
            this_text,
            task=this_prompt,
            image_size=(width, height)
        )
        text_dict[task].append(parsed_answer[this_prompt])

    bounded_text = text_dict[REGION_TO_DESCRIPTION]
    return bounded_text

def call_florence(prompt, cropped_images, model, processor, padding=False):
    if hasattr(model, 'module'):
        flo_model = model.module
    else:
        flo_model = model

    inputs = processor(text=prompt, images=cropped_images, return_tensors="pt", padding=padding)

    input_ids = inputs["input_ids"].to(flo_model.device)
    pixel_values = inputs["pixel_values"].to(flo_model.device)
    generated_ids = flo_model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=512,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text

def run_region_desc_florence(aug_images, bbs, model, processor, real_texts, config, testing, with_prompt=True):
    # Just choose one frame to create a caption for
    if not config.data.florence.random_frame:
        ff_images = [transforms.ToPILImage()(images[0]) for images in aug_images]
    else:
        ff_images = [transforms.ToPILImage()(images[random.randint(0, len(images)-1)]) for images in aug_images]
    bounded_text = []
    # Create caption prompts
    CAPTION = config.data.florence.caption_type
    prompt_cap = [f'<{CAPTION}>' for _ in ff_images]
    REGION_TO_DESCRIPTION = 'REGION_TO_DESCRIPTION'
    bounded_text = florence_bb(
                            bbs, 
                            ff_images, 
                            REGION_TO_DESCRIPTION, 
                            processor, 
                            flo_model=model.module, 
                            config=config, 
                            device=model.module.device)
    return concat_and_tokenize(bounded_text, real_texts=real_texts, testing=testing, with_prompt=with_prompt)

def run_caption_florence(aug_images, model, processor, real_texts, config, testing=False, with_prompt=True):
    # Just choose one frame to create a caption for
    if not config.data.florence.random_frame:
        ff_images = [transforms.ToPILImage()(images[0]) for images in aug_images]
    else:
        ff_images = [transforms.ToPILImage()(images[random.randint(0, len(images)-1)]) for images in aug_images]
    # Create caption prompts
    if config.data.florence.caption_level == 0:
        CAPTION = 'CAPTION'
    elif config.data.florence.caption_level == 1:
        CAPTION = 'DETAILED_CAPTION'
    elif config.data.florence.caption_level == 2:
        CAPTION = 'MORE_DETAILED_CAPTION'
    prompt_cap = [f'<{CAPTION}>' for _ in ff_images]
    generated_text = call_florence(prompt_cap, ff_images, model, processor)
    return concat_and_tokenize(generated_text, real_texts=real_texts, testing=testing, with_prompt=with_prompt)

def run_florence(aug_images, processor, model, real_texts, config, caption=True, bbs=None, label=None, text_input=None, debug=False, testing=False, with_prompt=True):

    ff_texts = None
    if caption:
        ff_texts = run_caption_florence(aug_images, model, processor, real_texts, config, testing, with_prompt=with_prompt)
    else:
        # FIXME
        ff_texts = torch.zeros(32, 512)
        ff_texts = ff_texts.to(device=model.module.device)
    
    if bbs != None:
        bounded_texts = run_region_desc_florence(aug_images, bbs, model, processor, real_texts, config, testing, with_prompt=with_prompt)
        return ff_texts, bounded_texts
    else:
        return ff_texts

def concat_and_tokenize(generated_text, real_texts = None, with_prompt=True, testing=False):
    if testing or not with_prompt:
        generated_text = [text.replace('<s>', '').replace('</s>', '') for text in generated_text]
        final_text = torch.stack([clip.tokenize(text) for text in generated_text])
        return final_text.squeeze(dim=1)
    else:
        generated_text = [prompt_text.capitalize() + ": " + text.replace('<s>', '').replace('</s>', '') for text, prompt_text in zip(generated_text, real_texts)]
        final_text = torch.stack([clip.tokenize(text) for text in generated_text])
        return final_text.squeeze(dim=1)
        

def validate(epoch, val_loader, classes, class_text, device, clip_model, fusion_model, config, num_text_aug, processor, flo_model, text_aug_dict, lambda_bb, lambda_ff, lambda_enff, lambda_enbb, print_metrics=False):
    clip_model.eval()
    fusion_model.eval()
    flo_model.eval()
    num = 0
    corr_1 = 0
    corr_k = 0
    corr2_1 = 0
    corr2_3 = 0
    corr3_1 = 0
    corr3_3 = 0

    definitions = [
        "A book is used or held by teacher or student",
        "Teacher is sitting (chair, stool, floor, crouching, on desk, kneeling)",
        "Teacher is standing (in generally the same spot to maintain the same orientation to students)",
        "Teacher is walking with purpose to change orientation to students",
        "A tangible object (ruler, math manipulative; anything in someone's hand other than what is already listed; does not include pencil/pen) is used or held by teacher or student for instructional purposes",
        "A worksheet is used or held by teacher or student"
    ]
    use_definitions = config.data.use_definitions

    labeled_ids = []
    correct_ids = []
    classes_aug = [v for k, v in text_aug_dict.items()]
    if use_definitions:
        classes = concat_and_tokenize(definitions, testing=True)
    similarities_image = []
    similarities_text  = []
    # if print_metrics:
    #     f = open('similarity.txt', 'w')
    #     f.write('correct,i_label,t_label,it_label,i_top3,t_top3,it_top3\n')
    with torch.no_grad():
        for iii, data in enumerate(tqdm(val_loader)):
            
            bb_video, ff_video, nonaug_images, bbs, class_id = data
            ff_video = ff_video.to(device)
            class_id = class_id.to(device)
            bbs = bbs.to(device)
            nonaug_images = nonaug_images.to(device)
            bb_video = bb_video.view((-1,config.data.num_segments,3)+bb_video.size()[-2:])

            if not config.data.florence.activate:
                # Original ActionCLIP
                text_inputs = classes.to(device)
                ff_en_text_features = clip_model.encode_text(text_inputs)
            elif config.data.florence.use_bounded_text:
                # With bounding box text content
                # if ff_caption == None:
                caption = False
                if lambda_enff > 0:
                    caption = True
                final_text, bounded_text = run_florence(nonaug_images, processor, flo_model, class_text, config, bbs=bbs, caption=caption, testing=True)
                # else:
                #     final_text = concat_and_tokenize(ff_caption, real_texts=class_text, testing=True)
                #     bounded_text = run_region_desc_florence(nonaug_images, bbs, flo_model, processor, class_text, config, testing=True)
                
                final_text = final_text.to(device)
                bounded_text = bounded_text.to(device)
                
                text_inputs = classes.to(device)
                generic_text_features = clip_model.encode_text(text_inputs)
                if lambda_enff > 0 and lambda_enbb > 0:
                    ff_en_text_features = clip_model.encode_text(final_text)
                    bb_en_text_features = clip_model.encode_text(bounded_text)
                elif lambda_enff > 0:
                    ff_en_text_features = clip_model.encode_text(final_text)
                elif lambda_enbb > 0:
                    bb_en_text_features = clip_model.encode_text(bounded_text)
            else:
                # No bounding box text content
                # if ff_caption == None:
                caption = False
                if lambda_enff > 0:
                    caption = True
                final_text = run_florence(nonaug_images, processor, flo_model, class_text, config, caption=caption, testing=True)
                # else:
                #     final_text = concat_and_tokenize(ff_caption, real_texts =class_text, testing=True)

                final_text = final_text.to(device)
                text_inputs = classes.to(device)
                
                ff_en_text_features = clip_model.encode_text(final_text)
                generic_text_features = clip_model.encode_text(text_inputs)

            # image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = bb_video.size()
            class_id = class_id.to(device)

            # Bounding Box Video and Full frame Video
            bb_image_input = bb_video.to(device).view(-1, c, h, w)
            bb_image_features = clip_model.encode_image(bb_image_input)
            bb_image_features = bb_image_features.view(b,t,-1)
            bb_image_features = fusion_model(bb_image_features)

            ff_video = ff_video.squeeze(dim=1)
            ff_image_input = ff_video.to(device).view(-1, c, h, w)
            ff_image_features = clip_model.encode_image(ff_image_input)
            ff_image_features = ff_image_features.view(b,t,-1)
            ff_image_features = fusion_model(ff_image_features)
            
            if config.data.loss_type == 0:
                if lambda_enbb > 0 and lambda_enff > 0:
                    # ffimg bbimg fftxt bbtxt
                    if lambda_ff > 0 and lambda_bb > 0:

                        # Bounding Box Video and Full frame Video
                        bb_image_input = bb_video.to(device).view(-1, c, h, w)
                        bb_image_features = clip_model.encode_image(bb_image_input)
                        bb_image_features = bb_image_features.view(b,t,-1)
                        bb_image_features = fusion_model(bb_image_features)

                        ff_video = ff_video.squeeze(dim=1)
                        ff_image_input = ff_video.to(device).view(-1, c, h, w)
                        ff_image_features = clip_model.encode_image(ff_image_input)
                        ff_image_features = ff_image_features.view(b,t,-1)
                        ff_image_features = fusion_model(ff_image_features)

                        # Normalize
                        bb_image_features /= bb_image_features.norm(dim=-1, keepdim=True)
                        ff_image_features /= ff_image_features.norm(dim=-1, keepdim=True)
                        
                        testing_features = generic_text_features 
                        testing_features /= testing_features.norm(dim=-1, keepdim=True)

                        # Create Logits
                        bb_logits_per_image = (100.0 * bb_image_features @ testing_features.T)
                        ff_logits_per_image = (100.0 * ff_image_features @ testing_features.T)
                    
                        # Decision making
                        similarity_image = calculate_similarity(bb_logits_per_image, b, num_text_aug)
                        similarity_image = similarity_image + calculate_similarity(ff_logits_per_image, b, num_text_aug)
                    # ffimg fftxt bbtxt
                    elif lambda_ff > 0:

                        ff_video = ff_video.squeeze(dim=1)
                        ff_image_input = ff_video.to(device).view(-1, c, h, w)
                        ff_image_features = clip_model.encode_image(ff_image_input)
                        ff_image_features = ff_image_features.view(b,t,-1)
                        ff_image_features = fusion_model(ff_image_features)

                        # Normalize
                        ff_image_features /= ff_image_features.norm(dim=-1, keepdim=True)
                        
                        testing_features = generic_text_features 
                        testing_features /= testing_features.norm(dim=-1, keepdim=True)

                        # Create Logits
                        ff_logits_per_image = (100.0 * ff_image_features @ testing_features.T)
                    
                        # Decision making
                        similarity_image = calculate_similarity(ff_logits_per_image, b, num_text_aug)
                        
                    # bbimg fftxt bbtxt
                    elif lambda_bb > 0:
                        # Bounding Box Video and Full frame Video
                        bb_image_input = bb_video.to(device).view(-1, c, h, w)
                        bb_image_features = clip_model.encode_image(bb_image_input)
                        bb_image_features = bb_image_features.view(b,t,-1)
                        bb_image_features = fusion_model(bb_image_features)

                        # Normalize
                        bb_image_features /= bb_image_features.norm(dim=-1, keepdim=True)
                        
                        testing_features = generic_text_features 
                        testing_features /= testing_features.norm(dim=-1, keepdim=True)

                        # Create Logits
                        bb_logits_per_image = (100.0 * bb_image_features @ testing_features.T)
                    
                        # Decision making
                        similarity_image = calculate_similarity(bb_logits_per_image, b, num_text_aug)
                    else:
                        raise ValueError("Error!")
                elif lambda_enff > 0:
                    # ffimg bbimg fftxt
                    if lambda_ff > 0 and lambda_bb > 0:

                        ff_video = ff_video.squeeze(dim=1)
                        ff_image_input = ff_video.to(device).view(-1, c, h, w)
                        ff_image_features = clip_model.encode_image(ff_image_input)
                        ff_image_features = ff_image_features.view(b,t,-1)
                        ff_image_features = fusion_model(ff_image_features)

                        # Normalize
                        ff_image_features /= ff_image_features.norm(dim=-1, keepdim=True)
                        
                        testing_features = generic_text_features 
                        testing_features /= testing_features.norm(dim=-1, keepdim=True)

                        # Create Logits
                        ff_logits_per_image = (100.0 * ff_image_features @ testing_features.T)
                    
                        # Decision making
                        similarity_image = calculate_similarity(ff_logits_per_image, b, num_text_aug)

                    # ffimg fftxt
                    elif lambda_ff > 0:

                        ff_video = ff_video.squeeze(dim=1)
                        ff_image_input = ff_video.to(device).view(-1, c, h, w)
                        ff_image_features = clip_model.encode_image(ff_image_input)
                        ff_image_features = ff_image_features.view(b,t,-1)
                        ff_image_features = fusion_model(ff_image_features)

                        # Normalize
                        ff_image_features /= ff_image_features.norm(dim=-1, keepdim=True)
                        
                        testing_features = generic_text_features 
                        testing_features /= testing_features.norm(dim=-1, keepdim=True)

                        # Create Logits
                        ff_logits_per_image = (100.0 * ff_image_features @ testing_features.T)
                    
                        # Decision making
                        similarity_image = calculate_similarity(ff_logits_per_image, b, num_text_aug)

                    # bbimg fftxt
                    elif lambda_bb > 0:

                        # Bounding Box Video and Full frame Video
                        bb_image_input = bb_video.to(device).view(-1, c, h, w)
                        bb_image_features = clip_model.encode_image(bb_image_input)
                        bb_image_features = bb_image_features.view(b,t,-1)
                        bb_image_features = fusion_model(bb_image_features)

                        # Normalize
                        bb_image_features /= bb_image_features.norm(dim=-1, keepdim=True)
                        
                        testing_features = generic_text_features 
                        testing_features /= testing_features.norm(dim=-1, keepdim=True)

                        # Create Logits
                        bb_logits_per_image = (100.0 * bb_image_features @ testing_features.T)
                    
                        # Decision making
                        similarity_image = calculate_similarity(bb_logits_per_image, b, num_text_aug)
                    else:
                        raise ValueError("Error!")
                elif lambda_enbb > 0:
                    # ffimg bbimg bbtxt
                    if lambda_ff > 0 and lambda_bb > 0:

                        # Bounding Box Video and Full frame Video
                        bb_image_input = bb_video.to(device).view(-1, c, h, w)
                        bb_image_features = clip_model.encode_image(bb_image_input)
                        bb_image_features = bb_image_features.view(b,t,-1)
                        bb_image_features = fusion_model(bb_image_features)

                        # Normalize
                        bb_image_features /= bb_image_features.norm(dim=-1, keepdim=True)
                        
                        testing_features = generic_text_features 
                        testing_features /= testing_features.norm(dim=-1, keepdim=True)

                        # Create Logits
                        bb_logits_per_image = (100.0 * bb_image_features @ testing_features.T)
                    
                        # Decision making
                        similarity_image = calculate_similarity(bb_logits_per_image, b, num_text_aug)

                    # ffimg bbtxt
                    elif lambda_ff > 0:

                        ff_video = ff_video.squeeze(dim=1)
                        ff_image_input = ff_video.to(device).view(-1, c, h, w)
                        ff_image_features = clip_model.encode_image(ff_image_input)
                        ff_image_features = ff_image_features.view(b,t,-1)
                        ff_image_features = fusion_model(ff_image_features)

                        # Normalize
                        ff_image_features /= ff_image_features.norm(dim=-1, keepdim=True)
                        
                        testing_features = generic_text_features 
                        testing_features /= testing_features.norm(dim=-1, keepdim=True)

                        # Create Logits
                        ff_logits_per_image = (100.0 * ff_image_features @ testing_features.T)
                    
                        # Decision making
                        similarity_image = calculate_similarity(ff_logits_per_image, b, num_text_aug)
                    # bbimg bbtxt
                    elif lambda_bb > 0:

                        # Bounding Box Video and Full frame Video
                        bb_image_input = bb_video.to(device).view(-1, c, h, w)
                        bb_image_features = clip_model.encode_image(bb_image_input)
                        bb_image_features = bb_image_features.view(b,t,-1)
                        bb_image_features = fusion_model(bb_image_features)

                        # Normalize
                        bb_image_features /= bb_image_features.norm(dim=-1, keepdim=True)
                        
                        testing_features = generic_text_features 
                        testing_features /= testing_features.norm(dim=-1, keepdim=True)

                        # Create Logits
                        bb_logits_per_image = (100.0 * bb_image_features @ testing_features.T)
                    
                        # Decision making
                        similarity_image = calculate_similarity(bb_logits_per_image, b, num_text_aug)
                    else:
                        raise ValueError("Error!")
                else:
                    raise ValueError("Error!")
            else:    
                # if config.data.weighted_features.use:
                # Fuse
                # if config.data.weighted_features.learned:
                #     image_features = lambda_bb.to(dtype=bb_image_features.dtype)*bb_image_features + lambda_ff.to(dtype=ff_image_features.dtype)*ff_image_features
                #     text_features = lambda_en.to(dtype=bb_image_features.dtype)*ff_en_text_features + lambda_enbb.to(dtype=ff_en_text_features.dtype)*generic_text_features
                # else:
                if lambda_ff > 0 and lambda_bb > 0:
                    image_features = lambda_bb*bb_image_features + lambda_ff*ff_image_features
                elif lambda_ff > 0:
                    image_features = lambda_ff*ff_image_features
                elif lambda_bb > 0:
                    image_features = lambda_bb*bb_image_features
                
                if lambda_enff > 0 and lambda_enbb > 0:
                    text_features = lambda_enff*ff_en_text_features + lambda_enbb*bb_en_text_features
                elif lambda_enff > 0:
                    text_features = lambda_enff*ff_en_text_features
                elif lambda_enbb > 0:
                    text_features = lambda_enbb*bb_en_text_features

                # Normalize
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                if config.data.florence.activate:
                    testing_features = generic_text_features 
                    testing_features /= testing_features.norm(dim=-1, keepdim=True)
                    # logits_per_text = (100.0 * text_features @ testing_features.T)
                    # similarity_text = calculate_similarity(logits_per_text, b, num_text_aug) # B x L
                else:
                    testing_features = ff_en_text_features 
                    testing_features /= testing_features.norm(dim=-1, keepdim=True)
                # Create Logits
                logits_per_image = (100.0 * image_features @ testing_features.T)
                # Decision making
                # raise ValueError(definitions, use_definitions, image_features.shape, testing_features.shape, logits_per_image.shape, b, num_text_aug)
                if use_definitions:
                    similarity_image = logits_per_image.view(b, -1).softmax(dim=-1)
                else:
                    similarity_image = calculate_similarity(logits_per_image, b, num_text_aug) # B x L
                # similarities_image.append(similarity_image)
                # similarities_text.append(similarity_text)
            # else:
            #     # Normalize
            #     bb_image_features /= bb_image_features.norm(dim=-1, keepdim=True)
            #     ff_image_features /= ff_image_features.norm(dim=-1, keepdim=True)
                
            #     # Create Logits
            #     bb_logits_per_image = (100.0 * bb_image_features @ ff_en_text_features.T)
            #     ff_logits_per_image = (100.0 * ff_image_features @ bb_text_features.T)
            
            #     # Decision making
            #     similarity = calculate_similarity(bb_logits_per_image, b, num_text_aug)
            #     similarity = similarity + calculate_similarity(ff_logits_per_image, b, num_text_aug)

            values_1, indices_1 = similarity_image.topk(1, dim=-1)
            values_k, indices_k = similarity_image.topk(config.data.k, dim=-1)
            # if config.data.florence.activate:
            #     values2_1, indices2_1 = similarity_text.topk(1, dim=-1)
            #     values2_3, indices2_3 = similarity_text.topk(3, dim=-1)
            #     values3_1, indices3_1 = (similarity_image+.5*similarity_text).topk(1, dim=-1)
            #     values3_3, indices3_3 = (similarity_image+.5*similarity_text).topk(3, dim=-1)
            num += b
            for i in range(b):
                if indices_1[i] == class_id[i]:
                    corr_1 += 1
                if class_id[i] in indices_k[i]:
                    corr_k += 1
                
            # yhat = torch.argmax(similarity_image, dim=1).to(dtype=int)
            labeled_ids.append(indices_k)
            correct_ids.extend(class_id.tolist())

    labeled_ids = torch.cat(labeled_ids).tolist()
    if print_metrics:
        # print('TEXT     Epoch: [{}/{}]: Top1: {}, Top3: {}'.format(epoch, config.solver.epochs, float(corr2_1)/ num * 100, float(corr2_3)/num*100))
        # print('COMBINED Epoch: [{}/{}]: Top1: {}, Top3: {}'.format(epoch, config.solver.epochs, float(corr3_1)/ num * 100, float(corr3_3)/num*100))
        plot_confusion_matrix(correct_ids, labeled_ids, np.array(["Using a Book", "Teacher Sitting", "Teacher Standing", "Teacher Writing", "Using Technology", "Using a Worksheet"]), config.test_name)
        # f.close()

    top1 = float(corr_1) / num * 100
    topk = float(corr_k) / num * 100
    wandb.log({"top1": top1})
    wandb.log({"topk": topk})
    print('Epoch: [{}/{}]: Top1: {}, Top{}: {}'.format(epoch, config.solver.epochs, top1, config.data.k, topk))
    return top1

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)
    wandb.init(project=config['network']['type'],
               name='{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'],
                                         config['data']['dataset']))
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('test.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    clip_model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32
    flo_model, processor = flor2.load("BASE_FT", device)

    transform_val = get_augmentation(False, config)

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    model_text = TextCLIP(clip_model)
    model_image = ImageCLIP(clip_model)

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    flo_model = torch.nn.DataParallel(flo_model).cuda()
    wandb.watch(clip_model)
    wandb.watch(fusion_model)

    def collate_fn(batch):
        cropped_videos, images, bbs, generated_images, ff_caption, labels = zip(*batch)
        # Check the labels for bb
        cropped_videos = torch.stack(cropped_videos) 
        images = torch.stack(images) 
        labels = torch.tensor(labels)
        generated_images = torch.stack(generated_images)
        bbs = torch.stack(bbs)
        return cropped_videos, images, generated_images, bbs, labels

    transform_val = get_augmentation(False,config)
    val_data = EducationBBDataset(
                    config.data.val_list,
                    config.data.label_list, 
                    random_shift=False,
                    num_segments=config.data.num_segments,
                    image_tmpl=config.data.image_tmpl,
                    transform=transform_val,
                    label_box=config.data.label_box,
                    caption_level=config.data.florence.caption_level)
    val_loader = DataLoader(
                    val_data,
                    batch_size=config.data.batch_size,
                    num_workers=config.data.workers,
                    shuffle=False,
                    pin_memory=False,
                    drop_last=True,
                    collate_fn=collate_fn)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(
            model_text)  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            clip_model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict, text_aug_dict, class_text = text_prompt(val_data, config.prompt)

    best_prec1 = 0.0
    lambda_bb = config.data.weights.lambda_bb
    lambda_ff = config.data.weights.lambda_ff
    lambda_enff = config.data.weights.lambda_enff
    lambda_enbb = config.data.weights.lambda_enbb
    prec1 = validate(
                     start_epoch, 
                     val_loader, 
                     classes, 
                     class_text, 
                     device, 
                     clip_model, 
                     fusion_model, 
                     config, 
                     num_text_aug, 
                     processor, 
                     flo_model, 
                     text_aug_dict, 
                     lambda_bb, 
                     lambda_ff, 
                     lambda_enff, 
                     lambda_enbb,
                     print_metrics = True
            )

if __name__ == '__main__':
    main()
