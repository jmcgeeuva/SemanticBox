# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
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
from modules.Visual_Prompt import visual_prompt
from utils.KLLoss import KLLoss
from test import validate, calculate_similarity, TextCLIP, ImageCLIP
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import  *
import sys
sys.path.insert(0, "../explainable_bounding_box/ml-no-token-left-behind/external/tamingtransformers/")
sys.path.append("./../explainable_bounding_box/ml-no-token-left-behind/external/TransformerMMExplainability/")
from helpers import *
import random

import torch
import itertools

from florence.configuration_florence2 import *
from florence.florence_attn import *
import florence.modeling_florence2 as flor2
from florence.processor import *

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

############################ FLORENCE ###############################

def normalize(bb, w, h):
    new_bb = []
    for i, b in enumerate(bb):
        if i % 2 != 0:
            new_bb.append(math.ceil((b/h)*1000))
        else:
            new_bb.append(math.ceil((b/w)*1000))
    return new_bb

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

def concat_and_tokenize(generated_text, real_texts = None, with_prompt=True, testing=False):
    if testing or not with_prompt:
        generated_text = [text.replace('<s>', '').replace('</s>', '') for text in generated_text]
        final_text = torch.stack([clip.tokenize(text) for text in generated_text])
        return final_text.squeeze(dim=1)
    else:
        generated_text = [prompt_text.capitalize() + ": " + text.replace('<s>', '').replace('</s>', '') for text, prompt_text in zip(generated_text, real_texts)]
        final_text = torch.stack([clip.tokenize(text) for text in generated_text])
        return final_text.squeeze(dim=1)

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
    else:
        raise ValueError(f'{config.data.florence.caption_level} is an invalid caption_level in the configuration file')
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

############################ FLORENCE ###############################

def train_classifier(
        # Models
        ff_image_clip, 
        ff_text_clip, 
        ff_temp_trans,
        flo_model,
        flo_processor,
        clip_perceptor,
        
        # Data Loaders 
        train_loader, 
        val_loader,
        
        # Miscelaneous configurations
        start_epoch, 
        device,
        config,
        working_dir, 
        
        # Loss
        loss_img,
        loss_txt,
        
        # Tokenized Text Dictionaries
        text_dict,
        text_aug_dict,
        num_text_aug, 
        classes,
        class_text
    ):
    best_prec1 = -1
    cross_entropy = nn.CrossEntropyLoss()
    clamp_with_grad = ClampWithGrad.apply
        
    if config.data.weighted_features.use:
        # if config.data.weighted_features.learned:
        #     lambda_bb = nn.Parameter(torch.ones(config.data.batch_size, 512, requires_grad=True, device=device))
        #     lambda_ff = nn.Parameter(torch.ones(config.data.batch_size, 512, requires_grad=True, device=device))
        #     lambda_enff = nn.Parameter(torch.ones(config.data.batch_size, 512, requires_grad=True, device=device))
        #     lambda_enbb = nn.Parameter(torch.ones(config.data.batch_size, 512, requires_grad=True, device=device))
        #     lambda_bb = lambda_bb.to(device=device, dtype=torch.float32)
        #     lambda_ff = lambda_ff.to(device=device, dtype=torch.float32)
        #     lambda_enff = lambda_enff.to(device=device, dtype=torch.float32)
        #     lambda_enbb = lambda_enbb.to(device=device, dtype=torch.float32)
        #     optimizer = _optimizer(config, clip_perceptor, ff_temp_trans, [lambda_bb, lambda_ff, lambda_enff, lambda_enbb])
        #     lr_scheduler = _lr_scheduler(config, optimizer)
        # else:
        lambda_bb = config.data.weights.lambda_bb
        lambda_ff = config.data.weights.lambda_ff
        lambda_enff = config.data.weights.lambda_enff
        lambda_enbb = config.data.weights.lambda_enbb
        optimizer = _optimizer(config, clip_perceptor, ff_temp_trans)
        lr_scheduler = _lr_scheduler(config, optimizer)
    else:
        optimizer = _optimizer(config, clip_perceptor, ff_temp_trans)
        lr_scheduler = _lr_scheduler(config, optimizer)
    

    ################### Train Classifier ####################################
    # scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, config.solver.epochs):
        ff_image_clip.train()
        ff_text_clip.train()
        ff_temp_trans.train()
        # flo_model.train()

        running_kl = 0
        running_ce = 0
        running_flo = 0
        total_loss = 0

        for kkk,data in enumerate(tqdm(train_loader)):
            if config.solver.type != 'monitor':
                if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            bb_video, ff_video, nonaug_images, bbs, list_id = data
            
            # if not config.data.florence.preset:
            #     ff_caption = None

            bb_video = bb_video.to(device)
            ff_video = ff_video.to(device)
            nonaug_images = nonaug_images.to(device)
            bbs = bbs.to(device)
            list_id = list_id.to(device)

            bb_video = bb_video.view((-1,config.data.num_segments,3)+bb_video.size()[-2:])
            b,t,c,h,w = bb_video.size()
            bb_video= bb_video.view(-1,c,h,w ) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            
            text_id = numpy.random.randint(num_text_aug,size=len(list_id))
            token_texts = torch.stack([text_dict[j][i,:] for i,j in zip(list_id,text_id)])
            real_texts = [text_aug_dict[j][i] for i,j in zip(list_id,text_id)]

            if not config.data.florence.activate:
                texts = token_texts.to(device)
            elif config.data.florence.use_bounded_text:
                with torch.no_grad():
                    # if ff_caption == None:
                    caption = False
                    if lambda_enff > 0:
                        caption = True
                    final_text, bounded_text = run_florence(nonaug_images, flo_processor, flo_model, real_texts, config, caption=caption, bbs=bbs, with_prompt=config.data.florence.with_prompt)
                    # else:
                    #     final_text = concat_and_tokenize(ff_caption, real_texts=real_texts, testing=True)
                    #     # bounding boxes need to be labeled on the fly so it is a little slower
                    #     bounded_texts = run_region_desc_florence(nonaug_images, bbs, flo_model, flo_processor, real_texts, config)
                    texts  = final_text.to(device)
                    bounded_texts = bounded_text.to(device)
            else:
                with torch.no_grad():
                    # if ff_caption == None:
                    caption = False
                    if lambda_enff > 0:
                        caption = True
                    final_text = run_florence(nonaug_images, flo_processor, flo_model, real_texts, config, caption=caption, with_prompt=config.data.florence.with_prompt)
                    # else:
                    #     final_text = concat_and_tokenize(ff_caption, real_texts =real_texts, testing=True)
                    texts  = final_text.to(device)

            ground_truth = torch.tensor(gen_label(list_id),device=device) 
            logit_scale = clip_perceptor.logit_scale.exp()
            use_generic = False
            if config.data.loss_type == 0:

                if lambda_enbb > 0 and lambda_enff > 0:
                    # ffimg bbimg fftxt bbtxt
                    if lambda_ff > 0 and lambda_bb > 0:
                        ff_gen_text_embedding = ff_text_clip(texts)
                        ff_video_tensor = ff_video.view(-1,c,h,w )
                        ff_image_embedding = ff_image_clip(ff_video_tensor)
                        ff_image_embedding = ff_image_embedding.view(b,t,-1)
                        ff_image_embedding = ff_temp_trans(ff_image_embedding)
                        ff_logits_per_image, ff_logits_per_text = create_logits(ff_image_embedding,ff_gen_text_embedding,logit_scale)
                        ff_loss_imgs = loss_img(ff_logits_per_image,ground_truth.to(dtype=ff_logits_per_image.dtype))
                        ff_loss_texts = loss_txt(ff_logits_per_text,ground_truth.to(dtype=ff_logits_per_text.dtype))

                        bb_gen_text_embedding = ff_text_clip(bounded_texts)
                        bb_video_tensor = bb_video.view(-1,c,h,w )
                        bb_image_embedding = ff_image_clip(bb_video_tensor)
                        bb_image_embedding = bb_image_embedding.view(b,t,-1)
                        bb_image_embedding = ff_temp_trans(bb_image_embedding)
                        bb_logits_per_image, bb_logits_per_text = create_logits(bb_image_embedding,bb_gen_text_embedding,logit_scale)
                        bb_loss_imgs = loss_img(bb_logits_per_image, ground_truth.to(dtype=bb_logits_per_image.dtype))
                        bb_loss_texts = loss_txt(bb_logits_per_text,ground_truth.to(dtype=bb_logits_per_text.dtype))

                        kl_loss = config.loss.image*((ff_loss_imgs + ff_loss_texts)/2) + config.loss.text*((bb_loss_imgs + bb_loss_texts)/2)
                    
                    # ffimg fftxt bbtxt
                    elif lambda_ff > 0:
                        ff_gen_text_embedding = ff_text_clip(texts)
                        ff_video_tensor = ff_video.view(-1,c,h,w )
                        ff_image_embedding = ff_image_clip(ff_video_tensor)
                        ff_image_embedding = ff_image_embedding.view(b,t,-1)
                        ff_image_embedding = ff_temp_trans(ff_image_embedding)
                        ff_logits_per_image, ff_logits_per_text = create_logits(ff_image_embedding,ff_gen_text_embedding,logit_scale)
                        ff_loss_imgs = loss_img(ff_logits_per_image,ground_truth.to(dtype=ff_logits_per_imag.dtype))
                        ff_loss_texts = loss_txt(ff_logits_per_text,ground_truth.to(dtype=ff_logits_per_tex.dtype))

                        image_embedding = ff_image_embedding

                        bb_gen_text_embedding = ff_text_clip(bounded_texts)
                        bb_logits_per_image, bb_logits_per_text = create_logits(image_embedding,bb_gen_text_embedding,logit_scale)
                        bb_loss_imgs = loss_img(bb_logits_per_image, ground_truth.to(dtype=bb_logits_per_image.dtype))
                        bb_loss_texts = loss_txt(bb_logits_per_text,ground_truth.to(dtype=bb_logits_per_text.dtype))


                        kl_loss = config.loss.image*((ff_loss_imgs + ff_loss_texts)/2) + config.loss.text*((bb_loss_imgs + bb_loss_texts)/2)
                    # bbimg fftxt bbtxt
                    elif lambda_bb > 0:
                        bb_gen_text_embedding = ff_text_clip(bounded_texts)
                        bb_video_tensor = bb_video.view(-1,c,h,w )
                        bb_image_embedding = ff_image_clip(bb_video_tensor)
                        bb_image_embedding = bb_image_embedding.view(b,t,-1)
                        bb_image_embedding = ff_temp_trans(bb_image_embedding)
                        bb_logits_per_image, bb_logits_per_text = create_logits(bb_image_embedding,bb_gen_text_embedding,logit_scale)
                        bb_loss_imgs = loss_img(bb_logits_per_image, ground_truth.to(dtype=bb_logits_per_image.dtype))
                        bb_loss_texts = loss_txt(bb_logits_per_text,ground_truth.to(dtype=bb_logits_per_text.dtype))

                        image_embedding = bb_image_embedding
                        
                        ff_gen_text_embedding = ff_text_clip(texts)
                        ff_logits_per_image, ff_logits_per_text = create_logits(image_embedding,ff_gen_text_embedding,logit_scale)
                        ff_loss_imgs = loss_img(ff_logits_per_image,ground_truth.to(dtype=ff_logits_per_imag.dtype))
                        ff_loss_texts = loss_txt(ff_logits_per_text,ground_truth.to(dtype=ff_logits_per_tex.dtype))


                        kl_loss = config.loss.image*((ff_loss_imgs + ff_loss_texts)/2) + config.loss.text*((bb_loss_imgs + bb_loss_texts)/2)
                    else: 
                        raise ValueError("Error!")
                elif lambda_enff > 0:
                    # ffimg bbimg fftxt
                    if lambda_ff > 0 and lambda_bb > 0:
                        
                        ff_gen_text_embedding = ff_text_clip(texts)
                        ff_video_tensor = ff_video.view(-1,c,h,w )
                        ff_image_embedding = ff_image_clip(ff_video_tensor)
                        ff_image_embedding = ff_image_embedding.view(b,t,-1)
                        ff_image_embedding = ff_temp_trans(ff_image_embedding)
                        ff_logits_per_image, ff_logits_per_text = create_logits(ff_image_embedding,ff_gen_text_embedding,logit_scale)
                        ff_loss_imgs = loss_img(ff_logits_per_image,ground_truth.to(dtype=ff_logits_per_imag.dtype))
                        ff_loss_texts = loss_txt(ff_logits_per_text,ground_truth.to(dtype=ff_logits_per_tex.dtype))

                        if use_generic:
                            text_embedding = generic_embedding
                        else:
                            text_embedding = ff_gen_text_embedding

                        bb_video_tensor = bb_video.view(-1,c,h,w )
                        bb_image_embedding = ff_image_clip(bb_video_tensor)
                        bb_image_embedding = bb_image_embedding.view(b,t,-1)
                        bb_image_embedding = ff_temp_trans(bb_image_embedding)
                        bb_logits_per_image, bb_logits_per_text = create_logits(bb_image_embedding,text_embedding,logit_scale)
                        bb_loss_imgs = loss_img(bb_logits_per_image, ground_truth.to(dtype=bb_logits_per_image.dtype))
                        bb_loss_texts = loss_txt(bb_logits_per_text,ground_truth.to(dtype=bb_logits_per_text.dtype))

                        kl_loss = config.loss.image*((ff_loss_imgs + ff_loss_texts)/2) + config.loss.text*((bb_loss_imgs + bb_loss_texts)/2)
                    # ffimg fftxt
                    elif lambda_ff > 0:
                        
                        ff_gen_text_embedding = ff_text_clip(texts)
                        ff_video_tensor = ff_video.view(-1,c,h,w )
                        ff_image_embedding = ff_image_clip(ff_video_tensor)
                        ff_image_embedding = ff_image_embedding.view(b,t,-1)
                        ff_image_embedding = ff_temp_trans(ff_image_embedding)
                        ff_logits_per_image, ff_logits_per_text = create_logits(ff_image_embedding,ff_gen_text_embedding,logit_scale)
                        ff_loss_imgs = loss_img(ff_logits_per_image,ground_truth.to(dtype=ff_logits_per_image.dtype))
                        ff_loss_texts = loss_txt(ff_logits_per_text,ground_truth.to(dtype=ff_logits_per_text.dtype))

                        kl_loss = config.loss.image*((ff_loss_imgs + ff_loss_texts)/2) 
                    # bbimg fftxt generic
                    elif lambda_bb > 0:
                        
                        if use_generic:
                            bb_gen_text_embedding = ff_text_clip(bounded_texts)
                            bb_video_tensor = bb_video.view(-1,c,h,w )
                            bb_image_embedding = ff_image_clip(bb_video_tensor)
                            bb_image_embedding = bb_image_embedding.view(b,t,-1)
                            bb_image_embedding = ff_temp_trans(bb_image_embedding)
                            bb_logits_per_image, bb_logits_per_text = create_logits(bb_image_embedding,generic_embedding,logit_scale)
                            bb_loss_imgs = loss_img(bb_logits_per_image, ground_truth.to(dtype=bb_logits_per_image.dtype))
                            bb_loss_texts = loss_txt(bb_logits_per_text,ground_truth.to(dtype=bb_logits_per_text.dtype))

                            image_embedding = bb_image_embedding

                            ff_gen_text_embedding = ff_text_clip(texts)
                            ff_video_tensor = ff_video.view(-1,c,h,w )
                            ff_image_embedding = ff_image_clip(ff_video_tensor)
                            ff_image_embedding = ff_image_embedding.view(b,t,-1)
                            ff_image_embedding = ff_temp_trans(ff_image_embedding)
                            ff_logits_per_image, ff_logits_per_text = create_logits(image_embedding,ff_gen_text_embedding,logit_scale)
                            ff_loss_imgs = loss_img(ff_logits_per_image,ground_truth.to(dtype=ff_logits_per_imag.dtype))
                            ff_loss_texts = loss_txt(ff_logits_per_text,ground_truth.to(dtype=ff_logits_per_tex.dtype))

                            kl_loss = config.loss.image*((ff_loss_imgs + ff_loss_texts)/2) + config.loss.text*((bb_loss_imgs + bb_loss_texts)/2)
                        else:
                            ff_gen_text_embedding = ff_text_clip(texts)
                            text_embedding = ff_gen_text_embedding

                            bb_video_tensor = bb_video.view(-1,c,h,w )
                            bb_image_embedding = ff_image_clip(bb_video_tensor)
                            bb_image_embedding = bb_image_embedding.view(b,t,-1)
                            bb_image_embedding = ff_temp_trans(bb_image_embedding)
                            bb_logits_per_image, bb_logits_per_text = create_logits(bb_image_embedding,text_embedding,logit_scale)
                            bb_loss_imgs = loss_img(bb_logits_per_image, ground_truth.to(dtype=bb_logits_per_image.dtype))
                            bb_loss_texts = loss_txt(bb_logits_per_text,ground_truth.to(dtype=bb_logits_per_text.dtype))
                            kl_loss = config.loss.text*((bb_loss_imgs+bb_loss_texts)/2) 

                    else:
                        raise ValueError("Error!")
                elif lambda_enbb > 0:
                    # ffimg bbimg bbtxt
                    if lambda_ff > 0 and lambda_bb > 0:
                        bb_gen_text_embedding = ff_text_clip(bounded_texts)
                        bb_video_tensor = bb_video.view(-1,c,h,w )
                        bb_image_embedding = ff_image_clip(bb_video_tensor)
                        bb_image_embedding = bb_image_embedding.view(b,t,-1)
                        bb_image_embedding = ff_temp_trans(bb_image_embedding)
                        bb_logits_per_image, bb_logits_per_text = create_logits(bb_image_embedding,bb_gen_text_embedding,logit_scale)
                        bb_loss_imgs = loss_img(bb_logits_per_image, ground_truth.to(dtype=bb_logits_per_image.dtype))
                        bb_loss_texts = loss_txt(bb_logits_per_text,ground_truth.to(dtype=bb_logits_per_text.dtype))

                        if use_generic:
                            text_embedding = generic
                        else:
                            text_embedding = bb_gen_text_embedding

                        ff_video_tensor = ff_video.view(-1,c,h,w )
                        ff_image_embedding = ff_image_clip(ff_video_tensor)
                        ff_image_embedding = ff_image_embedding.view(b,t,-1)
                        ff_image_embedding = ff_temp_trans(ff_image_embedding)
                        ff_logits_per_image, ff_logits_per_text = create_logits(ff_image_embedding,text_embedding,logit_scale)
                        ff_loss_imgs = loss_img(ff_logits_per_image,ground_truth.to(dtype=ff_logits_per_image.dtype))
                        ff_loss_texts = loss_txt(ff_logits_per_text,ground_truth.to(dtype=ff_logits_per_text.dtype))

                        kl_loss = config.loss.image*((ff_loss_imgs + ff_loss_texts)/2) + config.loss.text*((bb_loss_imgs + bb_loss_texts)/2)
                    # ffimg bbtxt
                    elif lambda_ff > 0:
                        if use_generic:
                            ff_video_tensor = ff_video.view(-1,c,h,w )
                            ff_image_embedding = ff_image_clip(ff_video_tensor)
                            ff_image_embedding = ff_image_embedding.view(b,t,-1)
                            ff_image_embedding = ff_temp_trans(ff_image_embedding)
                            ff_logits_per_image, ff_logits_per_text = create_logits(ff_image_embedding,generic_embedding,logit_scale)
                            ff_loss_imgs = loss_img(ff_logits_per_image,ground_truth.to(dtype=ff_logits_per_image.dtype))
                            ff_loss_texts = loss_txt(ff_logits_per_text,ground_truth.to(dtype=ff_logits_per_text.dtype))

                            image_embedding = ff_image_embedding

                            bb_gen_text_embedding = ff_text_clip(bounded_texts)
                            bb_logits_per_image, bb_logits_per_text = create_logits(image_embedding,bb_gen_text_embedding,logit_scale)
                            bb_loss_imgs = loss_img(bb_logits_per_image, ground_truth.to(dtype=bb_logits_per_image.dtype))
                            bb_loss_texts = loss_txt(bb_logits_per_text,ground_truth.to(dtype=bb_logits_per_text.dtype))

                            kl_loss = config.loss.image*((ff_loss_imgs + ff_loss_texts)/2) + config.loss.text*((bb_loss_imgs + bb_loss_texts)/2)
                        else:
                            bb_gen_text_embedding = ff_text_clip(bounded_texts)
                            ff_video_tensor = ff_video.view(-1,c,h,w )
                            ff_image_embedding = ff_image_clip(ff_video_tensor)
                            ff_image_embedding = ff_image_embedding.view(b,t,-1)
                            ff_image_embedding = ff_temp_trans(ff_image_embedding)
                            ff_logits_per_image, ff_logits_per_text = create_logits(ff_image_embedding,bb_gen_text_embedding,logit_scale)
                            ff_loss_imgs = loss_img(ff_logits_per_image,ground_truth.to(dtype=ff_logits_per_image.dtype))
                            ff_loss_texts = loss_txt(ff_logits_per_text,ground_truth.to(dtype=ff_logits_per_text.dtype))
                            kl_loss = config.loss.image*((ff_loss_imgs + ff_loss_texts)/2)

                    # bbimg bbtxt
                    elif lambda_bb > 0:
                        bb_gen_text_embedding = ff_text_clip(bounded_texts)
                        bb_video_tensor = bb_video.view(-1,c,h,w )
                        bb_image_embedding = ff_image_clip(bb_video_tensor)
                        bb_image_embedding = bb_image_embedding.view(b,t,-1)
                        bb_image_embedding = ff_temp_trans(bb_image_embedding)
                        bb_logits_per_image, bb_logits_per_text = create_logits(bb_image_embedding,bb_gen_text_embedding,logit_scale)
                        bb_loss_imgs = loss_img(bb_logits_per_image, ground_truth.to(dtype=bb_logits_per_image.dtype))
                        bb_loss_texts = loss_txt(bb_logits_per_text,ground_truth.to(dtype=bb_logits_per_text.dtype))
                        kl_loss = config.loss.image*((bb_loss_imgs + bb_loss_texts)/2)
                    else:
                        raise ValueError("Error!")
                else:
                    raise ValueError("Error!")
            else:
                # Process Video Embeddings for BB and Original (full frame)
                if lambda_bb > 0:
                    bb_video_tensor = bb_video.view(-1,c,h,w )
                    bb_image_embedding = ff_image_clip(bb_video_tensor)
                    bb_image_embedding = bb_image_embedding.view(b,t,-1)
                    bb_image_embedding = ff_temp_trans(bb_image_embedding)
                else:
                    bb_image_embedding = torch.zeros(b,512)
                    bb_image_embedding = bb_image_embedding.to(device=bb_video.device, dtype=bb_video.dtype)


                if lambda_ff > 0:
                    ff_video_tensor = ff_video.view(-1,c,h,w )
                    ff_image_embedding = ff_image_clip(ff_video_tensor)
                    ff_image_embedding = ff_image_embedding.view(b,t,-1)
                    ff_image_embedding = ff_temp_trans(ff_image_embedding)
                else:
                    ff_image_embedding = torch.zeros(b,512)
                    ff_image_embedding = ff_image_embedding.to(device=bb_video.device, dtype=bb_video.dtype)
                
                # Process Text Embeddings
                if lambda_enff > 0:
                    ff_gen_text_embedding = ff_text_clip(texts)
                else:
                    ff_gen_text_embedding = torch.zeros(b, 512)
                    ff_gen_text_embedding = ff_gen_text_embedding.to(device=texts.device, dtype=bb_video.dtype)

                if config.data.florence.use_bounded_text and lambda_enbb > 0:
                    bb_gen_text_embedding = ff_text_clip(bounded_texts)
                else:
                    bb_gen_text_embedding = torch.zeros(b, 512)
                    bb_gen_text_embedding = bb_gen_text_embedding.to(device=texts.device, dtype=bb_video.dtype)


                if config.network.fix_text:
                    ff_gen_text_embedding.detach_()
                    # bb_text_embedding.detach_()
                if lambda_ff > 0 and lambda_bb > 0:
                    image_embedding = lambda_bb*bb_image_embedding    + lambda_ff*ff_image_embedding
                elif lambda_bb > 0:
                    image_embedding = lambda_bb*bb_image_embedding
                elif lambda_ff > 0:
                    image_embedding = lambda_ff*ff_image_embedding
                
                # Add here the generic and captioning
                if lambda_enbb > 0 and lambda_enff > 0:
                    text_embedding  = lambda_enff*ff_gen_text_embedding + lambda_enbb*bb_gen_text_embedding
                elif lambda_enbb > 0:
                    text_embedding  = lambda_enbb*bb_gen_text_embedding
                elif lambda_enff > 0:
                    text_embedding  = lambda_enff*ff_gen_text_embedding
                

                logits_per_image, logits_per_text = create_logits(image_embedding,text_embedding,logit_scale)
                
                loss_imgs = loss_img(logits_per_image,ground_truth.to(dtype=logits_per_image.dtype))
                loss_texts = loss_txt(logits_per_text,ground_truth.to(dtype=logits_per_text.dtype))
                # loss_flo = outputs.loss
                kl_loss = config.loss.image*loss_imgs + config.loss.text*loss_texts #+ config.data.mu*loss_flo
                

                if config.data.weights.gamma > 0:
                    text_inputs = classes.to(device)
                    generic_features = ff_text_clip(text_inputs)
                    logits_per_image, logits_per_text = create_logits(image_embedding,generic_features,100)
                    similarity = calculate_similarity(logits_per_image, b, num_text_aug)
                    loss_ce = cross_entropy(similarity, list_id)

                wandb.log({"train_loss_imgs":  loss_imgs})
                wandb.log({"train_loss_texts": loss_texts})
                

            total_loss = kl_loss 
            running_kl += kl_loss.item()
            if config.data.weights.gamma > 0:
                total_loss += config.data.weights.gamma*loss_ce.item()
                running_ce += ce_loss.item()
            
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(clip_perceptor)
                optimizer.step()
                clip.model.convert_weights(clip_perceptor)

            # scaler.update()

        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            prec1 = validate(
                             epoch,
                             val_loader, 
                             classes, 
                             class_text, 
                             device, 
                             clip_perceptor,
                             ff_temp_trans, 
                             config,
                             num_text_aug, 
                             flo_processor, 
                             flo_model, 
                             text_aug_dict, 
                             lambda_bb, 
                             lambda_ff, 
                             lambda_enff, 
                             lambda_enbb
                        )

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        
        log_str = f'Testing: {prec1}/{best_prec1}, Avg Loss KL: {running_kl/len(train_loader)}'
        if config.data.weights.gamma > 0:
            log_str += f', Avg Loss CE: {running_ce/len(train_loader)}'
        # if config.data.weights.mu > 0:
        #     log_str += f', Avg Flo Loss: {running_flo/len(train_loader)}'
        
        print(log_str)
        print('Saving:')
        filename = "{}/last_model.pt".format(working_dir)

        epoch_saving(epoch, clip_perceptor, ff_temp_trans, optimizer, filename)
        if is_best:
            best_saving(working_dir, epoch, clip_perceptor, ff_temp_trans, optimizer)

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'], args.log_time)
    
    seed = int(config['seed'])
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  # For GPU operations
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    random.seed(seed)
    numpy.random.seed(seed) 

    wandb.init(project=config['network']['type'],name='{}_{}_{}_{}'.format(args.log_time,config['network']['type'], config['network']['arch'], config['data']['dataset']))
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
    shutil.copy('train.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

    perceptor, clip_state_dict = clip.load(config.network.arch,
                                           device=device,
                                           jit=False,
                                           tsm=config.network.tsm, 
                                           T=config.data.num_segments,
                                           dropout=config.network.drop_out, 
                                           emb_dropout=config.network.emb_dropout,
                                           pretrain=config.network.init, 
                                           joint = config.network.joint) #Must set jit=False for training  ViT-B/32
    flo_model, processor = flor2.load("BASE_FT", device)
    # flo_model.image_processor.crop_size['height'] = config.data.input_size
    # flo_model.image_processor.crop_size['width'] = config.data.input_size
    # flo_model.image_processor.size['height'] = config.data.input_size
    # flo_model.image_processor.size['width'] = config.data.input_size

    # vlm_state_dict["text_projection"] = torch.empty((1, flo_model.config.text_config.d_model))
    # vlm_state_dict["positional_embedding"] = torch.empty((flo_model.config.text_config.vocab_size,))
    # vlm_state_dict["ln_final.weight"] = torch.empty((flo_model.config.text_config.encoder_attention_heads*64,))
    
    transform_train = get_augmentation(True,config)
    transform_val = get_augmentation(False,config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)

    fusion_model = visual_prompt(config.network.sim_header,clip_state_dict,config.data.num_segments)
    model_text = TextCLIP(perceptor)
    model_image = ImageCLIP(perceptor)
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    flo_model = torch.nn.DataParallel(flo_model).cuda()
    wandb.watch(perceptor)
    wandb.watch(fusion_model)

    def collate_fn(batch):
        cropped_videos, images, bbs, nonaug_images, _, labels = zip(*batch)
        # Check the labels for bb
        cropped_videos = torch.stack(cropped_videos) 
        images = torch.stack(images) 
        labels = torch.tensor(labels)
        nonaug_images = torch.stack(nonaug_images)
        bbs = torch.stack(bbs)
        return cropped_videos, images, nonaug_images, bbs, labels

    train_data = EducationBBDataset(
                    config.data.train_list,
                    config.data.label_list,
                    num_segments=config.data.num_segments,
                    image_tmpl=config.data.image_tmpl,
                    random_shift=config.data.random_shift,
                    transform=transform_train,
                    label_box=config.data.label_box,
                    caption_level=config.data.florence.caption_level)
    train_loader = DataLoader(
                    train_data,
                    batch_size=config.data.batch_size,
                    num_workers=config.data.workers,
                    shuffle=True,
                    pin_memory=False,
                    drop_last=True, 
                    collate_fn=collate_fn)
    val_data = EducationBBDataset(
                    config.data.val_list,
                    config.data.label_list, 
                    random_shift=False,
                    num_segments=config.data.num_segments,
                    image_tmpl=config.data.image_tmpl,
                    transform=transform_val,
                    label_box=config.data.label_box)
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
    else :
        clip.model.convert_weights(model_text) # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    loss_img = KLLoss()
    loss_txt = KLLoss()

    start_epoch = config.solver.start_epoch
    
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain, map_location='cpu')
            perceptor.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))
    
    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            perceptor.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict, text_aug_dict, class_text = text_prompt(train_data, file_name=config.prompt)

    best_prec1 = 0.0
    if config.solver.evaluate:
        prec1 = validate(start_epoch,val_loader, classes, class_text, device, perceptor,fusion_model, config,num_text_aug, processor, flo_model, 1, 1, 1, 1)
        return

    train_classifier(
            # Models
            ff_image_clip = model_image, 
            ff_text_clip = model_text, 
            ff_temp_trans = fusion_model,
            flo_processor= processor, 
            flo_model = flo_model, 
            clip_perceptor = perceptor,
            
            # loss functions
            loss_img = loss_img,
            loss_txt = loss_txt,
            
            # Data loaders
            train_loader = train_loader, 
            val_loader = val_loader, 
            
            # Miscelaneous
            start_epoch = start_epoch,
            device = device, 
            config = config,
            working_dir = working_dir,
            
            # Tokenized Text
            text_dict = text_dict,
            text_aug_dict = text_aug_dict,
            num_text_aug = num_text_aug,
            classes = classes,
            class_text = class_text
    )


if __name__ == '__main__':
    main()
