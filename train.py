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
from test import validate, run_example
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
        if config.data.weighted_features.learned:
            lambda_bb = nn.Parameter(torch.ones(config.data.batch_size, 512, requires_grad=True, device=device))
            lambda_ff = nn.Parameter(torch.ones(config.data.batch_size, 512, requires_grad=True, device=device))
            lambda_en = nn.Parameter(torch.ones(config.data.batch_size, 512, requires_grad=True, device=device))
            lambda_ge = nn.Parameter(torch.ones(config.data.batch_size, 512, requires_grad=True, device=device))
            lambda_bb = lambda_bb.to(device=device, dtype=torch.float32)
            lambda_ff = lambda_ff.to(device=device, dtype=torch.float32)
            lambda_en = lambda_en.to(device=device, dtype=torch.float32)
            lambda_ge = lambda_ge.to(device=device, dtype=torch.float32)
            optimizer = _optimizer(config, clip_perceptor, ff_temp_trans, [lambda_bb, lambda_ff, lambda_en, lambda_ge])
            lr_scheduler = _lr_scheduler(config, optimizer)
        else:
            lambda_bb = config.data.weights.lambda_bb
            lambda_ff = config.data.weights.lambda_ff
            lambda_en = config.data.weights.lambda_en
            lambda_ge = config.data.weights.lambda_ge
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

        running_kl = 0

        for kkk,data in enumerate(tqdm(train_loader)):
            if config.solver.type != 'monitor':
                if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            bb_video, ff_video, generated_images, bbs, list_id = data
            ff_video = ff_video.to(device)
            list_id = list_id.to(device)
            # generated_images = generated_images.to(device)
            bb_video = bb_video.view((-1,config.data.num_segments,3)+bb_video.size()[-2:])
            
            b,t,c,h,w = bb_video.size()
            
            text_id = numpy.random.randint(num_text_aug,size=len(list_id))

            token_texts = torch.stack([text_dict[j][i,:] for i,j in zip(list_id,text_id)])
            real_texts = [text_aug_dict[j][i] for i,j in zip(list_id,text_id)]
            bb_video= bb_video.to(device).view(-1,c,h,w ) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class

            if not config.data.florence.activate:
                texts = token_texts.to(device)
            elif config.data.florence.use_bounded_text:
                with torch.no_grad():
                    final_text, bounded_text = run_example(generated_images, flo_processor, flo_model, real_texts, config, bbs=bbs)
                    texts  = final_text.to(device)
                    bounded_texts = bounded_text.to(device)
            else:
                with torch.no_grad():
                    final_text = run_example(generated_images, flo_processor, flo_model, real_texts, config)
                    texts  = final_text.to(device)

            # Process Video Embeddings for BB and Original (full frame)
            bb_video_tensor = bb_video.view(-1,c,h,w )
            bb_image_embedding = ff_image_clip(bb_video_tensor)
            bb_image_embedding = bb_image_embedding.view(b,t,-1)
            bb_image_embedding = ff_temp_trans(bb_image_embedding)

            ff_video_tensor = ff_video.view(-1,c,h,w )
            ff_image_embedding = ff_image_clip(ff_video_tensor)
            ff_image_embedding = ff_image_embedding.view(b,t,-1)
            ff_image_embedding = ff_temp_trans(ff_image_embedding)
            
            # Process Text Embeddings
            ff_gen_text_embedding = ff_text_clip(texts)
            # if config.data.florence.use_bounded_text:
            #     bb_text_embedding = ff_text_clip(bounded_text)
            # else:
            #     bb_text_embedding = ff_gen_text_embedding
            generic_texts = token_texts.to(device)
            set_text_embedding = ff_text_clip(generic_texts)

            if config.network.fix_text:
                ff_gen_text_embedding.detach_()
                # bb_text_embedding.detach_()

            ground_truth = torch.tensor(gen_label(list_id),dtype=ff_image_embedding.dtype,device=device)
            logit_scale = clip_perceptor.logit_scale.exp()
            if config.data.weighted_features.use:
                if config.data.weighted_features.learned:
                    image_embedding = lambda_bb.to(dtype=bb_image_embedding.dtype)*bb_image_embedding       + lambda_ff.to(dtype=ff_image_embedding.dtype)*ff_image_embedding
                    text_embedding  = lambda_en.to(dtype=ff_gen_text_embedding.dtype)*ff_gen_text_embedding + lambda_ge.to(dtype=set_text_embedding.dtype)*set_text_embedding
                else:
                    image_embedding = lambda_bb*bb_image_embedding    + lambda_ff*ff_image_embedding
                    # Add here the generic and captioning
                    text_embedding  = lambda_en*ff_gen_text_embedding + lambda_ge*set_text_embedding
                

                logits_per_image, logits_per_text = create_logits(image_embedding,text_embedding,logit_scale)
                
                loss_imgs = loss_img(logits_per_image,ground_truth)
                loss_texts = loss_txt(logits_per_text,ground_truth)
                # loss_flo = outputs.loss
                kl_loss = config.loss.image*loss_imgs + (1 - config.loss.image)*loss_texts #+ config.data.mu*loss_flo
                wandb.log({"train_loss_imgs":  loss_imgs})
                wandb.log({"train_loss_texts": loss_texts})
            else:
                bb_logits_per_image, bb_logits_per_text = create_logits(bb_image_embedding,bb_text_embedding,logit_scale)
                ff_logits_per_image, ff_logits_per_text = create_logits(ff_image_embedding,ff_text_embedding,logit_scale)
            
                ff_loss_imgs = loss_img(ff_logits_per_image,ground_truth)
                ff_loss_texts = loss_txt(ff_logits_per_text,ground_truth)
                ff_kl_loss = (ff_loss_imgs + ff_loss_texts)/2

                bb_loss_imgs = loss_img(bb_logits_per_image, ground_truth)
                bb_loss_texts = loss_txt(bb_logits_per_text,ground_truth)
                bb_kl_loss = (bb_loss_imgs + bb_loss_texts)/2

                kl_loss = config.data.weights.lambda_bb*bb_kl_loss + config.data.weights.lambda_ff*ff_kl_loss
                wandb.log({"train_ff_loss_imgs":  ff_loss_imgs})
                wandb.log({"train_ff_loss_texts": ff_loss_texts})
                wandb.log({"train_bb_loss_imgs":  bb_loss_imgs})
                wandb.log({"train_bb_loss_texts": bb_loss_texts})

            running_kl += kl_loss.item()

            wandb.log({"lr": optimizer.param_groups[0]['lr']})
            kl_loss.backward()

            if device == "cpu":
                optimizer.step()
                # optimizer.step()
            else:
                convert_models_to_fp32(clip_perceptor)
                optimizer.step()
                # optimizer.step()
                clip.model.convert_weights(clip_perceptor)

            # scaler.update()

        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            prec1 = validate(epoch,val_loader, classes, class_text, device, clip_perceptor,ff_temp_trans, config,num_text_aug, flo_processor, flo_model, text_aug_dict, lambda_bb, lambda_ff)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print(f'Testing: {prec1}/{best_prec1}, Avg Loss KL: {running_kl/len(train_loader)}')
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
        cropped_videos, images, bbs, generated_images, labels = zip(*batch)
        # Check the labels for bb
        cropped_videos = torch.stack(cropped_videos) 
        images = torch.stack(images) 
        labels = torch.tensor(labels)
        generated_images = torch.stack(generated_images)
        bbs = torch.stack(bbs)
        return cropped_videos, images, generated_images, bbs, labels

    train_data = EducationBBDataset(
                    config.data.train_list,
                    config.data.label_list,
                    num_segments=config.data.num_segments,
                    image_tmpl=config.data.image_tmpl,
                    random_shift=config.data.random_shift,
                    transform=transform_train,
                    label_box=config.data.label_box)
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

    replace_grad = ReplaceGrad.apply

    best_prec1 = 0.0
    if config.solver.evaluate:
        prec1 = validate(start_epoch,val_loader, classes, class_text, device, perceptor,fusion_model, config,num_text_aug, processor, flo_model, 1, 1)
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
