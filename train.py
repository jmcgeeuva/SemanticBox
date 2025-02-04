# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
import torch.nn as nn
from datasets import Action_DATASETS, Action_DATASETS_orig
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
from test import validate, calculate_similarity
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import  *
import sys
sys.path.insert(0, "../explain/ml-no-token-left-behind/external/tamingtransformers/")
sys.path.append("./../explain/ml-no-token-left-behind/external/TransformerMMExplainability/")
from prompt import PromptLoss, TextCLIP, ImageCLIP
from prompt import PromptLoss2 as pl2
from helpers import *
import random

import torch

# def create_prompt_loss_dict(label_names, perceptor, replace_grad, device):
#     # keys = sorted(list(label_names.keys()))
#     res = {}
#     for label_num, label_name in label_names:
#         prompt = f'Video of a teacher {" ".join(label_name.split("_"))}'
#         res[label_num] = PromptLoss(prompt, perceptor, replace_grad).to(device)
#     return res
    
# def bounding_box_loss(criterion_list, b, t, c, h, w, list_id, aug_masks, lambdas, texts, promptCrit, fusion_model):
#     lossAll = []
#     # per_frame = False
#     loop_list = True
#     text_list = []
#     image_list = []
#     if len(criterion_list) > 0:
#         # for each label in the list and for each invidividual video (there are 16 of them)
#         if loop_list:
#             videos = videos.reshape(b,t,c,h,w)
#             for idx, label in enumerate(list_id.detach().cpu()):
#                 # find the loss for this specific bounding box
#                 prompt_idx = int(label)
#                 crit = criterion_list[prompt_idx]
#                 curr_image = videos[idx]
#                 curr_mask = aug_masks[idx]
#                 curr_lambda = lambdas[idx]
#                 token = texts[idx]
    
#                 res, text_embedding, image_embedding = promptCrit(curr_image, curr_mask, curr_lambda, token)
#                 image_list.append(image_embedding)
    
#                 text_list.append(text_embedding[0])
#                 lossAll.append(res)
#             loss_all = (sum(lossAll)/len(lossAll))
#             text_embedding = torch.stack(text_list)
#             image_embedding = torch.stack(image_list)
#             image_embedding = image_embedding.view(b,t,-1)
#             image_embedding = fusion_model(image_embedding)
#             # print(sum(lossAll), len(lossAll))
#         # else:
#         #     # give the loss function the current text to unify the two
#         #     videos = videos.reshape(b,t,c,h,w)
#         #     print(videos.shape, aug_masks.shape, lambdas.shape, texts.shape)
#         #     loss_all = promptCrit(videos, aug_masks, lambdas, texts)
#         #     raise ValueError("test")
#     return loss_all, image_embedding, text_embedding


def calculate_ce(device, classes, perceptor, image_embedding, b, num_text_aug, cross_entropy):
    list_id = list_id.to(device)
    text_inputs = classes.to(device)
    text_features2 = perceptor.encode_text(text_inputs)
    logits_per_image2_orig = (100.0 * image_embedding @ text_features2.T)
    similarity = calculate_similarity(logits_per_image2, b, num_text_aug)
    similarity_orig = calculate_similarity(logits_per_image2_orig, b, num_text_aug)
    list_id = list_id.to(device)
    ce_loss = cross_entropy(similarity+similarity_orig, list_id)
    return ce_loss


def train_classifier(start_epoch, 
                     loss_img,
                     loss_txt,
                    #  criterion_list,
                    #  promptCrit,
                     lr_scheduler,
                     config, 
                     text_dict,
                     model_image, 
                     model_text, 
                     fusion_model, 
                     train_loader, 
                     val_loader, 
                     optimizer, 
                     num_text_aug,
                     classes,
                     perceptor,
                     working_dir,
                     device):
    best_prec1 = -1
    if config.data.ce.use:
        cross_entropy = nn.CrossEntropyLoss()
    # clamp_with_grad = ClampWithGrad.apply

    ################### Train Classifier ####################################
    # scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, config.solver.epochs):
        model_image.train()
        model_text.train()
        fusion_model.train()
        running_loss_all = 0
        running_kl = 0
        running_ce = 0
        running_total = 0
        for kkk,(cropped_videos, videos, list_id) in enumerate(tqdm(train_loader)):
            if config.solver.type != 'monitor':
                if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            cropped_videos = cropped_videos.view((-1,config.data.num_segments,3)+cropped_videos.size()[-2:])
            videos = videos.view((-1,config.data.num_segments,3)+videos.size()[-2:])
            b,t,c,h,w = cropped_videos.size()
            text_id = numpy.random.randint(num_text_aug,size=len(list_id))
            texts = torch.stack([text_dict[j][i,:] for i,j in zip(list_id,text_id)])
            
            cropped_videos= cropped_videos.to(device).view(-1,c,h,w ) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            videos = videos.to(device).view(-1,c,h,w)
            texts = texts.to(device)

            image_embedding = model_image(videos)
            image_embedding = image_embedding.view(b,t,-1)
            image_embedding = fusion_model(image_embedding)

            text_embedding = model_text(texts)

            if config.network.fix_text:
                text_embedding.detach_()

            logit_scale = perceptor.logit_scale.exp()
            logits_per_image, logits_per_text = create_logits(image_embedding,text_embedding,logit_scale)

            ground_truth = torch.tensor(gen_label(list_id),dtype=image_embedding.dtype,device=device)
            loss_imgs = loss_img(logits_per_image,ground_truth)
            loss_texts = loss_txt(logits_per_text,ground_truth)

            if config.data.cropped.use:
                image_emb_cropped = model_image(cropped_videos)
                image_emb_cropped = image_emb_cropped.view(b,t,-1)
                image_emb_cropped = fusion_model(image_emb_cropped)

                logits_per_image_cropped, logits_per_text_cropped = create_logits(image_emb_cropped,text_embedding,logit_scale)

                loss_imgs_cropped = loss_img(logits_per_image_cropped, ground_truth)

                loss_imgs = (config.data.orig.lambda_val*loss_imgs + config.data.cropped.lambda_val*loss_imgs_cropped)

            kl_loss = (loss_imgs + loss_texts)/2
            total_loss = kl_loss
            
            if config.data.ce.use:
                ce_loss = calculate_ce(device, classes, perceptor, image_embedding, b, num_text_aug, cross_entropy)
                total_loss += config.data.ce.lambda_val*ce_loss
                running_ce += ce_loss.item()
                wandb.log({"train_loss_ce": ce_loss})

            running_kl += kl_loss.item()
            running_total += total_loss.item()
            wandb.log({"train_total_loss": total_loss})
            wandb.log({"train_loss_imgs": loss_imgs})
            wandb.log({"train_loss_texts": loss_texts})
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(perceptor)
                optimizer.step()
                clip.model.convert_weights(perceptor)

        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            prec1 = validate(epoch,val_loader, classes, device, perceptor,fusion_model, config,num_text_aug)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print(f'Testing: {prec1}/{best_prec1}, Avg Loss KL: {running_kl/len(train_loader)}, Avg Loss All: {running_loss_all/len(train_loader)}, Avg CE Loss: {running_ce/len(train_loader)}, Avg TOTAL: {running_total/len(train_loader)}')
        print('Saving:')
        filename = "{}/last_model.pt".format(working_dir)

        epoch_saving(epoch, perceptor, fusion_model, optimizer, filename)
        if is_best:
            best_saving(working_dir, epoch, perceptor, fusion_model, optimizer)

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
    transform_train = get_augmentation(True,config)
    transform_val = get_augmentation(False,config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)


    # print('train transforms: {}'.format(transform_train.transforms))
    # print('val transforms: {}'.format(transform_val.transforms))

    fusion_model = visual_prompt(config.network.sim_header,clip_state_dict,config.data.num_segments)
    model_text = TextCLIP(perceptor)
    model_image = ImageCLIP(perceptor)
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    wandb.watch(perceptor)
    wandb.watch(fusion_model)

    mask_transform = get_mask_augmentation(cut_size=224, 
                                        cutn=1, 
                                        cut_pow=1., 
                                        noise_fac = 0.1)
    # if config.data.bb.use:
    #     def collate_fn(batch):
    #         videos, masks, lambda_val, labels = zip(*batch)
    #         # Check the labels for bb
    #         videos, labels = torch.stack(videos), torch.tensor(labels)
    #         lambda_val = torch.tensor(lambda_val)
    #         masks = torch.stack(masks, dim=0)
    #         # videos = videos.view((-1,config.data.num_segments,3)+videos.size()[-2:])
    #         # masks = masks.view((-1,config.data.num_segments,3)+masks.size()[-2:])
    #         masks = masks.squeeze(dim=1)
    #         videos = videos.squeeze(dim=1)
    #         data = {'videos': videos, 'masks': masks}
    #         iii, aug_masks = mask_transform(data)
    #         # iii = iii.squeeze()
    #         return videos, aug_masks, lambda_val, labels

    #     train_data = Action_DATASETS(
    #                     config.data.train_list,
    #                     config.data.label_list,
    #                     num_segments=config.data.num_segments,
    #                     image_tmpl=config.data.image_tmpl,
    #                     random_shift=config.data.random_shift,
    #                     image_transform=transform_train)
    #     train_loader = DataLoader(
    #                     train_data,
    #                     batch_size=config.data.batch_size,
    #                     num_workers=config.data.workers,
    #                     shuffle=True,
    #                     pin_memory=False,
    #                     drop_last=True, 
    #                     collate_fn=collate_fn)
    #     val_data = Action_DATASETS(
    #                     config.data.val_list,
    #                     config.data.label_list, 
    #                     random_shift=False,
    #                     num_segments=config.data.num_segments,
    #                     image_tmpl=config.data.image_tmpl,
    #                     image_transform=transform_val)
    #     val_loader = DataLoader(
    #                     val_data,
    #                     batch_size=config.data.batch_size,
    #                     num_workers=config.data.workers,
    #                     shuffle=False,
    #                     pin_memory=False,
    #                     drop_last=True,
    #                     collate_fn=collate_fn)
    def collate_fn(batch):
        cropped_videos, images, bbs, labels = zip(*batch)
        # Check the labels for bb
        cropped_videos = torch.stack(cropped_videos) 
        images = torch.stack(images) 
        labels = torch.tensor(labels)
        return cropped_videos, images, labels

    train_data = Action_DATASETS_orig(
                    config.data.train_list,
                    config.data.label_list,
                    num_segments=config.data.num_segments,
                    image_tmpl=config.data.image_tmpl,
                    random_shift=config.data.random_shift,
                    transform=transform_train)
    train_loader = DataLoader(
                    train_data,
                    batch_size=config.data.batch_size,
                    num_workers=config.data.workers,
                    shuffle=True,
                    pin_memory=False,
                    drop_last=True, 
                    collate_fn=collate_fn)
    val_data = Action_DATASETS_orig(
                    config.data.val_list,
                    config.data.label_list, 
                    random_shift=False,
                    num_segments=config.data.num_segments,
                    image_tmpl=config.data.image_tmpl,
                    transform=transform_val)
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
            checkpoint = torch.load(config.pretrain)
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

    classes, num_text_aug, text_dict = text_prompt(train_data, file_name=config.prompt)

    replace_grad = ReplaceGrad.apply
    
    # if config.data.bb.use:
    #     criterion_list = create_prompt_loss_dict(train_data.classes, perceptor, replace_grad, device)
    # else:
    #     criterion_list = []

    best_prec1 = 0.0
    if config.solver.evaluate:
        prec1 = validate(start_epoch,val_loader, classes, device, perceptor,fusion_model, config,num_text_aug)
        return

    # for k,v in model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))

    optimizer = _optimizer(config, perceptor, fusion_model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    # promptCrit = pl2(perceptor, replace_grad, im_emb_type=config.data.im_emb_type).to(device)

    train_classifier(start_epoch = start_epoch, 
                     loss_img = loss_img,
                     loss_txt = loss_txt,
                    #  criterion_list = criterion_list,
                    #  promptCrit = promptCrit,
                     lr_scheduler = lr_scheduler,
                     config = config, 
                     text_dict = text_dict,
                     model_image = model_image, 
                     model_text = model_text, 
                     fusion_model = fusion_model, 
                     train_loader = train_loader, 
                     val_loader = val_loader, 
                     optimizer = optimizer, 
                     num_text_aug = num_text_aug,
                     classes = classes,
                     perceptor = perceptor,
                     working_dir = working_dir,
                     device = device)


if __name__ == '__main__':
    main()
