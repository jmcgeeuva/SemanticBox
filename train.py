# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

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
from prompt import PromptLoss
from helpers import ReplaceGrad
from augmentation import get_mask_augmentation

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

def create_prompt_loss_dict(label_names, perceptor, replace_grad, device):
    # keys = sorted(list(label_names.keys()))
    res = {}
    for label_num, label_name in label_names:
        prompt = f'Video of a teacher {" ".join(label_name.split("_"))}'
        res[label_num] = PromptLoss(prompt, perceptor, replace_grad).to(device)
    return res
    
def train_classifier(start_epoch, 
                     loss_img,
                     loss_txt,
                     criterion_list,
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
    ################### Train Classifier ####################################
    for epoch in range(start_epoch, config.solver.epochs):
        model_image.train()
        model_text.train()
        fusion_model.train()

        correct, total = 0, 0
        running_loss = 0
        for kkk,(videos, masks, list_id) in enumerate(tqdm(train_loader)):
            if config.solver.type != 'monitor':
                if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()


            
            b,t,c,h,w = videos.size()
            
            aug_masks, lambdas = masks
            aug_masks = aug_masks.to(device)
            lambdas = lambdas.to(device)
            text_id = numpy.random.randint(num_text_aug,size=len(list_id))
            texts = torch.stack([text_dict[j][i,:] for i,j in zip(list_id,text_id)])

            videos= videos.to(device).view(-1,c,h,w ) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
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

            ############## Calculate explainability loss between attn and mask ###########
            # run through the prompt model and get the new results from the prompts
            lossAll = []
            per_frame = True
            if len(criterion_list) > 0 and False:
                # for each label in the list and for each invidividual video (there are 16 of them)
                for idx, label in enumerate(list_id.detach().cpu()):
                    # find the loss for this specific bounding box
                    prompt_idx = int(label)
                    crit = criterion_list[prompt_idx]
                    if per_frame:
                        subloss = []
                        # for each frame (in a group of 8 frames), the given mask, and its lambda
                        for curr_iter, (curr_image, curr_mask, curr_lambda) in enumerate(zip(videos[8*idx:8*(idx+1)], aug_masks[idx], lambdas[idx])):
                            # Needs to be 1x3x224x224 for curr_image
                            curr_image = curr_image.unsqueeze(0)
                            res = crit(curr_image, curr_mask, curr_lambda)
                            subloss.append(res)
                        lossAll.append(sum(subloss)/len(subloss))
                    else:
                        res = crit(videos[8*idx:8*(idx+1)].unsqueeze(0), aug_masks[idx], lambdas[idx])
                        lossAll.append(res)

            
            total_loss = ((loss_imgs + loss_texts)/2) #+ .3*(-1*(sum(lossAll)/len(lossAll)))
            running_loss += total_loss.item()

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
        print(f'Testing: {prec1}/{best_prec1}, Loss CE: {(running_loss/len(train_loader)):.4f}')
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
    
    if 'seed' in config:
        torch.manual_seed(int(config['seed'])) #1737328734)
        numpy.random.seed(int(config['seed'])) #1737328734)

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

    # Modify the attention mechanism in the visual transformer
    class ModifiedAttention(torch.nn.Module):
        def __init__(self, original_attention):
            super().__init__()
            self.original_attention = original_attention
            self.attn_output_weights = torch.zeros(1, 50, 50, requires_grad=True)
            self.attn_output_weights.retain_grad()

        def forward(self, query, key, value, need_weights, attn_mask):
            query = query / (query.size(-1) ** 0.5)
            self.attn_output_weights = torch.matmul(query, key.transpose(-2, -1))  # QK^T
            self.attn_output_weights = torch.nn.functional.softmax(self.attn_output_weights, dim=-1)

            # self.attn_output_weights.requires_grad_()
            # Retain gradient on attention weights
            # self.attn_output_weights.retain_grad()

            attn_output = torch.matmul(self.attn_output_weights, value)  # Weighted sum
            return attn_output, self.attn_output_weights
            # self.attn_output, self.attn_output_weights = self.original_attention(
            #     query, key, value, need_weights=True, attn_mask=attn_mask
            # )
            # self.attn_output_weights.retain_grad()
            # self.attn_output = torch.matmul(self.attn_output_weights, value)
            # # print(self.attn_output.shape, self.attn_output_weights.shape)
            # return self.attn_output, self.attn_output_weights

    # Apply the modified attention layer to all layers in the visual transformer
    for layer in perceptor.visual.transformer.resblocks:
        layer.attn = ModifiedAttention(layer.attn)
    # import pdb; pdb.set_trace()

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
                                           noise_fac = 0.1, 
                                           is_classifier=True)

    def collate_fn(batch):
        videos, masks, lambda_val, labels = zip(*batch)
        # Check the labels for bb
        videos, labels = torch.stack(videos), torch.tensor(labels)
        lambda_val = torch.tensor(lambda_val)
        masks = torch.stack(masks, dim=0)
        videos = videos.view((-1,config.data.num_segments,3)+videos.size()[-2:])
        # masks = masks.squeeze(dim=2)
        iii, aug_masks = mask_transform((videos, masks))
        # iii = iii.squeeze()
        return videos, (aug_masks, lambda_val), labels

    train_data = Action_DATASETS(
                    config.data.train_list,
                    config.data.label_list,
                    num_segments=config.data.num_segments,
                    image_tmpl=config.data.image_tmpl,
                    random_shift=config.data.random_shift,
                    image_transform=transform_train,
                    bounding_boxes=config.bounding_boxes)
    train_loader = DataLoader(
                    train_data,
                    batch_size=config.data.batch_size,
                    num_workers=config.data.workers,
                    shuffle=True,
                    pin_memory=False,
                    drop_last=True, 
                    collate_fn=collate_fn)
    val_data = Action_DATASETS(
                    config.data.val_list,
                    config.data.label_list, 
                    random_shift=False,
                    num_segments=config.data.num_segments,
                    image_tmpl=config.data.image_tmpl,
                    image_transform=transform_val,
                    bounding_boxes=config.bounding_boxes)
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
    # 1. Compile a proper list of the classes with label numbers
    # 3. Create a replace_grad
    
    # 2. Find what the perceptor is for me
    # label_dict = {idx:class_name for idx, class_name in enumerate(train_data.classes)}
    if config.bounding_boxes:
        criterion_list = create_prompt_loss_dict(train_data.classes, perceptor, replace_grad, device)
    else:
        criterion_list = []

    best_prec1 = 0.0
    if config.solver.evaluate:
        prec1 = validate(start_epoch,val_loader, classes, device, perceptor,fusion_model, config,num_text_aug)
        return

    # for k,v in model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))

    optimizer = _optimizer(config, perceptor, fusion_model)
    lr_scheduler = _lr_scheduler(config, optimizer)


    train_classifier(start_epoch = start_epoch, 
                     loss_img = loss_img,
                     loss_txt = loss_txt,
                     criterion_list = criterion_list,
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
