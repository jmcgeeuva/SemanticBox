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

    labeled_ids = []
    correct_ids = []
    classes_aug = [v for k, v in text_aug_dict.items()]
    similarities_image = []
    similarities_text  = []
    with torch.no_grad():
        for iii, data in enumerate(tqdm(val_loader)):
            
            bb_video, ff_video, nonaug_images, bbs, class_id = data
            ff_video = ff_video.to(device)
            class_id = class_id.to(device)
            bbs = bbs.to(device)
            nonaug_images = nonaug_images.to(device)
            bb_video = bb_video.view((-1,config.data.num_segments,3)+bb_video.size()[-2:])
            
            text_inputs = classes.to(device)
            generic_text_features = clip_model.encode_text(text_inputs)

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
            
            if lambda_ff > 0 and lambda_bb > 0:
                image_features = lambda_bb*bb_image_features + lambda_ff*ff_image_features
            elif lambda_ff > 0:
                image_features = lambda_ff*ff_image_features
            elif lambda_bb > 0:
                image_features = lambda_bb*bb_image_features

            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            testing_features = generic_text_features 
            testing_features /= testing_features.norm(dim=-1, keepdim=True)
            # Create Logits
            logits_per_image = (100.0 * image_features @ testing_features.T)
            similarity_image = calculate_similarity(logits_per_image, b, num_text_aug) # B x L

            values_1, indices_1 = similarity_image.topk(1, dim=-1)
            values_k, indices_k = similarity_image.topk(config.data.k, dim=-1)
            num += b
            for i in range(b):
                if indices_1[i] == class_id[i]:
                    corr_1 += 1
                if class_id[i] in indices_k[i]:
                    corr_k += 1
                
            labeled_ids.append(indices_k)
            correct_ids.extend(class_id.tolist())

    labeled_ids = torch.cat(labeled_ids).tolist()
    if print_metrics:
        plot_confusion_matrix(correct_ids, labeled_ids, np.array(["Using a Book", "Teacher Sitting", "Teacher Standing", "Teacher Writing", "Using Technology", "Using a Worksheet"]), config.test_name)

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
