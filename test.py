# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
import sys
sys.path.insert(0, "./../explainable_bounding_box/ml-no-token-left-behind/external/tamingtransformers/")
sys.path.append("./../explainable_bounding_box/ml-no-token-left-behind/external/TransformerMMExplainability/")
import CLIP.clip as clip
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

florence_activate = False
if not florence_activate:
    test1 = False
    test2 = False
    test3 = False
else:
    test1 = False
    test2 = False
    test3 = False

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
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

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

def run_example(generated_images, processor, model, real_texts, bbs=None, label=None, text_input=None, debug=False, test=False):

    REGION_TO_DESCRIPTION = 'REGION_TO_DESCRIPTION'
    CAPTION = 'CAPTION'

    prompt_type = f'<{CAPTION}>' #'<DETAILED_CAPTION>'
    prompt_cap = [prompt_type for _ in generated_images]

    # Just choose one frame to create a caption for
    if not test2:
        cropped_images = [transforms.ToPILImage()(images[0]) for images in generated_images]
    else:
        cropped_images = [transforms.ToPILImage()(images[random.randint(0, len(images)-1)]) for images in generated_images]
    
    bounded_text = []
    if bbs != None:
        # [16, 8, 4] = [B, T, C]
        prompts_r2d = []
        bbs = bbs[:, 0, :]
        width = cropped_images[0].size[0]
        height = cropped_images[0].size[1]
        for bb in bbs:
            norm_bb = normalize(bb.tolist(), width, height)
        
            prompt_r2d = f'<{REGION_TO_DESCRIPTION}>'
            for dim in norm_bb:
                prompt_r2d += f'<loc_{dim}>'
            prompts_r2d.append(prompt_r2d)
        
        generated_texts = []
        text_dict = {
            CAPTION: [],
            REGION_TO_DESCRIPTION: []
        }
        for task, prompt, padding in [(CAPTION, prompt_cap, False), (REGION_TO_DESCRIPTION, prompts_r2d, True)]:
            inputs = processor(text=prompt, images=cropped_images, return_tensors="pt", padding=padding)

            input_ids = inputs["input_ids"].to(model.module.device)
            pixel_values = inputs["pixel_values"].to(model.module.device)
            generated_ids = model.module.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=512,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            for this_text, this_prompt in zip(generated_text, prompt):
                parsed_answer = processor.post_process_generation(
                    this_text,
                    task=this_prompt,
                    image_size=(width, height)
                )
                text_dict[task].append(parsed_answer[this_prompt])

        
        generated_text = text_dict[CAPTION]
        bounded_text = text_dict[REGION_TO_DESCRIPTION]
    else:
        prompt = prompt_cap
        inputs = processor(text=prompt, images=cropped_images, return_tensors="pt", padding=False)

        input_ids = inputs["input_ids"].to(model.module.device)
        pixel_values = inputs["pixel_values"].to(model.module.device)
        generated_ids = model.module.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=512,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)


    if not test:
        generated_text = [prompt_text.capitalize() + ": " + text.replace('<s>', '').replace('</s>', '') for text, prompt_text in zip(generated_text, real_texts)]
        final_text = torch.stack([clip.tokenize(text) for text in generated_text])
        if bounded_text != []:
            bounded_test = [prompt_text.capitalize() + ": This is a " + text.replace('<s>', '').replace('</s>', '') for text, prompt_text in zip(bounded_text, real_texts)]
            final_bounded_text = torch.stack([clip.tokenize(text) for text in bounded_test])
            return final_text.squeeze(dim=1), final_bounded_text.squeeze(dim=1)
        return final_text.squeeze(dim=1)
    else:
        # There are 16 lists of phrases
        final_texts = []
        final_texts_dict = {
            CAPTION: [],
            REGION_TO_DESCRIPTION: []
        }
        for text_list in real_texts:
            # There are 6 classes for each phrase
            curr_gen_text = [prompt_text.capitalize() + ": " + text.replace('<s>', '').replace('</s>', '') for text, prompt_text in zip(generated_text, text_list)]
            final_text = torch.stack([clip.tokenize(text) for text in curr_gen_text])
            if bounded_text != []:
                bounded_test = [prompt_text.capitalize() + ": " + text.replace('<s>', '').replace('</s>', '') for text, prompt_text in zip(bounded_text, text_list)]
                final_bounded_text = torch.stack([clip.tokenize(text) for text in bounded_test])
                final_texts_dict[CAPTION].append(final_text.squeeze(dim=1))
                final_texts_dict[REGION_TO_DESCRIPTION].append(final_bounded_text.squeeze(dim=1))
            else:
                # List of 16 texts 
                final_texts.append(final_text)
        
        if bounded_text == []:
            final_texts = torch.stack(final_texts)
            final_texts = final_texts.view(-1, final_texts.shape[-1])
            return final_texts
        else:
            final_texts = torch.stack(final_texts_dict[CAPTION])
            final_texts = final_texts.view(-1, final_texts.shape[-1])

            final_bounded_texts = torch.stack(final_texts_dict[REGION_TO_DESCRIPTION])
            final_bounded_texts = final_bounded_texts.view(-1, final_bounded_texts.shape[-1])
            return final_texts, final_bounded_texts

    # print([len(text.split()) for text in generated_text])
    
    # final_text += ' This is a ' + re.sub(r"<loc_\d+>", "", generated_text[1][0]).replace('<s>', '').replace('</s>', '')
    # print(f"generated_text ---> {generated_text}")

    # if debug:
    #     with open('test.txt', 'w') as f:
    #         fig = plt.figure()
    #         plt.imshow(image)
    #         plt.title(label)
    #         plt.axis('off')
    #         fig.text(0.5, 0.05, final_text, wrap=True, ha='center')
    #         plt.savefig(f'test.png')

def validate(epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug, processor, flo_model, text_aug_dict):
    model.eval()
    fusion_model.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0

    labeled_ids = []
    correct_ids = []
    classes_aug = [v for k, v in text_aug_dict.items()]
    with torch.no_grad():
        for iii, data in enumerate(tqdm(val_loader)):
            
            image, orig_videos, generated_images, bbs, class_id = data
            orig_videos = orig_videos.to(device)
            class_id = class_id.to(device)
            image = image.view((-1,config.data.num_segments,3)+image.size()[-2:])

            if test1:
                final_text = run_example(generated_images, processor, flo_model, classes_aug, test=True)
                final_text = final_text.to(device)
                text_features = model.encode_text(final_text)
            elif test3:
                final_text, bounded_text = run_example(generated_images, processor, flo_model, classes_aug, test=True, bbs=bbs)
                final_text = final_text.to(device)
                bounded_text = bounded_text.to(device)
                text_features = model.encode_text(final_text)
                text_bounded_features = model.encode_text(bounded_text)
            else:
                text_inputs = classes.to(device)
                text_features = model.encode_text(text_inputs)




            # image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input)
            image_features = image_features.view(b,t,-1)
            image_features = fusion_model(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            logits_per_image = (100.0 * image_features @ text_features.T)
            similarity = calculate_similarity(logits_per_image, b, num_text_aug)

            if config.data.use_orig:
                orig_videos = orig_videos.squeeze(dim=1)
                image_input = orig_videos.to(device).view(-1, c, h, w)
                image_features = model.encode_image(image_input)
                image_features = image_features.view(b,t,-1)
                image_features = fusion_model(image_features)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                if test3:
                    text_bounded_features /= text_bounded_features.norm(dim=-1, keepdim=True)
                else:
                    text_bounded_features = text_features
                logits_per_image = (100.0 * image_features @ text_bounded_features.T)
                similarity = similarity + calculate_similarity(logits_per_image, b, num_text_aug)

            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            num += b
            for i in range(b):
                if indices_1[i] == class_id[i]:
                    corr_1 += 1
                if class_id[i] in indices_5[i]:
                    corr_5 += 1
    
            yhat = torch.argmax(similarity, dim=1).to(dtype=int)
            labeled_ids.extend(yhat.tolist())
            correct_ids.extend(class_id.tolist())

    plot_confusion_matrix(correct_ids, labeled_ids, np.array(["Using a Book", "Teacher Sitting", "Teacher Standing", "Teacher Writing", "Using Technology", "Using a Worksheet"]), config.test_name)

    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100
    wandb.log({"top1": top1})
    wandb.log({"top5": top5})
    print('Epoch: [{}/{}]: Top1: {}, Top5: {}'.format(epoch, config.solver.epochs, top1, top5))
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

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    transform_val = get_augmentation(False, config)

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    wandb.watch(model)
    wandb.watch(fusion_model)

    if not config.data.use_orig:
        mask_transform = get_mask_augmentation(cut_size=224, 
                                            cutn=1, 
                                            cut_pow=1., 
                                            noise_fac = 0.1)
        def collate_fn(batch):
            videos, masks, lambda_val, labels = zip(*batch)
            # Check the labels for bb
            videos, labels = torch.stack(videos), torch.tensor(labels)
            lambda_val = torch.tensor(lambda_val)
            masks = torch.stack(masks, dim=0)
            # videos = videos.view((-1,config.data.num_segments,3)+videos.size()[-2:])
            # masks = masks.view((-1,config.data.num_segments,3)+masks.size()[-2:])
            masks = masks.squeeze(dim=1)
            videos = videos.squeeze(dim=1)
            data = {'videos': videos, 'masks': masks}
            iii, aug_masks = mask_transform(data)
            # iii = iii.squeeze()
            return videos, aug_masks, lambda_val, labels

        val_data = Action_DATASETS(
                        config.data.val_list,
                        config.data.label_list, 
                        random_shift=False,
                        num_segments=config.data.num_segments,
                        image_tmpl=config.data.image_tmpl,
                        image_transform=transform_val)
        val_loader = DataLoader(
                        val_data,
                        batch_size=config.data.batch_size,
                        num_workers=config.data.workers,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=True,
                        collate_fn=collate_fn)
    else:
        def collate_fn(batch):
            cropped_videos, images, bbs, labels = zip(*batch)
            # Check the labels for bb
            cropped_videos = torch.stack(cropped_videos) 
            images = torch.stack(images) 
            labels = torch.tensor(labels)
            return cropped_videos, images, labels

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
    else:
        clip.model.convert_weights(
            model_text)  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict = text_prompt(val_data, config.prompt)

    best_prec1 = 0.0
    prec1 = validate(start_epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug)

if __name__ == '__main__':
    main()
