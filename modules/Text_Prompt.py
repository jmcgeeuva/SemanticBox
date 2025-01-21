# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch
import clip

def text_prompt(data, file_name='text_aug1.txt'):
    text_aug = []
    with open(file_name, 'r') as f:
        for line in f:
            text_aug.append(line.strip())

    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug,text_dict