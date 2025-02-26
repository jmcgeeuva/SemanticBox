# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch
import sys
sys.path.insert(0, "./../explain/ml-no-token-left-behind/external/tamingtransformers/")
sys.path.append("./../explain/ml-no-token-left-behind/external/TransformerMMExplainability/")
import CLIP.clip as clip

# def text_prompt(data, file_name=None):
#     text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
#                 f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
#                 f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
#                 f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
#                 f"The man is {{}}", f"The woman is {{}}"]
#     text_dict = {}
#     num_text_aug = len(text_aug)

#     for ii, txt in enumerate(text_aug):
#         text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])

#     classes = torch.cat([v for k, v in text_dict.items()])

#     return classes, num_text_aug,text_dict


def text_prompt(data, use_clip=True, file_name='text_aug1.txt'):
    text_aug = []
    with open(file_name, 'r') as f:
        for line in f:
            text_aug.append(line.strip())

    num_text_aug = len(text_aug)

    if use_clip or not use_clip:
        text_dict = {}
        text_str = {}
        if 'longest' in file_name:
            for ii, txt in enumerate(text_aug):
                text_str[ii] = [txt.format(c, c) for i, c in data.classes]
                text_dict[ii] = torch.cat([clip.tokenize(txt.format(c, c)) for i, c in data.classes])
        else:
            for ii, txt in enumerate(text_aug):
                text_str[ii] = [txt.format(c, c) for i, c in data.classes]
                text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for i, c in data.classes])
    else:
        print('Under Construction')
        raise ValueError()

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug,text_dict, text_str