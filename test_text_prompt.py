import pytest
from modules.Text_Prompt import *
import torch
import os
from datasets import Action_DATASETS
import yaml
from dotmap import DotMap
from utils.Augmentation import get_augmentation

def test_prompt():
    filename = 'test.txt'
    aug_len = 10

    with open(filename, 'w') as f:
        for i in range(aug_len):
            f.write(f'photo of {i} {{}}\n')


    class Dummy:
        def __init__(self):
            self.class_names = ['test']
            self.ids = [0]

        @property
        def classes(self):
            return zip(self.ids, self.class_names)

    classes, text_aug_len, text_dict = text_prompt(Dummy(), filename)

    assert text_aug_len == aug_len
    assert len(list(text_dict.keys())) == aug_len
    assert classes.shape[0] == aug_len
    os.remove(filename)

def test_datalaoder():
    config_file = '../../preprocessing/configs/education/edu_train.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config = DotMap(config)
    transform_train = get_augmentation(True,config)

    train_data = Action_DATASETS(
                        config.data.train_list,
                        config.data.label_list,
                        num_segments=config.data.num_segments,
                        image_tmpl=config.data.image_tmpl,
                        random_shift=config.data.random_shift,
                        transform=transform_train)
    idx = 0
    process_data, label, images = train_data[idx]

    assert label == idx
    assert process_data.shape[0] == 24
    assert process_data.shape[1] == 224
    assert process_data.shape[2] == 224