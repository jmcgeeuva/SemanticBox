# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
import pdb
import io
import time
import pandas as pd
import torchvision
import random
from PIL import Image, ImageOps
import cv2
import numbers
import math
import torch
from randaugment import RandAugment

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]
    
class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                print(len(img_group))
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                print(len(img_group))
                rst = np.concatenate(img_group, axis=2)
                return rst

    
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class Action_DATASETS(data.Dataset):
    def __init__(self, list_file, labels_file,
                 num_segments=1, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, index_bias=1, height=224, width=224):

        self.list_file = list_file
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop=False
        self.index_bias = index_bias
        self.labels_file = labels_file
        self.height = height
        self.width = width

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        self._parse_list()
        self.initialized = False

    def _load_image(self, directory, idx):

        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
    
    def _load_teacher_box(self, annotation, idx):
        teacher_box = self.annotation['frames'][frame]['teacher_box']
        x0, y0, x1, y1 = teacher_box.round().int()

        # Ensure the coordinates are within valid range
        x0 = x0.clamp(0, self.width)
        y0 = y0.clamp(0, self.height)
        x1 = x1.clamp(0, self.width)
        y1 = y1.clamp(0, self.height)
        return teacher_box
    
    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()
    
    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(
                    self.total_length) + randint(record.num_frames // 2),
                    record.num_frames) + self.index_bias
            offsets = np.concatenate((
                np.arange(record.num_frames),
                randint(record.num_frames, size=self.total_length - record.num_frames)))
            return np.sort(offsets) + self.index_bias
        offsets = list()
        ticks = [i * record.num_frames // self.num_segments
                 for i in range(self.num_segments + 1)]

        for i in range(self.num_segments):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= self.seg_length:
                tick += randint(tick_len - self.seg_length + 1)
            offsets.extend([j for j in range(tick, tick + self.seg_length)])
        return np.array(offsets) + self.index_bias

    def _get_val_indices(self, record):
        if self.num_segments == 1:
            return np.array([record.num_frames //2], dtype=int) + self.index_bias
        
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias
            return np.array([i * record.num_frames // self.total_length
                             for i in range(self.total_length)], dtype=int) + self.index_bias
        offset = (record.num_frames / self.num_segments - self.seg_length) / 2.0
        return np.array([i * record.num_frames / self.num_segments + offset + j
                         for i in range(self.num_segments)
                         for j in range(self.seg_length)], dtype=int) + self.index_bias

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        return self.get(record, segment_indices)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def adjust_bb(self, boxes, target_width, target_height):
        # Compute the center of each bounding box
        x_center = (boxes[:, 0] + boxes[:, 2]) / 2
        y_center = (boxes[:, 1] + boxes[:, 3]) / 2

        # Compute half-width and half-height, handling odd dimensions
        half_width_left = np.floor(target_width / 2)
        half_width_right = np.ceil(target_width / 2)
        half_height_top = np.floor(target_height / 2)
        half_height_bottom = np.ceil(target_height / 2)

        # Create new bounding box coordinates
        new_x0 = x_center - half_width_left
        new_y0 = y_center - half_height_top
        new_x1 = x_center + half_width_right
        new_y1 = y_center + half_height_bottom

        # Stack the new coordinates into the final tensor
        new_boxes = torch.stack([new_x0, new_y0, new_x1, new_y1], dim=1)
        return new_boxes


    def get(self, record, indices):
        images = list()
        bbs = list()
        
        with open(os.path.join(record.path, 'annotation.json')) as f:
            annotation = json.load(f, object_pairs_hook=OrderedDict)

        target_width = 0
        target_height = 0
        for i, seg_ind in enumerate(indices):
            p = int(seg_ind)
            try:
                seg_imgs = self._load_image(record.path, p)
                bb = self._load_teacher_box(annotation, p)
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(seg_imgs)
            width = bb[2] - bb[0]
            if width > target_width:
                target_width = width
            height = bb[3] - bb[0]
            if height > target_height:
                target_height = height
            bbs.append(bb)

        bbs = self.adjust_bb(tensor.stack(bbs))
        cropped_images = images[:, y0:y1, x0:x1]
        process_data = self.transform(cropped_images)

        import pdb; pdb.set_trace()
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
