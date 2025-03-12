import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch
from PIL import Image, ImageOps, ImageFilter
from enum import Enum
from torch import nn
import kornia.augmentation as K

class DATASET(Enum):
    MNIST = 1
    FLOWERS = 2
    CIFAR = 3
    OXFORD_PET = 4


class AddNoise(object):

    def __init__(self, noise_fac, cutn):
        self.noise_fac = 0.1
        self.cutn = cutn

    def __call__(self, data):
        batch, augmented_masks = data
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch, augmented_masks

# class Stack(object):

#     def __init__(self):
#         pass

#     def __call__(self, data):
#         cutouts, augmented_masks = data
#         if type(cutouts) != list:
#             batch = torch.stack([cutouts], dim=0)
#         else:
#             batch = torch.stack(cutouts, dim=0)
#         augmented_masks = torch.stack(augmented_masks, dim=0)
#         return batch, augmented_masks

class ImageAugmentations(object):

    def __init__(self):
        self.img_augs = nn.Sequential(
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1/.3), p=0.7),
        )

    def __call__(self, data):
        batch, augmented_masks = data
        batch = self.img_augs(batch)
        return batch, augmented_masks

# class GroupNormalize(object):
#     def __init__(self):
#         self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
#                                     std=[0.26862954, 0.26130258, 0.27577711])

#     def __call__(self, data):
#         batch, augmented_masks = data
#         batch, augmented_masks = self.normalize(batch), augmented_masks
#         return batch, augmented_masks

class GroupAug(object):
    """Randomly Grayscale flips the given PIL.Image with a probability
    """
    def __init__(self, cut_size, cutn, cut_pow, noise_fac=0.1):
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border', same_on_batch=True),
            K.RandomPerspective(0.7,p=0.7, same_on_batch=True)
        )
        # self.img_augs = ImageAugmentations(is_classifier)
        self.add_noise = AddNoise(noise_fac=noise_fac, cutn=cutn)
        self.frame_len = 8
        self.av_pool = nn.AdaptiveAvgPool3d((self.frame_len, self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool3d((self.frame_len, self.cut_size, self.cut_size))

    def __call__(self, data):
        input_img, input_masks = data['videos'], data['masks']
        cutouts = []
        augmented_masks = []

        masks = []

        permuted_input_img = input_img.permute(0, 2, 1, 3, 4) # switch time and channel
        for mask_i in range(len(input_masks)):
            curr_mask = input_masks[mask_i].permute(1, 0, 2, 3)
            curr_mask = (self.av_pool(curr_mask) + self.max_pool(curr_mask))/2
            curr_mask = curr_mask.unsqueeze(0)
            masks.append(curr_mask)
        cutout = (self.av_pool(permuted_input_img) + self.max_pool(permuted_input_img))/2

        sample = torch.cat([cutout] + masks, dim=0)
        
        beg_neutral_masks = len([cutout] + masks)

        sample = sample.permute(0, 2, 1, 3, 4) # switch time and channel back
        b, t, c, h, w = sample.size()
        sample = sample.reshape(-1, c, h, w)
        aug_sample = self.augs(sample)
        # (128+128)x3x224x224
        aug_sample = aug_sample.reshape(b, t, c, h, w)
        cutouts.append(aug_sample[0:input_img.shape[0]])

        curr_augmented_masks = aug_sample[input_img.shape[0]::]
        curr_augmented_masks = curr_augmented_masks[:, :, 0:1, ...]
        curr_augmented_masks = torch.round(curr_augmented_masks)
        augmented_masks.append(curr_augmented_masks)

        batch = torch.stack(cutouts, dim=0)
        batch = batch.reshape(-1, c, h, w)
        batch = batch.squeeze()
        augmented_masks = torch.stack(augmented_masks, dim=0)
        augmented_masks = augmented_masks.reshape(-1, 1, h, w)
        augmented_masks = augmented_masks.squeeze(dim=0)
        
        batch, augmented_masks = self.add_noise((batch, augmented_masks))
        batch = batch.reshape(input_img.shape[0], t, c, h, w)
        augmented_masks = augmented_masks.reshape(input_img.shape[0], t, 1, h, w)
        return batch, augmented_masks


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size
        out_images = list()
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images

class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, data):
        img_group, img_mask = data['video'], data['mask']
        if img_mask != None:
            img_mask = [self.worker(img) for img in img_mask]
        return {'video': [self.worker(img) for img in img_group], 'mask': img_mask}
    
class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_sth=False):
        self.is_sth = is_sth

    def random_horizontal_flip(self, img_group, v):
        if not self.is_sth and v < 0.5:
            
            img_group = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]

        return img_group

    def __call__(self, data, is_sth=False):
        img_group, img_mask = data['video'], data['mask']
        v = random.random()
        img_group = self.random_horizontal_flip(img_group, v)
        if img_mask != None:
            img_mask = self.random_horizontal_flip(img_mask, v)
        return {'video': img_group, 'mask': img_mask}
    
class GroupNormalize1(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.worker = torchvision.transforms.Normalize(mean,std)

    def __call__(self, img_group):
        
        return [self.worker(img) for img in img_group]
    
        
class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.norm = torchvision.transforms.Normalize(mean=mean, std=std)

    def normalize(self, tensor):
        # masks = masks.view((-1,config.data.num_segments,3)+masks.size()[-2:])
        # mean = self.mean * (tensor.size()[0]//len(self.mean))
        # std = self.std * (tensor.size()[0]//len(self.std))
        # mean = torch.Tensor(mean)
        # std = torch.Tensor(std)

        # if len(tensor.size()) == 3:
        #     # for 3-D tensor (T*C, H, W)
        #     tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        # elif len(tensor.size()) == 4:
        #     # for 4-D tensor (C, T, H, W)
        #     tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        # return tensor
        return self.norm(tensor)

    def __call__(self, data):
        tensor, img_mask = data['video'], data['mask']
        tensor = tensor.view((-1,8,3)+tensor.size()[-2:])
        img_group = self.normalize(tensor)
        
        if img_mask != None:
            img_mask = img_mask.view((-1,8,3)+img_mask.size()[-2:])
        # Removed for masks because it hurt the accuracy by blurring out the mask
        return {'video': img_group, 'mask': img_mask}


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, data):
        img_group, img_mask = data['video'], data['mask']
        if img_mask != None:
            img_mask = [self.worker(img) for img in img_mask]
        return {'video': [self.worker(img) for img in img_group], 'mask': img_mask}


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupFCSample(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fc_fix_offset(image_w, image_h, image_h, image_h)
        oversample_group = list()

        for o_w, o_h in offsets:
            normal_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + image_h, o_h + image_h))
                normal_group.append(crop)
            oversample_group.extend(normal_group)
        return oversample_group


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def multiscale_crop(self, img_group):
        im_size = img_group[0].size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in img_group]
        return ret_img_group

    def __call__(self, data):
        img_group, img_mask = data['video'], data['mask']

        ret_img_group = self.multiscale_crop(img_group)
        if img_mask != None:
            img_mask = self.multiscale_crop(img_mask)

        return {'video': ret_img_group, 'mask': img_mask}

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    @staticmethod
    def fill_fc_fix_offset(image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 2
        h_step = (image_h - crop_h) // 2

        ret = list()
        ret.append((0, 0))  # left
        ret.append((1 * w_step, 1 * h_step))  # center
        ret.append((2 * w_step, 2 * h_step))  # right

        return ret

class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                rst = np.concatenate(img_group, axis=2)
                # plt.imshow(rst[:,:,3:6])
                # plt.show()
                return rst

class Stack1(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        
        if self.roll:
            return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
        else:
            
            rst = np.concatenate(img_group, axis=0)
            # plt.imshow(rst[:,:,3:6])
            # plt.show()
            return torch.from_numpy(rst)

class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()
    

class ToTorchFormatTensor1(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.worker = torchvision.transforms.ToTensor()
    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class IdentityTransform(object):

    def __call__(self, data):
        return data
    
# custom transforms
class GroupRandomColorJitter(object):
    """Randomly ColorJitter the given PIL.Image with a probability
    """
    def __init__(self, p=0.8, brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1):
        self.p = p
        self.worker = torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                        saturation=saturation, hue=hue)

    def random_color_jitter(self, img_group, v):
        if v < self.p:
            img_group = [self.worker(img) for img in img_group]
            
        return img_group


    def __call__(self, data):
        img_group, img_mask = data['video'], data['mask']
        v = random.random()
        img_group = self.random_color_jitter(img_group, v)
        # img_mask = self.random_color_jitter(img_mask, v)
        return {'video': img_group, 'mask': img_mask}

class GroupRandomGrayscale(object):
    """Randomly Grayscale flips the given PIL.Image with a probability
    """
    def __init__(self, p=0.2):
        self.p = p
        self.worker = torchvision.transforms.Grayscale(num_output_channels=3)

    def random_grayscale(self, img_group, v):
        if v < self.p:
            img_group = [self.worker(img) for img in img_group]
            
        return img_group

    def __call__(self, data):
        img_group, img_mask = data['video'], data['mask']
       
        v = random.random()
        img_group = self.random_grayscale(img_group, v)
        # img_mask = self.random_grayscale(img_mask, v)
        return {'video': img_group, 'mask': img_mask}

class GroupGaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def gaussian_blur(self, img_group, v, sigma_rand):
        if v < self.p:
            sigma = sigma_rand * 1.9 + 0.1
            img_group = [img.filter(ImageFilter.GaussianBlur(sigma))  for img in img_group]

        return img_group

    def __call__(self, data):
        img_group, img_mask = data['video'], data['mask']
        v = random.random()
        sigma_rand = random.random()
        img_group = self.gaussian_blur(img_group, v, sigma_rand)
        if img_mask != None:
            img_mask = self.gaussian_blur(img_mask, v, sigma_rand)
        return {'video': img_group, 'mask': img_mask}

class GroupSolarization(object):
    def __init__(self, p):
        self.p = p

    def solarization(self, img_group, v):
        if v < self.p:
            img_group = [ImageOps.solarize(img)  for img in img_group]
        return img_group

    def __call__(self, data):
        img_group, img_mask = data['video'], data['mask']
        v = random.random()

        img_group = self.solarization(img_group, v)
        if img_mask != None:
            img_mask = self.solarization(img_mask, v)
        
        return {'video': img_group, 'mask': img_mask}


class GroupStack(object):

    def __init__(self, roll):
        self.roll = roll
        self.stack = Stack(roll=roll)

    def __call__(self, data):
        img_group, img_mask = data['video'], data['mask']

        img_group = self.stack(img_group)
        if img_mask != None:
            img_mask = self.stack(img_mask)
        return {'video': img_group, 'mask': img_mask}

class GroupToTorchFormatTensor(object):

    def __init__(self, div):
        self.to_float_tensor = ToTorchFormatTensor(div=div)

    def __call__(self, data):
        img_group, img_mask = data['video'], data['mask']

        img_group = self.to_float_tensor(img_group)
        if (type(img_mask) != type(None)) and (img_mask != None).any():
            img_mask = self.to_float_tensor(img_mask)

        return {'video': img_group, 'mask': img_mask}