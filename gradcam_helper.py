import torch
from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import sys
sys.path.insert(0, "./../explain/ml-no-token-left-behind/external/tamingtransformers")
sys.path.append("./../explain/ml-no-token-left-behind/external/TransformerMMExplainability")
# from ntlb import load_classifier
from torch import nn
from dotmap import DotMap
import yaml
from modules.Visual_Prompt import visual_prompt
from CLIP.clip import clip
from prompt import *
from datasets import Action_DATASETS, Action_DATASETS_orig
from torchvision import transforms
import cv2
from utils import tools
import matplotlib.cm as cm

def print_frames(video, grayscale_cam, idx=0, to_screen=True):
    for i in range(8):
        np.max(video.cpu().numpy())
        # print(len(video[0][i].cpu().numpy()))
        # print(label)
        input_vid = video[i].cpu().permute(1, 2, 0).numpy()
        visualization = show_cam_on_image(input_vid, grayscale_cam[-1, ...], use_rgb=True)
        # You can also get the model outputs without having to redo inference
        cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        # model_outputs = cam.outputs

        # gnd_truth = 'Havanese'
        # dog_pred = train_subset._label_names[int(argmax)]

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow((input_vid + 1) / 2)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(cam_image)
        plt.axis('off')

        if to_screen:
            plt.show()
        else:
            plt.savefig(f'./images/gradcam/grad{idx}_{i}.png')

class PredModel(torch.nn.Module):
    def __init__(self, model, fusion_model, classes, device, cropped=False):
        super(PredModel, self).__init__()
        self.model = model
        self.fusion_model = fusion_model
        self.classes = classes
        self.device = device
        self.cropped = cropped

    def forward(self, x):
        t, c, h,w = x.shape
        if self.cropped:
            x, x_cropped = torch.split(x, t//2, dim=0)
            t = t//2
        b = 1
        text_inputs = self.classes.to(self.device)
        text_features = self.model.encode_text(text_inputs)
        x = x.unsqueeze(0)
        image_features = self.model.encode_image(x.squeeze())
        image_features = image_features.view(b, t,-1)
        image_features = self.fusion_model(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if self.cropped:
            image_features2 = self.model.encode_image(x_cropped.squeeze())
            image_features2 = image_features2.view(b, t,-1)
            image_features2 = self.fusion_model(image_features2)
            image_features2 = image_features2 / image_features2.norm(dim=-1, keepdim=True)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image, logits_per_text = create_logits(image_features,text_features,logit_scale)
        if self.cropped:
            logits_per_image2, logits_per_text = create_logits(image_features2,text_features,logit_scale)
            logits_per_image = logits_per_image + logits_per_image2
        return logits_per_image

def reshape_transform(tensor, height=27, width=27):
    # t = tensor[0]
    b, t, c = tensor.shape
    print(tensor.shape)
    # result = tensor.reshape(b, -1)
    # result = result[:, 0:result.size(1)-60]
    result = tensor.permute(0, 2, 1)
    print(result.shape)
    result = result[:, 0:729, :]
    print(result.shape)
    result = result.reshape(b, height, width, t)
    print(result.shape)

    # result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_gradcamplusplus(video, model, fusion_model, target_layers, targets, reshape_transform, classes, device, cropped_video=None, cropped=False):
  pred_model = PredModel(model, fusion_model, classes, device, cropped=cropped)
  pred_model.eval()
  # reshape_transform = transforms.Compose([])#ThisTransform()])
  # Construct the CAM object once, and then re-use it on many images.
  with GradCAMPlusPlus(model=pred_model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
    cam.batch_size = 1
    print(f'Max: {torch.max(video)}, {video.shape}')
  #   torch.autograd.set_detect_anomaly(True)
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # print(video.shape)
    if cropped_video != None:
        combined_matrix = torch.cat((video, cropped_video), dim=0)
    else:
        combined_matrix = torch.cat((video, video), dim=0)
    print(combined_matrix.shape)
    grayscale_cam = cam(input_tensor=combined_matrix, targets=targets)
    # In this example grayscale_cam has only one image in the batch:
    # print(grayscale_cam.shape)
    
  return grayscale_cam

def load_classifier(pretrain, model, fusion_model):
    if os.path.isfile(pretrain):
        print(("=> loading checkpoint '{}'".format(pretrain)))
        checkpoint = torch.load(pretrain)
        model.load_state_dict(checkpoint['model_state_dict'])
        fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
        del checkpoint
    else:
        print(("=> no checkpoint found at '{}'".format(pretrain)))

    return model, fusion_model

def get_models(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                    T=config.data.num_segments, dropout=config.network.drop_out,
                                                    emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32


    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    model, fusion_model = load_classifier(config.pretrain, model, fusion_model)
    model_image.requires_grad_(True)
    model_text.requires_grad_(True)
    fusion_model.requires_grad_(True)
    model.requires_grad_(True)
    return model, config, fusion_model, model_text, model_image

# def run_gradcam(model, fusion_model, num_text_aug, video, classes_names, classes, device, idx=0):
#     model2 = PredModel(model, fusion_model, classes, device)
#     # model2 = model2.to(device)
#     # video = video.to(device)
#     logits_per_image, logits_per_text = model2(video)
#     similarity = logits_per_image.view(1, num_text_aug, -1).softmax(dim=-1)
#     similarity = similarity.mean(dim=1, keepdim=False)
#     answer = torch.argmax(similarity, dim=1).to(dtype=int)
#     print(f'Predicted label: {classes_names[int(answer)][1]}')

def run_gradcam(model, fusion_model, video, answer, idx, classes, device, cropped_video=None, cropped=False):
    # We have to specify the target we want to generate the CAM for.
    index = int(answer)
    targets = [ClassifierOutputTarget(index)]
    target_layer = model.visual.transformer.resblocks[0].ln_1
    target_layers = [target_layer]
    grayscale_cam = get_gradcamplusplus(video, model, fusion_model, target_layers, targets, reshape_transform, classes, device, cropped_video, cropped)
    # grayscale_cam.shape
    print_frames(video, grayscale_cam, idx)
    if cropped:
        print_frames(cropped_video, grayscale_cam, idx)