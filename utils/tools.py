# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import numpy

def gen_label(labels):
    num = len(labels)
    gt = numpy.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def convert_models_to_fp16(model):
    print(model)
    for p in model.parameters():
        p.data = p.data.half()
        p.grad.data = p.grad.data.half()


def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2


def create_cropped_logits(x1, x2, bnd_emb, lambda_img, lambda_bnd, logit_scale):
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    if lambda_bnd > 0 and lambda_img > 0:
        # lambda_img = 1  and lambda_bnd = 0 : only frame features
        # lambda_img = .5 and lambda_bnd = .5: average of frame and bb features
        # lambda_img = 0  and lambda_bnd = 1 : only bounding box features
        bnd_emb = bnd_emb / bnd_emb.norm(dim=-1, keepdim=True)
        x1 = (lambda_img*x1 + lambda_bnd*bnd_emb)
    elif lambda_bnd > 0:
        x1 = bnd_emb / bnd_emb.norm(dim=-1, keepdim=True)
    else:
        x1 = x1 / x1.norm(dim=-1, keepdim=True)


    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2