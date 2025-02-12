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


def create_cropped_logits(img_emb, img_emb_crp, lambda_emb, lambda_emb_crp, txt_emb, logit_scale):
    # norm_img_emb = img_emb.norm(dim=-1, keepdim=True)
    # norm_img_emb_crp = img_emb_crp.norm(dim=-1, keepdim=True)
    # img_emb_x1 = img_emb_crp*norm_img_emb + img_emb*norm_img_emb_crp
    # img_emb_x2 = (img_emb_crp*norm_img_emb).t() + (img_emb*norm_img_emb_crp).t()
    
    img_emb_crp = img_emb_crp / img_emb_crp.norm(dim=-1, keepdim=True)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * (lambda_emb_crp*img_emb_crp + lambda_emb*img_emb) @ txt_emb.t()
    logits_per_x2 = logit_scale * txt_emb @ ((lambda_emb_crp*img_emb_crp).t() + (lambda_emb*img_emb).t())

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2