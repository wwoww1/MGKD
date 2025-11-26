from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch


class SimAM(nn.Module):
    def __init__(self, lambda_val=1e-4):
        super(SimAM, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        # x: (N, C, H, W)
                
        n = x.shape[2] * x.shape[3] - 1
        mean = x.mean(dim=[2,3], keepdim=True)
        d = (x - mean).pow(2)
        v = d.sum(dim=[2,3], keepdim=True) / n
        e_inv = d / (4 * (v + self.lambda_val)) + 0.5
        attn = torch.sigmoid(e_inv)
        return x * attn, attn  

    def get_attention(self, x):

        n = x.shape[2] * x.shape[3] - 1
        mean = x.mean(dim=[2,3], keepdim=True)
        d = (x - mean).pow(2)
        v = d.sum(dim=[2,3], keepdim=True) / n
        e_inv = d / (4 * (v + self.lambda_val)) + 0.5
        return torch.sigmoid(e_inv)
    
class Masker(nn.Module):
    def __init__(self, in_dim, teacher_dim, middle=None, k=256):
        super(Masker, self).__init__()
        middle = middle or 4 * in_dim
        self.k = k
        self.ce_loss = nn.CrossEntropyLoss()
        middle = middle or 4 * (in_dim + teacher_dim)
        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim + teacher_dim, middle),
            nn.BatchNorm1d(middle),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(middle, middle),
            nn.BatchNorm1d(middle),
            nn.ReLU(inplace=True),
            nn.Linear(middle, in_dim),
            nn.BatchNorm1d(in_dim, affine=False)
        )

    def forward(self, f_student, f_teacher):

        mask_input = torch.cat([f_student, f_teacher], dim=1)
        mask_logits = self.layers(mask_input)
        z = torch.zeros_like(mask_logits)

        if self.k == 1:
            return F.gumbel_softmax(mask_logits, tau=0.5, hard=False)
        else:
            # [batch, in_dim] -> [k, batch, in_dim]
            mask_logits_expand = mask_logits.unsqueeze(0).expand(self.k, *mask_logits.shape)
            masks = F.gumbel_softmax(mask_logits_expand, tau=0.5, hard=False, dim=-1)
            z = masks.max(dim=0)[0]
            return z

class MaskedDistillKL(nn.Module):
    def __init__(self, feature_dim, teacher_dim, num_classes, T=4.0):
        super(MaskedDistillKL, self).__init__()
        self.T = T
        self.ce_loss = nn.CrossEntropyLoss()
        self.masker = Masker(in_dim=feature_dim, teacher_dim=teacher_dim)
        self.head_sup = nn.Linear(feature_dim, num_classes)
        self.head_inf = nn.Linear(feature_dim, num_classes)

    def forward(self, student_features, teacher_features, target):

        masks_sup = self.masker(student_features.detach(), teacher_features.detach())
        masks_inf = torch.ones_like(masks_sup) - masks_sup

        features_sup = student_features * masks_sup
        features_inf = student_features * masks_inf

        logits_sup = self.head_sup(features_sup)
        logits_inf = self.head_inf(features_inf)

        loss_sup = self.ce_loss(logits_sup, target)
        loss_inf = self.ce_loss(logits_inf, target)

        total_loss = 0.5 * loss_sup + 0.5 * loss_inf
        return total_loss

    def masker_loss(self, student_features, teacher_features, target):
        masks_sup = self.masker(student_features.detach(), teacher_features.detach())
        masks_inf = torch.ones_like(masks_sup) - masks_sup

        features_sup = student_features * masks_sup
        features_inf = student_features * masks_inf

        logits_sup = self.head_sup(features_sup)
        logits_inf = self.head_inf(features_inf)

        loss_sup = self.ce_loss(logits_sup, target)
        loss_inf = self.ce_loss(logits_inf, target)

        mask_loss = 0.5 * loss_sup - 0.5 * loss_inf  
        return mask_loss
    

class HCR(nn.Module):
    def __init__(self,weight=10.0):
        super(HCR, self).__init__()
        self.eps = 1e-12
        self.weight = 10.0

    def pairwise_dist(self, x):
        x_square = x.pow(2).sum(dim=1)
        prod = x @ x.t()
        pdist = (x_square.unsqueeze(1) + x_square.unsqueeze(0) - 2 * prod).clamp(min=self.eps)
        pdist[range(len(x)), range(len(x))] = 0.
        return pdist

    def pairwise_prob(self, pdist):
        return torch.exp(-pdist)

    def hcr_loss(self, h, g):
        q1, q2 = self.pairwise_prob(self.pairwise_dist(h)), self.pairwise_prob(self.pairwise_dist(g))
        return -1 * (q1 * torch.log(q2 + self.eps)).mean() + -1 * ((1 - q1) * torch.log((1 - q2) + self.eps)).mean()

    def forward(self, proj_s, proj_t,T=4.0):

        hcr_loss = self.hcr_loss(F.normalize(proj_s, dim=1), 
                                F.normalize(proj_t, dim=1).detach())

        return hcr_loss

