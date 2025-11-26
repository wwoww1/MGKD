import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillKL_Attn(nn.Module):
    """
    知识蒸馏损失（KL蒸馏 + 注意力蒸馏）
    alpha: KL logits loss 权重
    beta:  注意力对齐损失权重
    """
    # def __init__(self, T=4.0, alpha=0.5, beta=0.5):
    def __init__(self, T=4.0):
        super(DistillKL_Attn, self).__init__()
        self.T = T
        self.mse = nn.MSELoss()

    def forward(self, y_s, y_t, attn_s=None, attn_t=None):
        # # KL 蒸馏损失（Logits distillation）
        # p_s = F.log_softmax(y_s / self.T, dim=1)
        # p_t = F.softmax(y_t / self.T, dim=1)
        # loss_kd = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)

        # 注意力图 MSE 损失
        loss_attn = 0.0
        if (attn_s is not None) and (attn_t is not None):
            loss_attn = self.mse(attn_s, attn_t)

        # 融合总损失
        loss_attention = loss_attn
        return loss_attention