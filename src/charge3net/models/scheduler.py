# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import torch.optim as optim
import torch.nn as nn
import torch

class PowerDecayScheduler(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, alpha=0.96, beta=1e5):
        scheduler_fn = lambda step: alpha ** (step / beta)
        super().__init__(optimizer=optimizer, lr_lambda=scheduler_fn)

class HybridLoss(nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha  # 控制过渡比例
        self.mae = nn.L1Loss()

    def forward(self, y_pred, y_true):
        # MAE部分
        mae_loss = self.mae(y_pred, y_true)
        
        # sMAPE部分（解决含负值问题）
        epsilon = 1e-6  # 避免除零
        smape = 2 * torch.abs(y_true - y_pred) / (torch.abs(y_true) + torch.abs(y_pred) + epsilon)
        smape_loss = torch.mean(smape)
        
        # 混合损失
        return (1 - self.alpha) * mae_loss + self.alpha * smape_loss
