import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotPiLoss(nn.Module):
    '''SlotPi 损失函数。
    包含 slot 预测损失、静态特征方差正则 L_LC、GRL 反向预测损失 L_rev。
    所有 aux 输出值均为已乘系数的加权值，直接反映对 total 的贡献。'''
    def __init__(self, config):
        super().__init__()
        self.lambda_slots = getattr(config, "lambda_slots", 1.0)   # slot 预测损失权重
        self.lambda_images = getattr(config, "lambda_images", 0.0)
        self.lambda_energy = getattr(config, "lambda_energy", 0.01)
        self.C_LC = getattr(config, "lambda_static", 0.001)        # 静态特征方差正则权重（idea.md §3）
        self.lambda_rev = getattr(config, "lambda_rev", 0.1)       # GRL 反向预测损失权重（idea.md §4）

    def compute_rev_loss(self, rev_pred, C, T, N):
        '''L_rev = MSE(MLP_rev(GRL(S^d)), StopGradient(C)) / (N·T)'''
        C_expanded = C.unsqueeze(1).expand(-1, T, -1, -1).detach()
        return F.mse_loss(rev_pred, C_expanded) / (N * T)

    def forward(self, pred_slots, target_slots, pred_images=None,
                target_images=None, energy=None, slots_full_seq=None,
                rev_pred=None, C=None):
        # ---- slot 预测损失 ----
        slot_loss = F.mse_loss(pred_slots, target_slots)

        # ---- 图像重建损失（默认关闭） ----
        image_loss = torch.tensor(0.0, device=pred_slots.device)
        if pred_images is not None and target_images is not None:
            image_loss = F.mse_loss(pred_images, target_images)

        # ---- 能量守恒损失（默认关闭） ----
        energy_loss = torch.tensor(0.0, device=pred_slots.device)
        if energy is not None and len(energy) >= 2:
            energy_loss = F.mse_loss(energy[0], energy[1])

        # ---- 静态特征方差正则 L_LC（idea.md §3） ----
        static_loss = torch.tensor(0.0, device=pred_slots.device)
        if slots_full_seq is not None and self.C_LC > 0:
            D_sta = slots_full_seq.shape[-1] // 2
            static_features = slots_full_seq[..., :D_sta]
            variance = static_features.var(dim=1, unbiased=False)
            s2_m = variance.mean(dim=-1)
            N = slots_full_seq.shape[2]
            static_loss = self.C_LC * s2_m.sum(dim=-1).mean() / N

        # ---- GRL 反向预测损失 L_rev（idea.md §4） ----
        rev_loss = torch.tensor(0.0, device=pred_slots.device)
        if rev_pred is not None and C is not None and self.lambda_rev > 0:
            T = rev_pred.shape[1]
            N = rev_pred.shape[2]
            rev_loss = self.lambda_rev * self.compute_rev_loss(rev_pred, C, T, N)

        # ---- 总损失（所有项都已加权） ----
        total_loss = (self.lambda_slots * slot_loss +
                      self.lambda_images * image_loss +
                      self.lambda_energy * energy_loss +
                      static_loss + rev_loss)

        aux = {
            "slot_loss": (self.lambda_slots * slot_loss).item(),
            "static_loss": static_loss.item(),
            "rev_loss": rev_loss.item(),
        }
        return total_loss, aux
