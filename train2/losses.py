import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotPiLoss(nn.Module):
    '''
    Slot 预测联合损失函数（idea.md §3 修正版），包含四项：
      1. L_S = MSE(pred_slots, target_slots)
      2. L_I = MSE(pred_images, target_images)
      3. L_E = MSE(H_t, H_{t-1})
      4. L_C = C_LC * (1/(N*T)) * sum_m (variance of S^c_m over time T)
         其中 S^c 是 slot 的前半部分（静态特征），
         variance 沿时间维计算（包括 burnin 和 rollout 的所有时间步）
    '''
    def __init__(self, config):
        super().__init__()
        self.lambda_slots = getattr(config, "lambda_slots", 1.0)
        self.lambda_images = getattr(config, "lambda_images", 0.0)
        self.lambda_energy = getattr(config, "lambda_energy", 0.01)
        # C_LC: 静态特征方差正则化系数（idea.md §3）
        self.C_LC = getattr(config, "lambda_static", 0.001)

    def forward(self, pred_slots, target_slots, pred_images=None,
                target_images=None, energy=None, slots_full_seq=None):
        '''
        Args:
            pred_slots: 预测的 rollout slots, (B, T_roll, N, D)
            target_slots: 真实 rollout slots, (B, T_roll, N, D)
            pred_images: 预测图像 (optional)
            target_images: 真实图像 (optional)
            energy: 能量值列表 (optional)
            slots_full_seq: 完整 slots 序列（burnin+rollout）, (B, T_full, N, D)
        Returns:
            total_loss: 标量
            aux: dict, 各损失分量
        '''
        # L_S: Slot 重建损失
        slot_loss = F.mse_loss(pred_slots, target_slots)

        # L_I: 图像重建损失
        image_loss = torch.tensor(0.0, device=pred_slots.device)
        if pred_images is not None and target_images is not None:
            image_loss = F.mse_loss(pred_images, target_images)

        # L_E: 能量正则化损失
        energy_loss = torch.tensor(0.0, device=pred_slots.device)
        if energy is not None and len(energy) >= 2:
            energy_loss = F.mse_loss(energy[0], energy[1])

        # L_C: 静态特征方差正则化 (idea.md §3)
        static_loss = torch.tensor(0.0, device=pred_slots.device)
        if slots_full_seq is not None and self.C_LC > 0:
            # 取 slot 的前半部分作为静态特征 S^c
            D_sta = slots_full_seq.shape[-1] // 2
            static_features = slots_full_seq[..., :D_sta]  # (B, T, N, D')
            # 沿时间维计算每个物体 S^c 的方差
            variance = static_features.var(dim=1, unbiased=False)  # (B, N, D')
            # 在特征维上求均值，得到 s^2_m (每个物体一个标量方差)
            s2_m = variance.mean(dim=-1)  # (B, N)
            # L_C = C_LC * (1/(N*T)) * sum_m s^2_m
            # 其中 T = slots_full_seq.shape[1]
            T_full = slots_full_seq.shape[1]
            N = slots_full_seq.shape[2]
            static_loss = self.C_LC * s2_m.sum(dim=-1).mean() / (N * T_full)

        total_loss = (self.lambda_slots * slot_loss +
                      self.lambda_images * image_loss +
                      self.lambda_energy * energy_loss +
                      static_loss)

        aux = {
            "slot_loss": slot_loss.item(),
            "image_loss": image_loss.item(),
            "energy_loss": energy_loss.item(),
            "static_loss": static_loss.item(),
        }
        return total_loss, aux