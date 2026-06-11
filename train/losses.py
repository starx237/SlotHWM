import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_ratios(step):
    """返回各 loss 系数在给定步数的缩放比例（默认 1.0 = 使用配置文件原始值）。
    实际系数 = config_value * ratio(step)。
    每个比例的公式可单独自定义。"""
    ratios = {
        # "burnin": 1.0,
        # "rollout": 1.0 - 0.5 * max(0.0, min(1.0, (step - 5000) / (50000 - 5000))),
        # "slots": min(1.0, step / 40000),
        # "energy": min(1.0, step / 20000),
        # "rev": 1.0,
        # "static": min(1.0, step / 20000),
        "burnin": 1.0,
        "rollout": 1.0,
        "slots": 1.0,
        "energy": 1.0,
        "rev": 1.0,
        "static": 1.0,
    }
    return ratios


def get_rollout_frames(step, max_rollout):
    """返回当前步数应使用的 rollout 帧数（最大值不超过 max_rollout）。
    用于逐步增加预测长度等调度策略。"""
    # return 2 + 2 * (step >= 10000) + 2 * (step >= 30000)
    return max_rollout


class SlotPiLoss(nn.Module):
    '''SlotPi 损失函数。
    包含 slot 预测损失、静态特征方差正则 L_LC、GRL 反向预测损失 L_rev。
    所有 aux 输出值均为已乘系数的加权值，直接反映对 total 的贡献。'''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lambda_slots = getattr(config, "lambda_slots", 1.0)   # slot 预测损失权重
        self.lambda_images = getattr(config, "lambda_images", 0.0)
        self.lambda_energy = getattr(config, "lambda_energy", 0.01)
        self.C_LC = getattr(config, "lambda_static", 0.001)        # 静态特征方差正则权重（idea.md §3）
        self.lambda_rev = getattr(config, "lambda_rev", 0.1)       # GRL 反向预测损失权重（idea.md §4）

    def compute_rev_loss(self, rev_pred, S_c, T, N):
        '''L_rev = 1/(N·T) Σ||MLP_rev(GRL(S^d)) - StopGradient(S^c)||²。
           F.mse_loss 已对所有维度取平均，无需额外缩放。'''
        return F.mse_loss(rev_pred, S_c.detach())

    def forward(self, pred_slots, target_slots, pred_images=None,
                target_images=None, energy=None, slots_full_seq=None,
                rev_pred=None, C=None, S_c=None, ratios=None):
        r = ratios or {}
        # 梯度版本系数（用于 backward）
        ls_grad = r.get("slots", 1.0) * self.lambda_slots
        le_grad = r.get("energy", 1.0) * self.lambda_energy
        lc_grad = r.get("static", 1.0) * self.C_LC
        lr_grad = r.get("rev", 1.0) * self.lambda_rev

        # ---- slot 预测损失 ----
        slot_val = F.mse_loss(pred_slots, target_slots)

        # ---- 图像重建损失（默认关闭） ----
        image_val = torch.tensor(0.0, device=pred_slots.device)
        if pred_images is not None and target_images is not None:
            image_val = F.mse_loss(pred_images, target_images)

        # ---- 能量守恒损失（energy 为 [(E_before, E_after), ...]，每 rollout 步一对）----
        energy_val = torch.tensor(0.0, device=pred_slots.device)
        if energy is not None and len(energy) >= 1:
            losses = [F.mse_loss(e[0], e[1]) for e in energy]
            energy_val = sum(losses) / len(losses)

        # ---- 静态特征方差正则 L_LC（梯度不传播，仅监控） ----
        static_raw = torch.tensor(0.0, device=pred_slots.device)
        if slots_full_seq is not None and self.C_LC > 0:
            D_sta = getattr(self.config, 'static_dim', slots_full_seq.shape[-1] // 2)
            static_features = slots_full_seq[..., :D_sta].detach()
            variance = static_features.var(dim=1, unbiased=False)
            s2_m = variance.mean(dim=-1)
            N = slots_full_seq.shape[2]
            static_raw = self.C_LC * s2_m.sum(dim=-1).mean() / N
        static_grad = r.get("static", 1.0) * static_raw

        # ---- GRL 反向预测损失 L_rev（idea.md §4） ----
        rev_raw = torch.tensor(0.0, device=pred_slots.device)
        if rev_pred is not None and S_c is not None and self.lambda_rev > 0:
            T = rev_pred.shape[1]
            N = rev_pred.shape[2]
            rev_raw = self.lambda_rev * self.compute_rev_loss(rev_pred, S_c, T, N)
        rev_grad = r.get("rev", 1.0) * rev_raw

        # ---- 梯度更新用的加权和 ----
        total_grad = (ls_grad * slot_val +
                      self.lambda_images * image_val +
                      le_grad * energy_val +
                      static_grad + rev_grad)

        # ---- 监控用原始系数值 ----
        aux = {
            "slot_loss": (self.lambda_slots * slot_val).item(),
            "energy_loss": (self.lambda_energy * energy_val).item(),
            "static_loss": static_raw.item(),
            "rev_loss": rev_raw.item(),
        }
        return total_grad, aux
