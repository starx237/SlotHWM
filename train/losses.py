import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_ratios(step):
    ratios = {
        "burnin": 1.0,
        "rollout": 1.0,
        "slots": 1.0,
        "energy": 1.0,
        "rev": 1.0,
        "static": 1.0,
    }
    return ratios


def get_rollout_frames(step, max_rollout):
    return max_rollout


class SlotPiLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lambda_slots = getattr(config, "lambda_slots", 1.0)
        self.lambda_images = getattr(config, "lambda_images", 0.0)
        self.lambda_energy = getattr(config, "lambda_energy", 0.01)
        self.C_LC = getattr(config, "lambda_static", 0.001)
        self.lambda_rev = getattr(config, "lambda_rev", 0.1)
        self.freeze_appearance = getattr(config, "freeze_appearance", False)
        self.appearance_dim = getattr(config, "appearance_dim", 64)
        self.depth_weight = getattr(config, "depth_weight", 1.0)

    def compute_rev_loss(self, rev_pred, S_c, T, N):
        return F.mse_loss(rev_pred, S_c.detach())

    def forward(self, pred_slots, target_slots, pred_images=None,
                target_images=None, energy=None, slots_full_seq=None,
                rev_pred=None, C=None, S_c=None, ratios=None,
                depth_mask=None):
        r = ratios or {}
        ls_grad = r.get("slots", 1.0) * self.lambda_slots
        le_grad = r.get("energy", 1.0) * self.lambda_energy
        lc_grad = r.get("static", 1.0) * self.C_LC
        lr_grad = r.get("rev", 1.0) * self.lambda_rev

        # ---- slot 预测损失 ----
        rollout_decay = getattr(self.config, 'rollout_decay', 1.0)
        if depth_mask is not None:
            mask = depth_mask.unsqueeze(-1).float()
            T_rollout = mask.shape[1]
            if rollout_decay < 1.0 and T_rollout > 1:
                time_weights = torch.tensor(
                    [rollout_decay ** t for t in range(T_rollout)],
                    device=mask.device, dtype=mask.dtype
                ).view(1, T_rollout, 1, 1)
                mask = mask * time_weights
            if self.freeze_appearance:
                app_dim = self.appearance_dim
                pred_dyn = pred_slots[:, :, :, app_dim:]
                target_dyn = target_slots[:, :, :, app_dim:]
                per_elem = (pred_dyn - target_dyn) ** 2
                dw = self.depth_weight
                if dw != 1.0:
                    w = torch.tensor([1.0, 1.0, dw], device=per_elem.device)
                    per_elem = per_elem * w.view(1, 1, 1, -1)
                    eff_dims = 2.0 + dw
                else:
                    eff_dims = float(pred_dyn.shape[-1])
                masked = per_elem * mask
                slot_val_dyn = masked.sum() / (mask.sum() * eff_dims + 1e-8)
                pred_app = pred_slots[:, :, :, :app_dim].detach()
                target_app = target_slots[:, :, :, :app_dim]
                slot_val_app = F.mse_loss(pred_app, target_app)
                slot_val = slot_val_dyn + slot_val_app
                slot_val_dyn_raw = slot_val_dyn
                slot_val_app_raw = slot_val_app
                pos_elem = (pred_dyn[..., :2] - target_dyn[..., :2]) ** 2
                depth_elem = (pred_dyn[..., 2:3] - target_dyn[..., 2:3]) ** 2
                pos_val = (pos_elem * mask).sum() / (mask.sum() * 2 + 1e-8)
                depth_val = (depth_elem * mask).sum() / (mask.sum() * 1 + 1e-8)
            else:
                per_elem = (pred_slots - target_slots) ** 2
                dw = self.depth_weight
                if dw != 1.0:
                    dyn_w = torch.cat([
                        torch.ones(self.appearance_dim, device=per_elem.device),
                        torch.tensor([1.0, 1.0, dw], device=per_elem.device)
                    ])
                    per_elem = per_elem * dyn_w.view(1, 1, 1, -1)
                    eff_dims = float(self.appearance_dim) + 2.0 + dw
                else:
                    eff_dims = float(pred_slots.shape[-1])
                masked = per_elem * mask
                slot_val = masked.sum() / (mask.sum() * eff_dims + 1e-8)
                slot_val_dyn_raw = slot_val
                slot_val_app_raw = torch.tensor(0.0, device=pred_slots.device)
                pred_dyn = pred_slots[:, :, :, self.appearance_dim:]
                target_dyn = target_slots[:, :, :, self.appearance_dim:]
                pos_elem = (pred_dyn[..., :2] - target_dyn[..., :2]) ** 2
                depth_elem = (pred_dyn[..., 2:3] - target_dyn[..., 2:3]) ** 2
                pos_val = (pos_elem * mask).sum() / (mask.sum() * 2 + 1e-8)
                depth_val = (depth_elem * mask).sum() / (mask.sum() * 1 + 1e-8)
        else:
            if self.freeze_appearance:
                app_dim = self.appearance_dim
                pred_dyn = pred_slots[:, :, :, app_dim:]
                target_dyn = target_slots[:, :, :, app_dim:]
                pos_loss = F.mse_loss(pred_dyn[..., :2], target_dyn[..., :2])
                depth_loss = F.mse_loss(pred_dyn[..., 2:3], target_dyn[..., 2:3])
                dw = self.depth_weight
                slot_val_dyn = (2 * pos_loss + dw * depth_loss) / (2 + dw)
                pred_app = pred_slots[:, :, :, :app_dim].detach()
                target_app = target_slots[:, :, :, :app_dim]
                slot_val_app = F.mse_loss(pred_app, target_app)
                slot_val = slot_val_dyn + slot_val_app
                slot_val_dyn_raw = slot_val_dyn
                slot_val_app_raw = slot_val_app
                pos_val = pos_loss
                depth_val = depth_loss
            else:
                dw = self.depth_weight
                pred_dyn = pred_slots[:, :, :, self.appearance_dim:]
                target_dyn = target_slots[:, :, :, self.appearance_dim:]
                if dw != 1.0:
                    app_loss = F.mse_loss(pred_slots[:, :, :, :self.appearance_dim],
                                          target_slots[:, :, :, :self.appearance_dim])
                    pos_loss = F.mse_loss(pred_dyn[..., :2], target_dyn[..., :2])
                    depth_loss = F.mse_loss(pred_dyn[..., 2:3], target_dyn[..., 2:3])
                    eff_dims = float(self.appearance_dim) + 2.0 + dw
                    slot_val = (self.appearance_dim * app_loss + 2 * pos_loss + dw * depth_loss) / eff_dims
                else:
                    slot_val = F.mse_loss(pred_slots, target_slots)
                slot_val_dyn_raw = slot_val
                slot_val_app_raw = torch.tensor(0.0, device=pred_slots.device)
                pos_val = F.mse_loss(pred_dyn[..., :2], target_dyn[..., :2])
                depth_val = F.mse_loss(pred_dyn[..., 2:3], target_dyn[..., 2:3])

        image_val = torch.tensor(0.0, device=pred_slots.device)
        if pred_images is not None and target_images is not None:
            image_val = F.mse_loss(pred_images, target_images)

        energy_val = torch.tensor(0.0, device=pred_slots.device)
        if energy is not None and len(energy) >= 1:
            losses = [F.mse_loss(e[0], e[1]) for e in energy]
            energy_val = sum(losses) / len(losses)

        static_raw_local = torch.tensor(0.0, device=pred_slots.device)
        if S_c is not None and self.C_LC > 0:
            static_features = S_c.detach()
            variance = static_features.var(dim=1, unbiased=False)
            s2_m = variance.mean(dim=-1)
            N = S_c.shape[2]
            static_raw_local = self.C_LC * s2_m.sum(dim=-1).mean() / N
        static_grad = r.get("static", 1.0) * static_raw_local

        rev_raw = torch.tensor(0.0, device=pred_slots.device)
        if rev_pred is not None and S_c is not None and self.lambda_rev > 0:
            T = rev_pred.shape[1]
            N = rev_pred.shape[2]
            rev_raw = self.lambda_rev * self.compute_rev_loss(rev_pred, S_c, T, N)
        rev_grad = r.get("rev", 1.0) * rev_raw

        total_grad = (ls_grad * slot_val +
                      self.lambda_images * image_val +
                      le_grad * energy_val +
                      static_grad + rev_grad)

        aux = {
            "slot_loss": (self.lambda_slots * slot_val).item(),
            "slot_loss_dyn": (self.lambda_slots * slot_val_dyn_raw).item(),
            "slot_loss_app": (self.lambda_slots * slot_val_app_raw).item(),
            "slot_loss_pos": (self.lambda_slots * pos_val).item(),
            "slot_loss_depth": (self.lambda_slots * depth_val).item(),
            "energy_loss": (self.lambda_energy * energy_val).item(),
            "static_loss": static_raw_local.item(),
            "rev_loss": rev_raw.item(),
        }
        if depth_mask is not None:
            aux["depth_mask_ratio"] = depth_mask.float().mean().item()
        return total_grad, aux
