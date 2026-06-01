import os, glob, math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .losses import SlotPiLoss


class WandBLogger:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.run = None
        if enabled:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, disabling")
                self.enabled = False

    def init(self, **kwargs):
        if self.enabled:
            self.run = self.wandb.init(**kwargs)

    def log(self, data, step=None):
        if self.enabled and self.run is not None:
            self.wandb.log(data, step=step)

    def finish(self):
        if self.enabled and self.run is not None:
            self.wandb.finish()


def compute_gamma(step, num_steps, gamma_max, c_gamma):
    '''计算 GRL 的 gamma 系数，按 sigmoid 曲线从 0 渐升至 gamma_max（idea.md §4）。'''
    p = step / max(1, num_steps)
    return (2.0 / (1.0 + math.exp(-c_gamma * p)) - 1.0) * gamma_max


class Trainer:
    def __init__(self, model, optimizer, scheduler, config, loss_fn=None,
                 wandb_logger=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.writer = SummaryWriter(log_dir=os.path.join(config.workdir, "tb_logs"))
        self.loss_fn = loss_fn or SlotPiLoss(config)
        self.best_loss = float('inf')
        self.burnin = getattr(config, 'burnin_frames', 6)
        self.rollout = getattr(config, 'rollout_frames', 10)
        self.max_grad_norm = getattr(config, 'max_grad_norm', 1.0)
        self.log_every = getattr(config, 'log_every', 10)
        self.save_every = getattr(config, 'save_every', 1000)
        self.keep_last = getattr(config, 'keep_last_checkpoints', 0)
        self.ckpt_dir = os.path.join(config.workdir, "checkpoints")
        self.wandb = wandb_logger or WandBLogger(enabled=False)
        self.num_steps = getattr(config, 'num_steps', 160000)
        self.gamma_max = getattr(config, 'gamma_max', 1.0)
        self.c_gamma = getattr(config, 'c_gamma', 10.0)
        self.lambda_recon_burnin = getattr(config, 'lambda_recon_burnin', 1.0)
        self.lambda_recon_rollout = getattr(config, 'lambda_recon_rollout', 1.0)
        self.rev_grad_max_norm = getattr(config, 'rev_grad_max_norm', 5.0)

    def _set_gamma(self, step):
        '''计算当前步数的 gamma 并设到模型的 GRL 上。'''
        gamma = compute_gamma(step, self.num_steps, self.gamma_max, self.c_gamma)
        if hasattr(self.model, 'grl'):
            self.model.grl.set_gamma(gamma)
        return gamma

    def _compute_loss(self, batch):
        '''计算单 batch 的损失。返回 total_loss = recon + slot_pred_loss。'''
        frames = batch["video"].to(self.device)
        out = self.model(frames)

        # 解码器输出尺寸与目标不匹配时，下采样目标
        dec_size = out["outputs"]["video_burnin"].shape[-1]
        target_size = frames.shape[-1]
        if dec_size != target_size:
            b, t = frames.shape[:2]
            flat = frames[:, :self.burnin].reshape(-1, 3, target_size, target_size)
            target_burnin = nn.functional.interpolate(flat, size=dec_size, mode='bilinear'
            ).reshape(b, self.burnin, 3, dec_size, dec_size)
            flat = frames[:, self.burnin:self.burnin + self.rollout].reshape(-1, 3, target_size, target_size)
            target_rollout = nn.functional.interpolate(flat, size=dec_size, mode='bilinear'
            ).reshape(b, self.rollout, 3, dec_size, dec_size)
        else:
            target_burnin = frames[:, :self.burnin]
            target_rollout = frames[:, self.burnin:self.burnin + self.rollout]

        # 重建损失（各自独立权重，防过拟合）
        recon_burnin = self.lambda_recon_burnin * nn.functional.mse_loss(
            out["outputs"]["video_burnin"], target_burnin)
        recon_rollout = self.lambda_recon_rollout * nn.functional.mse_loss(
            out["outputs"]["video_pred"], target_rollout)
        recon_loss = recon_burnin + recon_rollout

        # slot 预测损失（含 slot_loss + static_loss + rev_loss，由 loss_fn 统一返回）
        all_slots = torch.cat([out["slots"]["corrected"], out["slots"]["predicted"]], dim=1)
        slot_pred_loss, aux = self.loss_fn(
            out["slots"]["predicted"], out["slots"]["target"],
            slots_full_seq=all_slots,
            rev_pred=out.get("rev_pred"),
            C=out.get("C"))

        # 总损失 = 重建 + slot 预测 + 静态正则 + GRL 反向
        total_loss = recon_loss + slot_pred_loss
        aux['recon_burnin'] = recon_burnin.item()
        aux['recon_rollout'] = recon_rollout.item()
        return total_loss, aux

    def train_step(self, batch):
        '''单步训练：前向 → 反向 → 梯度裁剪 → 更新参数。'''
        self.model.train()
        self.optimizer.zero_grad()
        loss, aux = self._compute_loss(batch)
        loss.backward()
        # 对 rev 分支单独做梯度裁剪（防止 L_rev 过大破坏编码器）
        if hasattr(self.model, 'mlp_rev') and self.rev_grad_max_norm > 0:
            nn.utils.clip_grad_norm_(self.model.mlp_rev.parameters(), self.rev_grad_max_norm)
        # 全局梯度裁剪
        if self.max_grad_norm > 0:
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        else:
            grad_norm = 0.0
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item(), aux, grad_norm

    def evaluate(self, dataloader):
        '''验证集评估。'''
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                loss, _ = self._compute_loss(batch)
                total_loss += loss.item()
        return total_loss / max(1, len(dataloader))

    def _cleanup_old_checkpoints(self):
        '''只保留最近的 keep_last 个存档。'''
        if self.keep_last <= 0:
            return
        ckpts = sorted(glob.glob(os.path.join(self.ckpt_dir, "step_*.pt")))
        while len(ckpts) > self.keep_last:
            os.remove(ckpts[0])
            ckpts = ckpts[1:]

    def save_checkpoint(self, path, step, loss):
        '''保存完整存档（model + optimizer + scheduler）。'''
        torch.save({
            "step": step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "loss": loss,
        }, path)

    def load_checkpoint(self, path):
        '''加载存档，返回 (step, loss)。'''
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        return ckpt.get("step", 0), ckpt.get("loss", float("inf"))

    def train(self, train_loader, val_loader, num_steps, start_step=0):
        '''完整训练循环。
           total_loss = recon_burnin + recon_rollout + slot_pred_loss
           slot_pred_loss 由 loss_fn 返回，内含 slot + static + rev 三项（均已加权）。
        '''
        global_step = start_step
        pbar = tqdm(total=num_steps, initial=start_step, desc="Training")

        while global_step < num_steps:
            for batch in train_loader:
                if global_step >= num_steps:
                    break

                gamma = self._set_gamma(global_step)
                loss_val, aux, grad_norm = self.train_step(batch)
                global_step += 1
                pbar.update(1)

                if global_step % self.log_every == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        "loss": f"{loss_val:.4f}",
                        "slot": f"{aux['slot_loss']:.4f}",
                        "static": f"{aux['static_loss']:.6f}",
                        "rev": f"{aux['rev_loss']:.6f}",
                        "lr": f"{lr:.2e}",
                        "gamma": f"{gamma:.3f}",
                    })

                    # TensorBoard
                    self.writer.add_scalar("loss/total", loss_val, global_step)
                    self.writer.add_scalar("loss/recon_burnin", aux['recon_burnin'], global_step)
                    self.writer.add_scalar("loss/recon_rollout", aux['recon_rollout'], global_step)
                    self.writer.add_scalar("loss/slot", aux['slot_loss'], global_step)
                    self.writer.add_scalar("loss/static", aux['static_loss'], global_step)
                    self.writer.add_scalar("loss/rev", aux['rev_loss'], global_step)
                    self.writer.add_scalar("lr", lr, global_step)
                    self.writer.add_scalar("rev_weight", gamma, global_step)
                    self.writer.add_scalar("grad_norm", grad_norm, global_step)

                    # WandB
                    self.wandb.log({
                        "loss/total": loss_val,
                        "loss/recon_burnin": aux['recon_burnin'],
                        "loss/recon_rollout": aux['recon_rollout'],
                        "loss/slot": aux['slot_loss'],
                        "loss/static": aux['static_loss'],
                        "loss/rev": aux['rev_loss'],
                        "train/lr": lr,
                        "train/rev_weight": gamma,
                        "train/grad_norm": grad_norm,
                    }, step=global_step)

                if global_step % self.save_every == 0:
                    self.save_checkpoint(
                        os.path.join(self.ckpt_dir, f"step_{global_step}.pt"),
                        global_step, loss_val)
                    self._cleanup_old_checkpoints()
                    if loss_val < self.best_loss:
                        self.best_loss = loss_val
                        self.save_checkpoint(
                            os.path.join(self.ckpt_dir, "best.pt"),
                            global_step, loss_val)

            val_loss = self.evaluate(val_loader)
            self.writer.add_scalar("loss/eval", val_loss, global_step)
            self.wandb.log({"loss/eval": val_loss}, step=global_step)
            print(f"Eval step {global_step}: val_loss={val_loss:.6f}")

        pbar.close()
        self.writer.close()
        self.wandb.finish()
