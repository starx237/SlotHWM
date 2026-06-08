import os, glob, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .losses import SlotPiLoss

try:
    from PIL import Image, ImageDraw
    _PIL_AVAIL = True
except ImportError:
    _PIL_AVAIL = False


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
        # 精细梯度裁剪（每模块单独 max_norm，0 = 不裁剪）
        self.max_encoder_gnorm = getattr(config, 'max_encoder_gnorm', 0.0)
        self.max_decoder_gnorm = getattr(config, 'max_decoder_gnorm', 0.0)
        self.max_slot_attention_gnorm = getattr(config, 'max_slot_attention_gnorm', 0.0)
        # 冻结 slot 感知模块（训练 rollout 时可选保持 encoder + slot_attention 不动）
        self.freeze_slot = getattr(config, 'freeze_slot', False)
        self.eval_every_epochs = getattr(config, 'eval_every_epochs', 10)
        if self.freeze_slot:
            for name in ['encoder', 'slot_attention', 'decoder']:
                mod = getattr(self.model, name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad_(False)
            print(f"Frozen: encoder + slot_attention (rollout only)")

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

        # 解码器输出在 broadcast_size 低分辨率，上采样到原分辨率计算 loss
        dec_size = out["outputs"]["video_burnin"].shape[-1]
        target_size = frames.shape[-1]
        if dec_size != target_size:
            b = frames.shape[0]
            up = lambda x: nn.functional.interpolate(
                x.reshape(-1, 3, dec_size, dec_size),
                size=target_size, mode='bilinear'
            ).reshape(b, -1, 3, target_size, target_size)
            video_burnin = up(out["outputs"]["video_burnin"])
            video_pred = up(out["outputs"]["video_pred"])
        else:
            video_burnin = out["outputs"]["video_burnin"]
            video_pred = out["outputs"]["video_pred"]

        target_burnin = frames[:, :self.burnin]
        target_rollout = frames[:, self.burnin:self.burnin + self.rollout]

        # 重建损失
        recon_burnin = self.lambda_recon_burnin * nn.functional.mse_loss(
            video_burnin, target_burnin)
        recon_rollout = self.lambda_recon_rollout * nn.functional.mse_loss(
            video_pred, target_rollout)
        recon_loss = recon_burnin + recon_rollout

        # slot 预测损失（含 slot_loss + static_loss + rev_loss，由 loss_fn 统一返回）
        all_slots = torch.cat([out["slots"]["corrected"], out["slots"]["predicted"]], dim=1)
        slot_pred_loss, aux = self.loss_fn(
            out["slots"]["predicted"], out["slots"]["target"],
            slots_full_seq=all_slots,
            rev_pred=out.get("rev_pred"),
            S_c=out.get("S_c"),
            energy=out.get("energy_pairs"))

        # slot 监控指标
        with torch.no_grad():
            # slot 间方差（趋于 0 表示所有 slot 趋同 → 坍缩前兆）
            aux['slot_var_across'] = all_slots.var(dim=2, unbiased=False).mean().item()

        # 总损失 = 重建 + slot 预测 + 静态正则 + GRL 反向
        total_loss = recon_loss + slot_pred_loss
        aux['recon_burnin'] = recon_burnin.item()
        aux['recon_rollout'] = recon_rollout.item()
        return total_loss, aux

    def _compute_grad_norms(self):
        '''计算各子模块梯度范数（需 unscale 后、global clip 前调用）。'''
        info = {}
        for name in ['encoder', 'decoder', 'slot_attention', 'slotpi', 'mlp_rev']:
            mod = getattr(self.model, name, None)
            if mod is None:
                continue
            gn = sum(p.grad.norm().item()**2 for p in mod.parameters() if p.grad is not None)
            info[f'grad/{name}'] = gn**0.5 if gn > 0 else 0.0
        return info

    def train_step(self, batch):
        '''单步训练：前向 → 反向 → 梯度裁剪 → 更新参数（全 fp32）。'''
        self.model.train()
        self.optimizer.zero_grad()
        loss, aux = self._compute_loss(batch)
        loss.backward()

        # 安全性检查：检测 inf/nan 梯度，跳过本步
        skip_step = False
        for p in self.model.parameters():
            if p.grad is not None and (torch.isinf(p.grad).any() or torch.isnan(p.grad).any()):
                skip_step = True
                break
        if skip_step:
            self.optimizer.zero_grad()
            return loss.item(), aux, float('inf')

        # rev 分支单独裁剪
        if hasattr(self.model, 'mlp_rev') and self.rev_grad_max_norm > 0:
            nn.utils.clip_grad_norm_(self.model.mlp_rev.parameters(), self.rev_grad_max_norm)
        # 精细裁剪：按模块独立限制更新幅度（0 = 不裁剪）
        module_clip = [
            ('encoder', self.max_encoder_gnorm),
            ('decoder', self.max_decoder_gnorm),
            ('slot_attention', self.max_slot_attention_gnorm),
        ]
        for name, max_norm in module_clip:
            if max_norm > 0:
                mod = getattr(self.model, name, None)
                if mod is not None:
                    nn.utils.clip_grad_norm_(mod.parameters(), max_norm)
        # 全局裁剪
        if self.max_grad_norm > 0:
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        else:
            grad_norm = 0.0
        # 子模块梯度监控（在全局 clip 之后，确保 ‖g‖² ≤ max_grad_norm²）
        aux.update(self._compute_grad_norms())
        self.optimizer.step()
        # JEPA target network EMA update
        jepa_alpha = getattr(self.config, 'jepa_alpha', 0.0)
        if jepa_alpha > 0 and hasattr(self.model, 'update_target_network'):
            self.model.update_target_network(jepa_alpha)
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item(), aux, grad_norm

    def evaluate(self, dataloader, step=None, num_viz=5):
        '''验证集评估 + 保存 5 张对比图到 eval_images/step_{step}/。'''
        self.model.eval()
        total_loss = 0
        viz_samples = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                loss, _ = self._compute_loss(batch)
                total_loss += loss.item()

                if _PIL_AVAIL and step is not None and len(viz_samples) < num_viz:
                    frames = batch["video"].to(self.device)
                    with torch.enable_grad():
                        out = self.model(frames)
                    viz_samples.append((out, frames[0:1]))

        if _PIL_AVAIL and step is not None and viz_samples:
            self._save_viz_batch(viz_samples, step)

        return total_loss / max(1, len(dataloader))

    def _save_viz_batch(self, samples, step):
        '''保存 num_viz 张对比图，每张 4 行 × N 列。'''
        viz_dir = os.path.join(self.config.workdir, 'eval_images')
        for old in glob.glob(os.path.join(viz_dir, 'step_*')):
            import shutil
            shutil.rmtree(old, ignore_errors=True)

        step_dir = os.path.join(viz_dir, f'step_{step}')
        os.makedirs(step_dir, exist_ok=True)

        for idx, (out, frames) in enumerate(samples):
            B, T, C, H, W = frames.shape
            burnin, rollout = self.burnin, self.rollout
            target_size = W

            def upscale(dec_tensor):
                ds = dec_tensor.shape[-1]
                flat = dec_tensor.reshape(-1, 3, ds, ds)
                up = F.interpolate(flat, size=target_size, mode='bilinear')
                return up.reshape(dec_tensor.shape[0], dec_tensor.shape[1], 3, target_size, target_size)

            dec_b = upscale(out["outputs"]["video_burnin"][:1])
            dec_p = upscale(out["outputs"]["video_pred"][:1])
            gt_b = frames[:, :burnin]
            gt_r = frames[:, burnin:burnin + rollout]

            n_cols = max(burnin, rollout)
            S = min(80, 2000 // max(n_cols, 1))
            canvas = Image.new('RGB', (n_cols * S, 4 * S), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)

            def put(t, row, col):
                arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                im = Image.fromarray((arr * 255).astype('uint8'))
                im = im.resize((S, S), Image.BILINEAR)
                canvas.paste(im, (col * S, row * S))

            for t in range(burnin):
                put(gt_b[0, t], 0, t)
                put(dec_b[0, t], 1, t)
            for t in range(rollout):
                put(gt_r[0, t], 2, t)
                put(dec_p[0, t], 3, t)

            for i, label in enumerate(['GT Burnin', 'Recon Burnin', 'GT Rollout', 'Pred Rollout']):
                draw.text((2, i * S + 2), label, fill=(0, 0, 0))

            path = os.path.join(step_dir, f'sample_{idx}.png')
            canvas.save(path)

            if self.wandb.enabled:
                self.wandb.log({f"eval/recon_{idx}": self.wandb.wandb.Image(path)}, step=step)

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
        epoch = 0
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
                        "energy": f"{aux['energy_loss']:.6f}",
                        "lr": f"{lr:.2e}",
                        "gamma": f"{gamma:.3f}",
                    })

                    # TensorBoard
                    self.writer.add_scalar("loss/total", loss_val*10, global_step)
                    self.writer.add_scalar("loss/recon_burnin", aux['recon_burnin']*10, global_step)
                    self.writer.add_scalar("loss/recon_rollout", aux['recon_rollout']*10, global_step)
                    self.writer.add_scalar("loss/slot", aux['slot_loss']*10, global_step)
                    self.writer.add_scalar("loss/static", aux['static_loss']*10, global_step)
                    self.writer.add_scalar("loss/rev", aux['rev_loss']*10, global_step)
                    self.writer.add_scalar("loss/energy", aux['energy_loss']*10, global_step)
                    self.writer.add_scalar("lr", lr, global_step)
                    self.writer.add_scalar("rev_weight", gamma, global_step)
                    self.writer.add_scalar("grad_norm", grad_norm, global_step)

                    # 监控指标日志（梯度 + slot 统计）
                    monitor_keys = ['grad/encoder', 'grad/decoder',
                                    'grad/slot_attention', 'grad/slotpi', 'grad/mlp_rev',
                                    'slot_var_across']
                    for k in monitor_keys:
                        if k in aux:
                            self.writer.add_scalar(k, aux[k], global_step)

                    # WandB
                    log_dict = {
                        "loss/recon_burnin": aux['recon_burnin']*10,
                        "loss/recon_rollout": aux['recon_rollout']*10,
                        "loss/slot": aux['slot_loss']*10,
                        "loss/static": aux['static_loss']*10,
                        "loss/rev": aux['rev_loss']*10,
                        "loss/energy": aux['energy_loss']*10,
                        "loss/total": loss_val*10,
                        "train/lr": lr,
                        "train/rev_weight": gamma,
                        "train/grad_norm": grad_norm,
                    }
                    for k in monitor_keys:
                        if k in aux:
                            log_dict[k] = aux[k]
                    self.wandb.log(log_dict, step=global_step)

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

            epoch += 1
            if epoch % self.eval_every_epochs == 0:
                val_loss = self.evaluate(val_loader, step=global_step)
                self.writer.add_scalar("loss/eval", val_loss*10, global_step)
                self.wandb.log({"loss/eval": val_loss*10}, step=global_step)
                print(f"Eval step {global_step}: val_loss={val_loss:.6f}")
        pbar.close()
        self.writer.close()
        self.wandb.finish()
