import os, glob, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .losses import SlotPiLoss, get_loss_ratios, get_rollout_frames

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
    '''计算 GRL 的 gamma 系数，按 sigmoid 曲线从 0 渐升至 gamma_max'''
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
        self.lambda_pos = getattr(config, 'lambda_pos', 0.0)
        self.lambda_cos = getattr(config, 'lambda_cos', 0.0)
        self.pretrain = getattr(config, 'pretrain', False)
        self.freeze_slot = getattr(config, 'freeze_slot', False)
        self.jepa = getattr(config, 'jepa', False)
        self.jepa_alpha = getattr(config, 'jepa_alpha', 0.996)
        self.eval_every_epochs = getattr(config, 'eval_every_epochs', 10)
        self.post_save_callback = getattr(config, 'post_save_callback', None)
        if self.pretrain:
            self.rollout = 0
            print(f"Pretrain mode: burnin={self.burnin}, no rollout")

    def _set_gamma(self, step):
        '''计算当前步数的 gamma 并设到模型的 GRL 上。'''
        gamma = compute_gamma(step, self.num_steps, self.gamma_max, self.c_gamma)
        if hasattr(self.model, 'grl'):
            self.model.grl.set_gamma(gamma)
        return gamma

    def _ema_update(self):
        '''EMA 更新 target encoder + target slot_attention（JEPA 模式）。'''
        if not self.jepa:
            return
        alpha = self.jepa_alpha
        for online_name, target_name in [('encoder', 'target_encoder'), ('slot_attention', 'target_slot_attention')]:
            online = getattr(self.model, online_name, None)
            target = getattr(self.model, target_name, None)
            if online is not None and target is not None:
                for p_online, p_target in zip(online.parameters(), target.parameters()):
                    p_target.data.mul_(alpha).add_(p_online.data, alpha=1 - alpha)

    def _compute_loss(self, batch, step=0):
        '''计算单 batch 的损失。支持 pretrain、finetune、JEPA 三种模式。
        aux 中所有 loss 值均为原始系数（不受 ratios 影响），用于监控。
        返回的 total_loss 含 ratio 缩放，用于 backward。'''
        frames = batch["video"].to(self.device)
        out = self.model(frames)

        dec_size = out["outputs"]["video_burnin"].shape[-1]
        target_size = frames.shape[-1]
        if dec_size != target_size:
            b = frames.shape[0]
            up = lambda x: nn.functional.interpolate(
                x.reshape(-1, 3, dec_size, dec_size),
                size=target_size, mode='bilinear'
            ).reshape(b, -1, 3, target_size, target_size)
            video_burnin = up(out["outputs"]["video_burnin"])
            video_pred = up(out["outputs"]["video_pred"]) if out["outputs"]["video_pred"] is not None else None
        else:
            video_burnin = out["outputs"]["video_burnin"]
            video_pred = out["outputs"]["video_pred"]

        ratios = get_loss_ratios(step)
        target_burnin = frames[:, :self.burnin]
        target_rollout = frames[:, self.burnin:self.burnin + self.rollout] if self.rollout > 0 else None

        # 重建损失（原始系数用于 aux，ratio 缩放用于梯度）
        recon_burnin_val = self.lambda_recon_burnin * nn.functional.mse_loss(
            video_burnin, target_burnin)
        recon_burnin_grad = recon_burnin_val * ratios["burnin"]

        if self.pretrain or self.rollout == 0:
            total_grad = recon_burnin_grad
            extra_loss = 0.0
            aux = {"slot_loss": 0.0, "static_loss": 0.0, "rev_loss": 0.0, "energy_loss": 0.0, "loss_q": 0.0, "loss_p": 0.0}

            if self.lambda_pos > 0 or self.lambda_cos > 0:
                slots = out["slots"]["corrected"][:, 0]
                B, N, D = slots.shape

                if self.lambda_cos > 0:
                    attn = out["attn"]
                    attn_dot = torch.bmm(attn, attn.transpose(1, 2))
                    diag = torch.eye(N, device=slots.device)
                    off_diag = (attn_dot * (1 - diag.unsqueeze(0))).sum(dim=[-2, -1])
                    loss_cos = off_diag.mean() / (N * (N - 1))
                    extra_loss = extra_loss + self.lambda_cos * loss_cos
                    aux['loss_cos'] = (self.lambda_cos * loss_cos).item()

                if self.lambda_pos > 0:
                    slots_detached = out["slots"]["corrected"][:, 0].detach()
                    _, alpha, _ = self.model.decoder(slots_detached, return_rgb=True)
                    Sp = slots_detached[:, :, -3:-1]
                    H, W = alpha.shape[-2:]
                    gy, gx = torch.meshgrid(
                        torch.linspace(-1, 1, H, device=slots_detached.device),
                        torch.linspace(-1, 1, W, device=slots_detached.device),
                        indexing='ij',
                    )
                    a = alpha.squeeze(2)
                    denom = a.sum(dim=[-2, -1]) + 1e-8
                    cx = (a * gx).sum(dim=[-2, -1]) / denom
                    cy = (a * gy).sum(dim=[-2, -1]) / denom
                    centroid = torch.stack([cx, cy], dim=-1)
                    loss_pos = F.mse_loss(centroid, Sp)
                    extra_loss = extra_loss + self.lambda_pos * loss_pos
                    aux['loss_pos'] = (self.lambda_pos * loss_pos).item()

            total_grad = recon_burnin_grad + extra_loss
            aux['recon_burnin'] = recon_burnin_val.item()
            aux['recon_rollout'] = 0.0
            with torch.no_grad():
                aux['slot_var_across'] = out["slots"]["corrected"].var(dim=2, unbiased=False).mean().item()
            aux['total'] = recon_burnin_val.item() + (extra_loss.item() if isinstance(extra_loss, torch.Tensor) else extra_loss)
            aux['total_scaled'] = total_grad.item()
            aux['rollout_actual'] = 0
            return total_grad, aux

        # Finetune / JEPA: 完整损失
        current_rollout = get_rollout_frames(step, self.rollout)
        r = min(current_rollout, self.rollout)
        recon_rollout_val = self.lambda_recon_rollout * nn.functional.mse_loss(
            video_pred[:, :r], target_rollout[:, :r])
        recon_rollout_grad = recon_rollout_val * ratios["rollout"]

        all_slots = torch.cat([out["slots"]["corrected"], out["slots"]["predicted"]], dim=1)
        slot_pred_grad, aux = self.loss_fn(
            out["slots"]["predicted"][:, :r], out["slots"]["target"][:, :r],
            slots_full_seq=all_slots,
            rev_pred=out.get("rev_pred"),
            S_c=out.get("S_c"),
            energy=out.get("energy_pairs"),
            ratios=ratios)

        with torch.no_grad():
            aux['slot_var_across'] = all_slots.var(dim=2, unbiased=False).mean().item()

        total_val = (recon_burnin_val + recon_rollout_val +
                     aux.get('slot_loss', 0.0) +
                     aux.get('energy_loss', 0.0) +
                     aux.get('static_loss', 0.0) +
                     aux.get('rev_loss', 0.0))
        total_grad = recon_burnin_grad + recon_rollout_grad + slot_pred_grad
        aux['recon_burnin'] = recon_burnin_val.item()
        aux['recon_rollout'] = recon_rollout_val.item()
        aux['recon_burnin_scaled'] = recon_burnin_grad.item()
        aux['recon_rollout_scaled'] = recon_rollout_grad.item()
        aux['total'] = total_val.item()
        aux['total_scaled'] = total_grad.item()
        aux['rollout_actual'] = r

        with torch.no_grad():
            qp = out.get("qp_metrics")
            if qp is not None:
                aux['loss_q'] = qp.get("loss_q", 0.0)
                aux['loss_p'] = qp.get("loss_p", 0.0)

        return total_grad, aux

    def _compute_grad_norms(self):
        '''计算各子模块梯度范数（需 unscale 后、global clip 前调用）。'''
        info = {}
        for name in ['encoder', 'decoder', 'slot_attention', 'predictor', 'f_z', 'mlp_rev']:
            mod = getattr(self.model, name, None)
            if mod is None:
                continue
            gn = sum(p.grad.norm().item()**2 for p in mod.parameters() if p.grad is not None)
            info[f'grad/{name}'] = gn**0.5 if gn > 0 else 0.0
        return info

    def train_step(self, batch, step=0):
        '''单步训练：前向 → 反向 → 梯度裁剪 → 更新参数（全 fp32）。'''
        self.model.train()
        self.optimizer.zero_grad()
        loss, aux = self._compute_loss(batch, step=step)
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
        self._ema_update()
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
                _, aux = self._compute_loss(batch, step=step or 0)
                total_loss += aux.get('total', 0.0)

                if _PIL_AVAIL and step is not None and len(viz_samples) < num_viz:
                    frames = batch["video"].to(self.device)
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
            gt_b = frames[:, :burnin]

            n_cols = max(burnin, rollout) if rollout > 0 else burnin
            n_rows = 4 if rollout > 0 and out["outputs"]["video_pred"] is not None else 2
            S = min(80, 2000 // max(n_cols, 1))
            canvas = Image.new('RGB', (n_cols * S, n_rows * S), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)

            def put(t, row, col):
                arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                im = Image.fromarray((arr * 255).astype('uint8'))
                im = im.resize((S, S), Image.BILINEAR)
                canvas.paste(im, (col * S, row * S))

            for t in range(burnin):
                put(gt_b[0, t], 0, t)
                put(dec_b[0, t], 1, t)

            if rollout > 0 and out["outputs"]["video_pred"] is not None:
                dec_p = upscale(out["outputs"]["video_pred"][:1])
                gt_r = frames[:, burnin:burnin + rollout]
                for t in range(rollout):
                    put(gt_r[0, t], 2, t)
                    put(dec_p[0, t], 3, t)
                labels = ['GT Burnin', 'Recon Burnin', 'GT Rollout', 'Pred Rollout']
            else:
                labels = ['GT Burnin', 'Recon Burnin']

            for i, label in enumerate(labels):
                draw.text((2, i * S + 2), label, fill=(0, 0, 0))

            path = os.path.join(step_dir, f'sample_{idx}.png')
            canvas.save(path)

            if self.wandb.enabled:
                self.wandb.log({f"eval/recon_{idx}": self.wandb.wandb.Image(path)}, step=step)

    def _cleanup_old_checkpoints(self, current_step):
        '''只保留当前步数往前最近的 keep_last 个存档，删除落后/未来的存档。'''
        if self.keep_last <= 0:
            return
        ckpts = sorted(glob.glob(os.path.join(self.ckpt_dir, "step_*.pt")),
                       key=lambda p: int(os.path.basename(p).replace('step_', '').replace('.pt', '')))
        keep_threshold = current_step - self.keep_last * self.save_every
        for fp in ckpts:
            step = int(os.path.basename(fp).replace('step_', '').replace('.pt', ''))
            if step <= keep_threshold:
                os.remove(fp)

    @staticmethod
    def _match_and_load(model_state, ckpt_state):
        '''鲁棒 key 匹配加载：自动处理 _orig_mod 前缀差异 + 形状不匹配 + 重命名兼容。'''
        # 历史重命名映射：旧存档 key 前缀 → 新模型 key 前缀
        rename_map = {
            'slotpi.': 'predictor.',
            'SlotPiModel.': 'predictor.',
        }
        def translate(k):
            for old, new in rename_map.items():
                if k.startswith(old):
                    return new + k[len(old):]
            return k

        def match(ck, mk):
            return (ck == mk or
                    ck.replace('_orig_mod.', '') == mk or
                    ck == mk.replace('_orig_mod.', ''))
        loaded = {}
        ckpt_clean = {}
        for k, v in ckpt_state.items():
            ckpt_clean[k.replace('_orig_mod.', '')] = v
        # 如果直接匹配结果差（<20%），尝试用映射后的 key 重试
        for ck_key, v in ckpt_state.items():
            matched_mk = None
            for mk_key in model_state:
                if match(ck_key, mk_key) and v.shape == model_state[mk_key].shape:
                    matched_mk = mk_key
                    break
            if matched_mk is None:
                # 尝试映射后重匹配
                mapped = translate(ck_key)
                if mapped != ck_key:
                    for mk_key in model_state:
                        if match(mapped, mk_key) and v.shape == model_state[mk_key].shape:
                            matched_mk = mk_key
                            break
            if matched_mk is not None:
                loaded[matched_mk] = v
        return loaded

    def save_checkpoint(self, path, step, loss):
        '''保存完整存档，自动去除 _orig_mod 前缀。'''
        raw = self.model.state_dict()
        clean = {k.replace('_orig_mod.', ''): v for k, v in raw.items()}
        torch.save({
            "step": step,
            "model": clean,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "loss": loss,
        }, path)

    def load_checkpoint(self, path):
        '''加载存档。自动跳过形状不匹配的 key，兼容架构变更后的续训。'''
        ckpt = torch.load(path, map_location=self.device)
        model_state = self.model.state_dict()
        ckpt_state = ckpt["model"]

        print("  CKPT keys (first 5):", list(ckpt_state.keys())[:5])
        print("  Model keys (first 5):", list(model_state.keys())[:5])

        filtered = self._match_and_load(model_state, ckpt_state)
        skipped = len(ckpt_state) - len(filtered)
        key_stats = {"encoder": 0, "slot_attention": 0, "decoder": 0, "other": 0}
        for k in filtered:
            for prefix in key_stats:
                if k.startswith(prefix):
                    key_stats[prefix] += 1
                    break
            else:
                key_stats["other"] += 1

        self.model.load_state_dict(filtered, strict=False)
        if skipped or key_stats.get("other", 0) > 0:
            print(f"Checkpoint load: {len(filtered)}/{len(ckpt_state)} keys loaded, {skipped} skipped")
            print(f"  Loaded breakdown: encoder={key_stats['encoder']} slot_attn={key_stats['slot_attention']} decoder={key_stats['decoder']} other={key_stats['other']}")
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, RuntimeError) as e:
            print(f"Optimizer state incompatible (architecture change), reinitializing: {e}")
        if self.scheduler and ckpt.get("scheduler"):
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except (ValueError, RuntimeError) as e:
                print(f"Scheduler state incompatible (architecture change), reinitializing: {e}")
        return ckpt.get("step", 0), ckpt.get("loss", float("inf"))

    def load_pretrained(self, path):
        '''仅加载 STATM-SAVi 预训练权重（encoder, slot_attention, decoder）。
        忽略预测模块（f_z, predictor, mlp_rev 等）的缺失/不匹配。'''
        ckpt = torch.load(path, map_location=self.device)
        model_state = self.model.state_dict()
        ckpt_state = ckpt["model"]

        loaded = self._match_and_load(model_state, ckpt_state)
        skipped = {k: v.shape for k, v in ckpt_state.items()
                   if not any(k.replace('_orig_mod.', '') == mk.replace('_orig_mod.', '')
                              and v.shape == model_state[mk].shape for mk in model_state)}

        self.model.load_state_dict(loaded, strict=False)

        missing = set(model_state.keys()) - set(loaded.keys())
        if missing:
            print(f"Pretrain load — missing (initialized randomly): {len(missing)} keys")
        if skipped:
            print(f"Pretrain load — skipped (shape mismatch or not in model): {len(skipped)} keys")
        return loaded, skipped

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
                loss_val, aux, grad_norm = self.train_step(batch, step=global_step)
                global_step += 1
                pbar.update(1)

                if global_step % self.log_every == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        "loss": f"{aux.get('total', loss_val):.4f}",
                        "recon_b": f"{aux['recon_burnin']:.4f}",
                        "recon_r": f"{aux['recon_rollout']:.4f}",
                        "slot": f"{aux['slot_loss']:.4f}",
                        "static": f"{aux['static_loss']:.6f}",
                        "rev": f"{aux['rev_loss']:.6f}",
                        "energy": f"{aux['energy_loss']:.6f}",
                        "lr": f"{lr:.2e}",
                        "gamma": f"{gamma:.3f}",
                    })

                    # TensorBoard
                    self.writer.add_scalar("loss/total", aux.get('total', loss_val)*10, global_step)
                    self.writer.add_scalar("loss/recon_burnin", aux['recon_burnin']*10, global_step)
                    self.writer.add_scalar("loss/recon_rollout", aux['recon_rollout']*10, global_step)
                    self.writer.add_scalar("loss/slot", aux['slot_loss']*10, global_step)
                    self.writer.add_scalar("loss/static", aux['static_loss']*10, global_step)
                    self.writer.add_scalar("loss/rev", aux['rev_loss']*10, global_step)
                    self.writer.add_scalar("loss/energy", aux['energy_loss']*10, global_step)
                    if 'loss_pos' in aux:
                        self.writer.add_scalar("loss/pos", aux['loss_pos']*10, global_step)
                    if 'loss_cos' in aux:
                        self.writer.add_scalar("loss/cos", aux['loss_cos']*10, global_step)
                    self.writer.add_scalar("lr", lr, global_step)
                    self.writer.add_scalar("rev_weight", gamma, global_step)
                    self.writer.add_scalar("grad_norm", grad_norm, global_step)
                    self.writer.add_scalar("rollout/actual", aux.get('rollout_actual', self.rollout), global_step)

                    # 监控指标日志（梯度 + slot 统计）
                    monitor_keys = ['grad/encoder', 'grad/decoder',
                                    'grad/slot_attention', 'grad/predictor', 'grad/mlp_rev',
                                    'slot_var_across',
                                    'loss_q', 'loss_p']
                    for k in monitor_keys:
                        if k in aux:
                            self.writer.add_scalar(k, aux[k], global_step)

                    # WandB
                    log_dict = {
                        "loss/total": aux.get('total', loss_val)*10,
                        "loss/recon_burnin": aux['recon_burnin']*10,
                        "loss/recon_rollout": aux['recon_rollout']*10,
                        "loss/slot": aux['slot_loss']*10,
                        "loss/static": aux['static_loss']*10,
                        "loss/rev": aux['rev_loss']*10,
                        "loss/energy": aux['energy_loss']*10,
                        "train/lr": lr,
                        "train/rev_weight": gamma,
                        "train/grad_norm": grad_norm,
                        "train/rollout_actual": aux.get('rollout_actual', self.rollout),
                    }
                    for k in ['loss_pos', 'loss_cos']:
                        if k in aux:
                            log_dict[f"loss/{k}"] = aux[k]*10
                    for k in monitor_keys:
                        if k in aux:
                            log_dict[k] = aux[k]
                    self.wandb.log(log_dict, step=global_step)

                if global_step % self.save_every == 0:
                    self.save_checkpoint(
                        os.path.join(self.ckpt_dir, f"step_{global_step}.pt"),
                        global_step, loss_val)
                    self._cleanup_old_checkpoints(global_step)
                    if self.post_save_callback:
                        self.post_save_callback(global_step)

                best_candidate = aux.get('total', loss_val)
                if best_candidate < self.best_loss:
                    self.best_loss = best_candidate
                    self.save_checkpoint(
                        os.path.join(self.ckpt_dir, "best.pt"),
                        global_step, best_candidate)

            epoch += 1
            if epoch % self.eval_every_epochs == 0:
                val_loss = self.evaluate(val_loader, step=global_step)
                self.writer.add_scalar("loss/eval", val_loss*10, global_step)
                self.wandb.log({"loss/eval": val_loss*10}, step=global_step)
                print(f"Eval step {global_step}: val_loss={val_loss:.6f}")
        pbar.close()
        self.writer.close()
        self.wandb.finish()
