# train2/trainer.py —— 第二阶段训练器
# 负责阶段二的 slot 时序预测训练：包含 burnin 预处理和 rollout 滚动预测

import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from train2.losses import SlotPiLoss


class Stage2Trainer:
    # 第二阶段训练器：基于历史 slot 预测未来 slot 序列
    def __init__(self, model, optimizer, scheduler, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        # 自动选择 GPU（CUDA）或 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # TensorBoard 日志记录器
        self.writer = SummaryWriter(log_dir=os.path.join(config.workdir, "tb_logs"))
        self.loss_fn = SlotPiLoss(config)   # 损失函数
        self.best_loss = float('inf')       # 记录最佳验证损失

    def train_epoch(self, dataloader, epoch):
        # 训练一个 epoch
        # 流程：burnin（用历史帧初始化 buffer）→ rollout（循环预测）→ 计算损失 → 反向传播
        self.model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch in dataloader:
            slots_in = batch["slots"].to(self.device)
            B, T, N, D = slots_in.shape
            burnin = self.config.burnin_frames     # 历史帧数（用于初始化）
            rollout = self.config.rollout_frames   # 预测帧数

            # 切分数据：burnin 部分和 ground truth 部分
            slots_burnin = slots_in[:, :burnin]
            slots_gt = slots_in[:, burnin:burnin + rollout]

            # 初始化滑动缓冲区，用 burnin 帧填充
            buffer = torch.zeros(B, self.config.buffer_len, N, D, device=self.device)
            for t in range(burnin):
                buffer = torch.cat([buffer[:, 1:], slots_burnin[:, t].unsqueeze(1)], dim=1)

            # 计算静态上下文 C（由 burnin 帧的静态部分聚合得到）
            C = None
            if hasattr(self.model, 'compute_C') and getattr(self.config, "use_static_context", True):
                C = self.model.compute_C(slots_burnin)

            self.optimizer.zero_grad()

            pred_slots_list = []
            energy_list = []

            # 滚动预测 rollout 步
            slots_t = slots_burnin[:, -1]
            for _ in range(rollout):
                if getattr(self.config, "return_energy", False):
                    next_slots, energy = self.model(slots_t, buffer, C=C, return_energy=True)
                    energy_list.append(energy)
                else:
                    next_slots = self.model(slots_t, buffer, C=C)
                pred_slots_list.append(next_slots)
                # 更新缓冲区：移除最旧帧，加入最新预测帧
                buffer = torch.cat([buffer[:, 1:], next_slots.unsqueeze(1)], dim=1)
                slots_t = next_slots

            pred_slots = torch.stack(pred_slots_list, dim=1)
            # 构造完整序列（burnin + rollout）用于 L_LC 静态特征方差正则化（idea.md §3）
            slots_full_seq = torch.cat([slots_burnin, pred_slots], dim=1)

            energy_tensor = None
            if energy_list:
                energy_tensor = torch.stack(energy_list)

            loss, aux = self.loss_fn(pred_slots, slots_gt, energy=energy_tensor,
                                     slots_full_seq=slots_full_seq)
            loss.backward()
            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), **{k: f"{v:.6f}" for k, v in aux.items()}})

        avg_loss = total_loss / len(dataloader)
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        if self.scheduler is not None:
            self.scheduler.step()
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)
        return avg_loss

    def evaluate(self, dataloader, epoch):
        # 在验证集上评估：与训练流程相同但不计算梯度
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                slots_in = batch["slots"].to(self.device)
                B, T, N, D = slots_in.shape
                burnin = self.config.burnin_frames
                rollout = self.config.rollout_frames
                slots_burnin = slots_in[:, :burnin]
                slots_gt = slots_in[:, burnin:burnin + rollout]
                buffer = torch.zeros(B, self.config.buffer_len, N, D, device=self.device)
                for t in range(burnin):
                    buffer = torch.cat([buffer[:, 1:], slots_burnin[:, t].unsqueeze(1)], dim=1)
                # 计算静态上下文 C
                C = None
                if hasattr(self.model, 'compute_C') and getattr(self.config, "use_static_context", True):
                    C = self.model.compute_C(slots_burnin)
                pred_slots_list = []
                slots_t = slots_burnin[:, -1]
                for _ in range(rollout):
                    next_slots = self.model(slots_t, buffer, C=C)
                    pred_slots_list.append(next_slots)
                    buffer = torch.cat([buffer[:, 1:], next_slots.unsqueeze(1)], dim=1)
                    slots_t = next_slots
                pred_slots = torch.stack(pred_slots_list, dim=1)
                # 构造完整序列用于 L_LC 静态特征方差正则化
                slots_full_seq = torch.cat([slots_burnin, pred_slots], dim=1)
                loss, _ = self.loss_fn(pred_slots, slots_gt,
                                       slots_full_seq=slots_full_seq)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.writer.add_scalar("Loss/eval", avg_loss, epoch)
        return avg_loss

    def save_checkpoint(self, path, epoch, loss):
        # 保存模型检查点
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "loss": loss,
        }, path)

    def load_checkpoint(self, path):
        # 加载检查点，恢复模型、优化器和调度器状态
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"], checkpoint["loss"]

    def train(self, train_loader, val_loader, num_epochs):
        # 完整训练流程：逐 epoch 训练 → 验证 → 保存最佳模型 → 定期保存检查点
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.evaluate(val_loader, epoch)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(
                    os.path.join(self.config.workdir, "checkpoints", "best.pth"),
                    epoch, val_loss)

            if epoch % self.config.get("save_every", 10) == 0:
                self.save_checkpoint(
                    os.path.join(self.config.workdir, "checkpoints", f"epoch_{epoch}.pth"),
                    epoch, val_loss)

            print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")