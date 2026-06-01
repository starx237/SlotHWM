# train1/trainer.py —— 第一阶段训练器
# 负责阶段一的训练流程：前向传播、反向传播、验证、检查点管理

import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np


class Stage1Trainer:
    # 第一阶段训练器：视频重建训练
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
        self.best_loss = float('inf')   # 记录最佳验证损失

    def train_epoch(self, dataloader, epoch):
        # 训练一个 epoch：前向传播 → 计算损失 → 反向传播 → 梯度裁剪 → 参数更新
        self.model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch in pbar:
            frames = batch["video"].to(self.device)
            self.optimizer.zero_grad()

            out = self.model(frames)
            pred_frames = out["outputs"]["video"]
            loss = nn.functional.mse_loss(pred_frames, frames)

            loss.backward()
            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        if self.scheduler is not None:
            self.scheduler.step()
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

        return avg_loss

    def evaluate(self, dataloader, epoch):
        # 在验证集上评估：不计算梯度，仅前向传播计算损失
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                frames = batch["video"].to(self.device)
                out = self.model(frames)
                pred_frames = out["outputs"]["video"]
                loss = nn.functional.mse_loss(pred_frames, frames)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.writer.add_scalar("Loss/eval", avg_loss, epoch)
        return avg_loss

    def save_checkpoint(self, path, epoch, loss):
        # 保存模型检查点：包含 epoch、模型参数、优化器状态、调度器状态和损失值
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "loss": loss,
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        # 加载检查点，恢复模型、优化器和调度器状态
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Checkpoint loaded from {path}")
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