# evaluation/evaluator.py —— 模型评估器
# 在验证集上运行完整评估流程：计算 slot 预测损失和 ARI 分割指标

import torch
import numpy as np
from tqdm import tqdm
from evaluation.metrics import Ari, AriNoBg


class Evaluator:
    # 评估器：对模型进行全面的 slot 预测评估
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self, model, dataloader):
        # 执行评估：burnin → rollout → 计算损失和 ARI 指标
        model.eval()
        ari_metric = Ari()               # 完整 ARI 指标
        ari_nobg_metric = AriNoBg()      # 无背景 ARI 指标
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                slots_in = batch["slots"].to(self.device)
                B, T, N, D = slots_in.shape
                burnin = self.config.burnin_frames
                rollout = self.config.rollout_frames

                # 切分 burnin 和 ground truth
                slots_burnin = slots_in[:, :burnin]
                slots_gt = slots_in[:, burnin:burnin + rollout]

                # 初始化滑动缓冲区
                buffer = torch.zeros(B, self.config.buffer_len, N, D, device=self.device)
                for t in range(burnin):
                    buffer = torch.cat([buffer[:, 1:], slots_burnin[:, t].unsqueeze(1)], dim=1)

                # 滚动预测 rollout 步
                pred_slots_list = []
                slots_t = slots_burnin[:, -1]
                for _ in range(rollout):
                    next_slots = model(slots_t, buffer)
                    pred_slots_list.append(next_slots)
                    buffer = torch.cat([buffer[:, 1:], next_slots.unsqueeze(1)], dim=1)
                    slots_t = next_slots

                pred_slots = torch.stack(pred_slots_list, dim=1)
                slot_loss = torch.nn.functional.mse_loss(pred_slots, slots_gt)
                total_loss += slot_loss.item()

        avg_loss = total_loss / len(dataloader)
        ari = ari_metric.result()
        ari_nobg = ari_nobg_metric.result()

        results = {
            "loss": avg_loss,
            "ari": ari,
            "ari_nobg": ari_nobg,
        }
        return results