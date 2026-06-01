# evaluation/metrics.py —— 评估指标定义
# 提供调整兰德指数（ARI）及其无背景变体，用于分割质量评估

import torch
import numpy as np


def adjusted_rand_index(true_ids, pred_ids, num_instances_true, num_instances_pred,
                        padding_mask=None, ignore_background=False):
    # 计算调整兰德指数（ARI）：衡量预测分割与真实分割的一致性
    # true_ids/pred_ids: 真实和预测的分割标签  [B, T, H, W]
    # num_instances_true/pred: 真实和预测的实例数量
    # padding_mask: 填充掩码，忽略无效区域
    # ignore_background: 是否忽略背景类（索引 0）
    true_oh = torch.nn.functional.one_hot(true_ids, num_instances_true).float()
    pred_oh = torch.nn.functional.one_hot(pred_ids, num_instances_pred).float()
    if padding_mask is not None:
        true_oh = true_oh * padding_mask.unsqueeze(-1).float()
    if ignore_background:
        true_oh = true_oh[..., 1:]
    # 计算混淆矩阵 N（真值类别 × 预测类别）
    N = torch.einsum("bthwc,bthwk->bck", true_oh, pred_oh)
    A = N.sum(dim=-1)      # 每行求和（真值分布）
    B = N.sum(dim=-2)      # 每列求和（预测分布）
    num_points = A.sum(dim=1)
    # ARI 公式中的各项指标
    rindex = (N * (N - 1)).sum(dim=[1, 2])
    aindex = (A * (A - 1)).sum(dim=1)
    bindex = (B * (B - 1)).sum(dim=1)
    expected_rindex = aindex * bindex / torch.clamp(num_points * (num_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator
    # 当分母为零时返回 1（完全一致）
    return torch.where(denominator != 0, ari, torch.ones_like(ari))


class Ari:
    # ARI 指标累加器：逐步累加每个 batch 的 ARI 值，最后计算平均
    def __init__(self):
        self.total = 0.0    # ARI 值累加和
        self.count = 0      # 样本计数器

    def update(self, pred_seg, gt_seg, padding_mask, num_instances_pred, num_instances_true):
        # 用当前 batch 的数据更新 ARI 累加器
        batch_size = pred_seg.shape[0]
        mask = torch.ones(batch_size, device=pred_seg.device).bool()
        ari_batch = adjusted_rand_index(
            pred_ids=pred_seg, true_ids=gt_seg,
            num_instances_pred=num_instances_pred,
            num_instances_true=num_instances_true,
            padding_mask=padding_mask)
        self.total += ari_batch[mask].sum().item()
        self.count += mask.sum().item()

    def result(self):
        # 返回当前平均 ARI
        return self.total / max(self.count, 1)

    def reset(self):
        # 重置累加器
        self.total = 0.0
        self.count = 0


class AriNoBg(Ari):
    # 无背景 ARI：计算 ARI 时忽略背景类别
    def update(self, pred_seg, gt_seg, padding_mask, num_instances_pred, num_instances_true):
        batch_size = pred_seg.shape[0]
        mask = torch.ones(batch_size, device=pred_seg.device).bool()
        ari_batch = adjusted_rand_index(
            pred_ids=pred_seg, true_ids=gt_seg,
            num_instances_pred=num_instances_pred,
            num_instances_true=num_instances_true,
            padding_mask=padding_mask,
            ignore_background=True)
        self.total += ari_batch[mask].sum().item()
        self.count += mask.sum().item()