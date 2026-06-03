# base_dataset.py
# 基础视频数据集基类，定义了所有视频数据集的通用接口和方法

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Union
import random


class BaseVideoDataset(Dataset):
    """视频数据集的基类，继承自 PyTorch 的 Dataset。
    提供基础的视频帧采样和 DataLoader 创建功能，子类需实现 __getitem__ 方法。
    """
    def __init__(self, data_path, num_frames=6, img_size=(64, 64), stride=1,
                 split='train'):
        """初始化基类数据集。

        Args:
            data_path: 数据集路径（字符串）或配置对象
            num_frames: 每个样本采样的帧数，默认 6
            img_size: 图像尺寸 (H, W)，默认 (64, 64)
            stride: 帧采样步长，默认 1
            split: 数据划分（train/val/test），默认 'train'
        """
        # 支持传入 config 对象
        if not isinstance(data_path, (str, bytes)):
            cfg = data_path
            data_path = cfg.data_root
            num_frames = cfg.burnin_frames + cfg.rollout_frames
            split = getattr(cfg, 'split', split)
        self.data_path = data_path
        self.num_frames = num_frames
        self.img_size = img_size
        self.stride = stride
        self.split = split
        self.video_ids = []  # 存储所有视频的 ID 列表

    def __len__(self):
        """返回数据集中视频的总数"""
        return len(self.video_ids)

    def __getitem__(self, idx):
        """获取指定索引的视频数据，子类必须实现此方法"""
        raise NotImplementedError

    def get_dataloader(self, batch_size=64, shuffle=True, num_workers=4, generator=None):
        """创建并返回一个 DataLoader 实例。

        Args:
            batch_size: 批大小，默认 64
            shuffle: 是否打乱数据，默认 True
            num_workers: 数据加载的工作进程数，默认 4
            generator: 用于可复现 shuffling 的 torch.Generator

        Returns:
            DataLoader: 配置好的 DataLoader 对象
        """
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            generator=generator,
        )