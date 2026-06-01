# obj3d_dataset.py
# ObjectsRoom 3D 数据集加载模块
# 包含多个 3D 物体在房间内运动的合成视频场景

import torch
import random
from .base_dataset import BaseVideoDataset


class OBJ3DDataset(BaseVideoDataset):
    """ObjectsRoom 3D 数据集类，继承自 BaseVideoDataset。
    支持两种模式：
    1. extract_slots=False: 返回随机生成的模拟视频帧
    2. extract_slots=True: 从预先保存的 slot 文件中加载 slot 序列
    """
    def __init__(self, data_path, split="train", num_frames=6, img_size=(64, 64),
                 stride=1, subsample=1, slot_dim=128, extract_slots=False,
                 slot_path=None):
        """初始化 OBJ3D 数据集。

        Args:
            data_path: 数据集路径
            split: 数据划分（train/val/test），默认 "train"
            num_frames: 每个样本的帧数，默认 6
            img_size: 图像尺寸 (H, W)，默认 (64, 64)
            stride: 帧采样步长，默认 1
            subsample: 下采样因子，默认 1
            slot_dim: slot 特征维度，默认 128
            extract_slots: 是否从预提取的 slot 文件加载，默认 False
            slot_path: 预提取 slot 文件路径，extract_slots=True 时必须提供
        """
        super().__init__(data_path, num_frames, img_size, stride)
        self.split = split
        self.subsample = subsample
        self.extract_slots = extract_slots
        self.slot_path = slot_path

        if extract_slots:
            # 从预保存的 slot 文件加载数据
            self.slots_data = torch.load(slot_path)
            self.video_ids = list(self.slots_data.keys())
        else:
            # 使用模拟数据，固定 10000 个视频 ID
            self.video_ids = list(range(10000))

    def __len__(self):
        """返回数据集中视频的总数"""
        return len(self.video_ids)

    def __getitem__(self, idx):
        """获取指定索引的视频数据。

        Args:
            idx: 视频索引

        Returns:
            dict: 包含视频帧或 slot 序列以及视频 ID 的字典
        """
        vid = self.video_ids[idx]
        if self.extract_slots:
            # 从预提取的 slot 数据中随机采样一个连续子序列
            slots = self.slots_data[vid]
            T = slots.shape[0]
            t = random.randint(1, T - self.num_frames)
            slots_seq = slots[t:t + self.num_frames]
            return {"slots": slots_seq, "video_id": vid}
        else:
            # 生成随机模拟视频帧（仅用于测试/调试）
            frames = torch.randn(self.num_frames, 3, *self.img_size)
            return {"video": frames, "video_id": vid}