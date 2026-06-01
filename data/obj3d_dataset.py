import torch
import random
import os
from .base_dataset import BaseVideoDataset


class OBJ3DDataset(BaseVideoDataset):
    def __init__(self, data_path, split="train", num_frames=6, img_size=(64, 64),
                 stride=1, subsample=1, slot_dim=128, extract_slots=False,
                 slot_path=None):
        super().__init__(data_path, num_frames, img_size, stride)
        self.split = split
        self.subsample = subsample
        self.extract_slots = extract_slots
        self.slot_path = slot_path
        self.num_frames = num_frames

        if extract_slots:
            self.slots_data = torch.load(slot_path, weights_only=True)
            self.video_ids = list(self.slots_data.keys())
        else:
            data_file = os.path.join(data_path, 'obj3d_data.pt')
            if os.path.exists(data_file):
                self.video_data = torch.load(data_file, weights_only=True)
                self.video_ids = sorted(self.video_data.keys())
            else:
                self.video_ids = list(range(10000))

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        if self.extract_slots:
            slots = self.slots_data[vid]
            T = slots.shape[0]
            t = random.randint(1, T - self.num_frames)
            slots_seq = slots[t:t + self.num_frames]
            return {"slots": slots_seq, "video_id": vid}
        else:
            if hasattr(self, 'video_data'):
                video = self.video_data[vid].float() / 255.0
                T = video.shape[0]
                t = random.randint(0, T - self.num_frames)
                frames = video[t:t + self.num_frames]
                return {"video": frames, "video_id": vid}
            else:
                frames = torch.randn(self.num_frames, 3, 64, 64)
                return {"video": frames, "video_id": vid}
