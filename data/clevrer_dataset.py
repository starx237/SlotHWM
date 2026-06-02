import torch
import random
import os
from .base_dataset import BaseVideoDataset


class CLEVRERDataset(BaseVideoDataset):
    def __init__(self, data_path, split="train", num_frames=5, img_size=(128, 128),
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
            self.video_ids = sorted(self.slots_data.keys())
        else:
            data_file = os.path.join(data_path, 'clevrer_data.pt')
            if os.path.exists(data_file):
                self.video_data = torch.load(data_file, weights_only=True)
                self.video_ids = sorted(self.video_data.keys())
                T = self.video_data[self.video_ids[0]].shape[0]
                self.windows = []
                for vid in self.video_ids:
                    for t in range(0, T - num_frames + 1, stride):
                        self.windows.append((vid, t))
            else:
                self.video_ids = list(range(10000))
                self.windows = [(i, 0) for i in range(10000)]

    def __len__(self):
        if self.extract_slots:
            return len(self.video_ids)
        return len(self.windows)

    def __getitem__(self, idx):
        if self.extract_slots:
            vid = self.video_ids[idx]
            slots = self.slots_data[vid]
            T = slots.shape[0]
            t = random.randint(1, T - self.num_frames)
            return {"slots": slots[t:t + self.num_frames], "video_id": vid}

        vid, t = self.windows[idx]
        if hasattr(self, 'video_data'):
            video = self.video_data[vid].float() / 255.0
        else:
            video = torch.randn(5, 3, 128, 128)
        frames = video[t:t + self.num_frames]
        return {"video": frames, "video_id": f"{vid}_t{t}"}
