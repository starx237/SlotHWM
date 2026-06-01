#!/usr/bin/env python3
# OBJ3D TFRecord -> 单个 .pt 字典 {video_id: (100, 3, 64, 64) uint8}
# 使用 C++ protobuf 快速解析
# Usage: python scripts/preprocess_obj3d.py [--tfrecord PATH] [--output PATH] [--num_videos N]

import argparse, struct, os, sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'downloads', 'obj3d'))
from scene_pb2 import Scene, Features


def decode_frame(record_bytes):
    try:
        scene = Scene()
        scene.ParseFromString(record_bytes)
        features = Features()
        features.ParseFromString(scene.image)
        feat = features.feature.get('image')
        if feat and feat.HasField('bytes_list'):
            raw = b''.join(feat.bytes_list.value)  # 0.7ms for 12288 values
            if len(raw) == 12288:
                return np.frombuffer(raw, dtype=np.uint8).reshape(3, 64, 64)
    except Exception:
        pass
    return np.zeros((3, 64, 64), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord', default='downloads/obj3d/objects_room_train_subset.tfrecords')
    parser.add_argument('--output', default='data/obj3d/obj3d_data.pt')
    parser.add_argument('--num_videos', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    data_dict = {}
    frame_buf = []
    video_idx = 0

    with open(args.tfrecord, 'rb') as f:
        pbar = tqdm(desc="Processing")
        while True:
            header = f.read(12)
            if len(header) < 12: break
            length = struct.unpack('<Q', header[:8])[0]
            if length > 10**7: break
            rec = f.read(length)
            f.read(4)
            pbar.update(1)

            frame_buf.append(decode_frame(rec))

            if len(frame_buf) == 100:
                video = np.stack(frame_buf, axis=0)
                data_dict[f'video_{video_idx:04d}'] = torch.from_numpy(video)
                frame_buf = []
                video_idx += 1
                if args.num_videos and video_idx >= args.num_videos:
                    break

    pbar.close()
    torch.save(data_dict, args.output)
    print(f"Saved {len(data_dict)} videos to {args.output}")
    print(f"  File size: {os.path.getsize(args.output) / 1024**2:.1f} MB")


if __name__ == '__main__':
    main()
