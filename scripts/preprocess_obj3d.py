#!/usr/bin/env python3
# OBJ3D TFRecord -> .pt 字典 {video_id: (100, 3, 64, 64) uint8}
# 支持分批续传：已有 .pt 文件时自动跳过已处理的视频
# 支持从 GCS 流式下载
# Usage:
#   python scripts/preprocess_obj3d.py --tfrecord <path> [--num_videos N]
#   gsutil cat gs://.../objects_room_train.tfrecords | \
#     python scripts/preprocess_obj3d.py --stdin [--num_videos N]

import argparse, struct, os, sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'downloads', 'obj3d'))
# 强制纯 Python protobuf，兼容老 protoc 生成的文件
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from scene_pb2 import Scene, Features

FRAMES_PER_VIDEO = 100


def decode_frame(record_bytes):
    try:
        scene = Scene()
        scene.ParseFromString(record_bytes)
        features = Features()
        features.ParseFromString(scene.image)
        # 兼容纯 Python protobuf（RepeatedCompositeFieldContainer）和 C++ 模式（map）
        feat = None
        if hasattr(features.feature, 'get'):
            feat = features.feature.get('image')
        else:
            for entry in features.feature:
                if entry.key == 'image':
                    feat = entry.value
                    break
        if feat and feat.HasField('bytes_list'):
            raw = b''.join(feat.bytes_list.value)
            if len(raw) == 12288:
                return np.frombuffer(raw, dtype=np.uint8).reshape(64, 64, 3).transpose(2, 0, 1)
    except Exception:
        pass
    return np.zeros((3, 64, 64), dtype=np.uint8)


def load_existing(output):
    if os.path.exists(output):
        size = os.path.getsize(output) / 1024**2
        print(f"Resuming: loading existing {output} ({size:.1f} MB)...")
        data = torch.load(output, weights_only=True)
        return data, len(data)
    return {}, 0


def skip_records(f, count):
    skipped = 0
    with tqdm(total=count, desc="Skipping existing", unit="rec") as pbar:
        while skipped < count:
            header = f.read(12)
            if len(header) < 12:
                return False
            length = struct.unpack('<Q', header[:8])[0]
            if length > 10**7:
                return False
            discarded = f.read(length + 4)
            if len(discarded) < length + 4:
                return False
            skipped += 1
            pbar.update(1)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord', type=str, default=None)
    parser.add_argument('--stdin', action='store_true')
    parser.add_argument('--output', default='data/obj3d/obj3d_data.pt')
    parser.add_argument('--num_videos', type=int, default=None,
                        help='Total videos desired after this run')
    args = parser.parse_args()

    if not args.tfrecord and not args.stdin:
        args.tfrecord = 'downloads/obj3d/objects_room_train_subset.tfrecords'

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    data_dict, videos_done = load_existing(args.output)
    records_to_skip = videos_done * FRAMES_PER_VIDEO

    if args.stdin:
        import gzip
        src = gzip.open(sys.stdin.buffer, 'rb')
    else:
        src = open(args.tfrecord, 'rb')

    with src as f:
        # 跳过已处理的记录
        if records_to_skip > 0:
            print(f"Skipping {records_to_skip} records ({videos_done} videos)...")
            if not skip_records(f, records_to_skip):
                print("No more data after skip — nothing new to process.")
                print(f"Total: {videos_done} videos")
                return

        max_videos = args.num_videos if args.num_videos else float('inf')
        frame_buf = []
        new_videos = 0
        save_interval = 25
        pbar = tqdm(desc="Processing")

        while new_videos + videos_done < max_videos:
            header = f.read(12)
            if len(header) < 12: break
            length = struct.unpack('<Q', header[:8])[0]
            if length > 10**7: break
            rec = f.read(length)
            f.read(4)
            pbar.update(1)

            frame_buf.append(decode_frame(rec))

            if len(frame_buf) == FRAMES_PER_VIDEO:
                video = np.stack(frame_buf, axis=0)
                data_dict[f'video_{new_videos + videos_done:04d}'] = torch.from_numpy(video)
                frame_buf = []
                new_videos += 1

                if new_videos % save_interval == 0:
                    torch.save(data_dict, args.output)
                    pbar.set_postfix({"saved": f"{new_videos + videos_done}"})

        pbar.close()

    if new_videos > 0:
        torch.save(data_dict, args.output)
    print(f"Existing: {videos_done}, New: {new_videos}, Total: {len(data_dict)}")
    size_mb = os.path.getsize(args.output) / 1024**2
    print(f"Saved to {args.output} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
