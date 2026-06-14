#!/usr/bin/env python3
# OBJ3D PNG -> .pt 字典 {video_id: (100, 3, 64, 64) uint8}
# 用法: python scripts/preprocess_obj3d.py [--num_videos N] [--workers 16]

import argparse, os, sys, glob, time, re, multiprocessing as mp

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

FRAMES_PER_VIDEO = 100
TARGET_SIZE = 64


def natural_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def process_video(video_dir):
    import numpy as np
    import torch
    from PIL import Image
    try:
        files = sorted(
            [f for f in os.listdir(video_dir) if f.endswith('.png')],
            key=natural_key
        )
        if len(files) != FRAMES_PER_VIDEO:
            return video_dir, None

        frames = []
        for f in files:
            img = Image.open(os.path.join(video_dir, f))
            img = img.convert('RGB').resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
            arr = np.array(img).transpose(2, 0, 1)
            frames.append(arr)

        video = np.stack(frames, axis=0)
        return video_dir, torch.from_numpy(video).to(torch.uint8)
    except Exception:
        return video_dir, None


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='downloads/obj3d/OBJ3D/train')
    parser.add_argument('--output', default='data/obj3d/obj3d_data.pt')
    parser.add_argument('--num_videos', type=int, default=None)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    data_dir = os.path.join(os.path.dirname(__file__), '..', args.data_dir)
    data_dir = os.path.abspath(data_dir)
    video_dirs = sorted([
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if args.num_videos:
        video_dirs = video_dirs[:args.num_videos]

    total = len(video_dirs)
    if total == 0:
        print(f"ERROR: no video directories found in {data_dir}!")
        sys.exit(1)
    print(f"Found {total} video directories, workers={args.workers}")

    # Resume: 跳过已处理的视频
    data = {}
    existing_ids = set()
    if os.path.exists(args.output):
        prev = torch.load(args.output, weights_only=True)
        existing_ids = set(prev.keys())
        data.update(prev)
        print(f"Resuming: loaded {len(prev)} existing videos from {args.output}")

    # 过滤出未处理的视频目录
    todo = []
    for vd in video_dirs:
        vid = os.path.basename(vd)
        if vid not in existing_ids:
            todo.append(vd)

    if not todo:
        print("All videos already processed. Nothing to do.")
        return

    skipped = total - len(todo)
    print(f"Skipping {skipped} already-processed, processing {len(todo)} videos...")

    t0 = time.time()
    done = 0
    good = len(data)

    with mp.Pool(args.workers) as pool:
        for path, tensor in pool.imap_unordered(process_video, todo, chunksize=4):
            done += 1
            if tensor is not None:
                vid = os.path.basename(path)
                data[vid] = tensor
                good = len(data)

            if done % 10 == 0 or done == len(todo):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(todo) - done) / rate if rate > 0 else 0
                print(
                    f"Progress: {done}/{len(todo)} ({done/len(todo)*100:.0f}%) | "
                    f"{rate:.1f} vid/s | ETA: {eta/60:.1f}min | good: {good}"
                )

    elapsed = time.time() - t0
    print(f"Done: {good}/{total} in {elapsed:.0f}s ({elapsed/total*1000:.0f} ms/vid)")

    if good == 0:
        print("ERROR: no videos processed successfully!")
        sys.exit(1)

    torch.save(data, args.output)
    size_mb = os.path.getsize(args.output) / 1024**2
    print(f"Saved {good} videos ({size_mb:.1f} MB) to {args.output}")

    k = sorted(data.keys())[0]
    print(f"  Sample: {k}, shape={list(data[k].shape)}, dtype={data[k].dtype}")


if __name__ == '__main__':
    mp.freeze_support()
    main()
