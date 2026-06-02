#!/usr/bin/env python3
# CLEVRER 预处理 — 多进程加速版 v3
# 输出到 log.txt。用法: python scripts/preprocess_clevrer.py [--workers 32]

import argparse, os, sys, glob, time, multiprocessing as mp

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

TARGET_FRAMES = 16
TARGET_SIZE = 128


def log(msg):
    with open('log.txt', 'a') as f:
        f.write(f'[{time.strftime("%H:%M:%S")}] {msg}\n')


def process_video(path):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import decord
    try:
        decord.bridge.set_bridge('torch')
        vr = decord.VideoReader(path)
        total = len(vr)
        indices = np.linspace(0, max(total - 1, 0), TARGET_FRAMES, dtype=np.int32)
        video = vr.get_batch(indices)
        video = video.permute(0, 3, 1, 2).float()
        H, W = video.shape[2], video.shape[3]
        if H != TARGET_SIZE or W != TARGET_SIZE:
            video = F.interpolate(video, size=(TARGET_SIZE, TARGET_SIZE), mode='bilinear')
        return path, video.to(torch.uint8)
    except Exception as e:
        return path, None


def main():
    import torch
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--extracted_dir', default='downloads/clevrer_extracted')
    parser.add_argument('--output', default='data/clevrer/clevrer_data.pt')
    parser.add_argument('--num_videos', type=int, default=None)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.extracted_dir, '**', '*.mp4'), recursive=True))
    if args.num_videos:
        files = files[:args.num_videos]
    total = len(files)

    log(f"Found {total} mp4 files, workers={args.workers}")
    if total == 0:
        log("ERROR: no mp4 files found!")
        sys.exit(1)

    t0 = time.time()
    data = {}
    done = 0
    with mp.Pool(args.workers) as pool:
        for path, tensor in pool.imap_unordered(process_video, files, chunksize=8):
            done += 1
            if tensor is not None:
                vid = os.path.relpath(path, args.extracted_dir).replace('/', '_').replace('.mp4', '')
                data[vid] = tensor
            if done % 10 == 0 or done == total:
                pct = done / total * 100
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate
                log(f"Progress: {done}/{total} ({pct:.0f}%) | {rate:.1f} vid/s | ETA: {eta/60:.1f}min | good: {len(data)}")

    elapsed = time.time() - t0
    good = len(data)
    log(f"Done: {good}/{total} in {elapsed:.0f}s ({elapsed/total*1000:.0f} ms/vid)")
    if good == 0:
        log("ERROR: no videos processed successfully!")
        sys.exit(1)

    torch.save(data, args.output)
    size = os.path.getsize(args.output) / 1024**2
    log(f"Saved {good} videos ({size:.0f} MB)")
    k = sorted(data.keys())[0]
    log(f"  Sample: {k}, shape={list(data[k].shape)}")


if __name__ == '__main__':
    mp.freeze_support()
    main()
