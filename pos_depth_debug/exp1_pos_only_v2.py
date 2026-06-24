"""
实验1 简化版: 只用 pos loss，更小 batch，更少步数
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset
from collections import deque

with open('/autodl-fs/data/SlotHWM/config/obj3d.yaml') as f:
    cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)

app_dim = cfg.appearance_dim
rollout = cfg.rollout_frames

def load_model():
    model = SlotDynamicsModel(cfg).cuda()
    ckpt = torch.load(cfg.pretrained_path, map_location='cpu')
    model_state = model.state_dict()
    loaded = {}
    for mk in model_state:
        mk_c = mk.replace('_orig_mod.', '')
        for ck in ckpt['model']:
            ck_c = ck.replace('_orig_mod.', '')
            if ck_c == mk_c and ckpt['model'][ck].shape == model_state[mk].shape:
                loaded[mk] = ckpt['model'][ck]
                break
    model.load_state_dict(loaded, strict=False)
    return model

# A: pos only  B: pos+depth
model_A = load_model()
model_B = load_model()

for m in [model_A, model_B]:
    for name, param in m.named_parameters():
        if 'spatiotemporal' not in name:
            param.requires_grad = False

opt_A = torch.optim.Adam(filter(lambda p: p.requires_grad, model_A.parameters()), lr=1e-4)
opt_B = torch.optim.Adam(filter(lambda p: p.requires_grad, model_B.parameters()), lr=1e-4)

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)
loader = ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)

avg_A = deque(maxlen=30)
avg_B = deque(maxlen=30)

for step in range(300):
    batch = next(iter(loader))
    frames = batch['video'].cuda()
    
    for mdl, opt, use_pos_only, avg_buf, label in [
        (model_A, opt_A, True, avg_A, "A(pos)"),
        (model_B, opt_B, False, avg_B, "B(all)"),
    ]:
        mdl.eval()
        with torch.no_grad():
            out_eval = mdl(frames)
        target = out_eval['slots']['target'][:, :rollout]
        dm = out_eval['depth_mask'][:, :rollout]
        
        mdl.train()
        out = mdl(frames)
        pred = out['slots']['predicted'][:, :rollout]
        mask = dm.unsqueeze(-1).float()
        
        if use_pos_only:
            pred_d = pred[:, :, :, app_dim:app_dim+2]
            tgt_d = target[:, :, :, app_dim:app_dim+2]
            loss = ((pred_d - tgt_d)**2 * mask).sum() / (mask.sum() * 2 + 1e-8)
        else:
            pred_d = pred[:, :, :, app_dim:]
            tgt_d = target[:, :, :, app_dim:]
            loss = ((pred_d - tgt_d)**2 * mask).sum() / (mask.sum() * 3 + 1e-8)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, mdl.parameters()), 1.0)
        opt.step()
        avg_buf.append(loss.item())
    
    if step % 30 == 0:
        a = sum(avg_A)/len(avg_A) if avg_A else 0
        b = sum(avg_B)/len(avg_B) if avg_B else 0
        print(f"step {step:3d}: A(pos)={a:.6f} B(all)={b:.6f}")

# 评估
print("\n=== Eval ===")
torch.manual_seed(999)
batch = next(iter(ds.get_dataloader(batch_size=4, shuffle=True, num_workers=0)))
frames = batch['video'].cuda()

for label, mdl in [("A(pos)", model_A), ("B(all)", model_B), ("Base", load_model())]:
    mdl.eval()
    with torch.no_grad():
        out = mdl(frames)
    pd = out['slots']['predicted'][:, :rollout, :, app_dim:]
    td = out['slots']['target'][:, :rollout, :, app_dim:]
    m = out['depth_mask'][:, :rollout].unsqueeze(-1).float()
    pm = ((pd[...,:2]-td[...,:2])**2*m).sum()/(m.sum()*2+1e-8)
    dm = ((pd[...,2:3]-td[...,2:3])**2*m).sum()/(m.sum()+1e-8)
    print(f"  {label:8s}: pos={pm.item():.6f} depth={dm.item():.6f}")
