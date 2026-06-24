"""
实验6: 长训练验证 - B(bnd_mask) vs A(orig)，500步
加 rollout_decay 看远帧 loss 权重衰减是否有帮助
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')
import warnings; warnings.filterwarnings('ignore')
import torch
import numpy as np
import yaml
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset

with open('config/obj3d.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))

app_dim = cfg.appearance_dim
depth_max = getattr(cfg, 'depth_max', 0.30)
boundary_threshold = 0.75
rollout = cfg.rollout_frames

def make():
    m = SlotDynamicsModel(cfg).cuda()
    ckpt = torch.load(cfg.pretrained_path, map_location='cpu')
    sd = m.state_dict()
    ld = {}
    for mk in sd:
        mc = mk.replace('_orig_mod.','')
        for ck in ckpt['model']:
            cc = ck.replace('_orig_mod.','')
            if cc==mc and ckpt['model'][ck].shape==sd[mk].shape:
                ld[mk]=ckpt['model'][ck]; break
    m.load_state_dict(ld, strict=False)
    for n,p in m.named_parameters():
        if 'spatiotemporal' not in n: p.requires_grad=False
    return m

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)

# 加载更多数据
data_list = []
for seed in range(6):
    torch.manual_seed(seed)
    data_list.append(next(iter(ds.get_dataloader(batch_size=2,shuffle=False,num_workers=0)))['video'].cuda())

def compute_boundary_mask(slots, depth_mask):
    B, T, N, D = slots.shape
    pos_x = slots[:, :, :, app_dim]
    pos_y = slots[:, :, :, app_dim+1]
    depth = slots[:, :, :, app_dim+2]
    boundary_mask = (pos_x.abs() < boundary_threshold) & (pos_y.abs() < boundary_threshold) & (depth < depth_max)
    return (depth_mask & boundary_mask).float()

# 3个模型: A(orig), B(bnd_mask), E(bnd_mask+decay)
mA = make()
mB = make()
mE = make()
models = {'A(orig)': mA, 'B(bnd_mask)': mB, 'E(bnd+decay)': mE}
opts = {k: torch.optim.Adam(filter(lambda p:p.requires_grad, m.parameters()), lr=1e-4) for k, m in models.items()}

# rollout_decay: 远帧权重递减
decay_weights = torch.tensor([0.9**t for t in range(rollout)], device='cuda')

for step in range(500):
    frames = data_list[step % len(data_list)]
    
    for name, mdl in models.items():
        mdl.train()
        out = mdl(frames)
        
        target = out['slots']['target'][:, :rollout].detach()
        pred = out['slots']['predicted'][:, :rollout]
        dm = out['depth_mask'][:, :rollout].detach()
        
        bnd_mask = compute_boundary_mask(target, dm)
        m0 = dm.unsqueeze(-1).float()
        bnd = bnd_mask.unsqueeze(-1)
        
        pred_dyn = pred[:, :, :, app_dim:]
        tgt_dyn = target[:, :, :, app_dim:]
        
        if name == 'A(orig)':
            loss = ((pred_dyn - tgt_dyn)**2 * m0).sum() / (m0.sum()*3+1e-8)
        elif name == 'B(bnd_mask)':
            loss = ((pred_dyn - tgt_dyn)**2 * bnd).sum() / (bnd.sum()*3+1e-8)
        elif name == 'E(bnd+decay)':
            # 加入时间衰减权重
            dw = decay_weights.view(1, -1, 1, 1)
            loss = ((pred_dyn - tgt_dyn)**2 * bnd * dw).sum() / (bnd.sum()*3+1e-8)
        
        opt = opts[name]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad, mdl.parameters()), 1.0)
        opt.step()
        
        if step % 100 == 0:
            print(f"step {step:3d} {name}: {loss.item():.6f}", flush=True)

# 评估
print("\n=== Evaluation ===")
torch.manual_seed(99)
ev = next(iter(ds.get_dataloader(batch_size=2,shuffle=False,num_workers=0)))['video'].cuda()

for name, mdl in [('Base', make())] + list(models.items()):
    mdl.eval()
    with torch.no_grad():
        out = mdl(ev)
    
    pred = out['slots']['predicted'][:, :rollout, :, app_dim:]
    tgt = out['slots']['target'][:, :rollout, :, app_dim:]
    dm = out['depth_mask'][:, :rollout].unsqueeze(-1).float()
    bnd = compute_boundary_mask(out['slots']['target'][:, :rollout], out['depth_mask'][:, :rollout]).unsqueeze(-1)
    
    pos_mse = ((pred[...,:2]-tgt[...,:2])**2*dm).sum()/(dm.sum()*2+1e-8)
    depth_mse = ((pred[...,2:3]-tgt[...,2:3])**2*dm).sum()/(dm.sum()+1e-8)
    bnd_pos = ((pred[...,:2]-tgt[...,:2])**2*bnd).sum()/(bnd.sum()*2+1e-8)
    bnd_depth = ((pred[...,2:3]-tgt[...,2:3])**2*bnd).sum()/(bnd.sum()+1e-8)
    
    print(f"  {name:20s}: pos={pos_mse.item():.6f} depth={depth_mse.item():.6f} | bnd_pos={bnd_pos.item():.6f} bnd_depth={bnd_depth.item():.6f}")
    
    for t in [0, 2, 4, 6, 8, 9]:
        m_t = dm[:,t]; b_t = bnd[:,t]
        pm = ((pred[:,t,:,:2]-tgt[:,t,:,:2])**2*m_t).sum()/(m_t.sum()*2+1e-8)
        dm_t = ((pred[:,t,:,2:3]-tgt[:,t,:,2:3])**2*m_t).sum()/(m_t.sum()+1e-8)
        bp = ((pred[:,t,:,:2]-tgt[:,t,:,:2])**2*b_t).sum()/(b_t.sum()*2+1e-8)
        bd = ((pred[:,t,:,2:3]-tgt[:,t,:,2:3])**2*b_t).sum()/(b_t.sum()+1e-8)
        print(f"    t={t}: pos={pm.item():.6f} depth={dm_t.item():.6f} | bnd_p={bp.item():.6f} bnd_d={bd.item():.6f}")
