"""
实验1 最简版: pos-only vs pos+depth，不用eval/train切换，不用DataLoader
"""
import sys
sys.path.insert(0, '/autodl-fs/data/SlotHWM')
import torch, torch.nn.functional as F, yaml
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset

with open('config/obj3d.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))

app_dim = cfg.appearance_dim
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
# 预加载2个batch的数据
torch.manual_seed(0)
b1 = next(iter(ds.get_dataloader(batch_size=4,shuffle=False,num_workers=0)))['video'].cuda()
torch.manual_seed(1)
b2 = next(iter(ds.get_dataloader(batch_size=4,shuffle=False,num_workers=0)))['video'].cuda()
batches = [b1, b2]

mA = make()
mB = make()
oA = torch.optim.Adam(filter(lambda p:p.requires_grad, mA.parameters()), lr=1e-4)
oB = torch.optim.Adam(filter(lambda p:p.requires_grad, mB.parameters()), lr=1e-4)

print("Starting training...")
for step in range(200):
    frames = batches[step % 2]
    
    # A: pos only
    with torch.no_grad():
        outA = mA(frames)
    tgt = outA['slots']['target'][:, :rollout]
    dm = outA['depth_mask'][:, :rollout].unsqueeze(-1).float()
    
    outA2 = mA(frames)
    pred = outA2['slots']['predicted'][:, :rollout]
    lossA = ((pred[:,:,:,app_dim:app_dim+2]-tgt[:,:,:,app_dim:app_dim+2])**2*dm).sum()/(dm.sum()*2+1e-8)
    oA.zero_grad(); lossA.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad,mA.parameters()),1.0)
    oA.step()
    
    # B: pos+depth
    with torch.no_grad():
        outB = mB(frames)
    tgtB = outB['slots']['target'][:, :rollout]
    dmB = outB['depth_mask'][:, :rollout].unsqueeze(-1).float()
    
    outB2 = mB(frames)
    predB = outB2['slots']['predicted'][:, :rollout]
    lossB = ((predB[:,:,:,app_dim:]-tgtB[:,:,:,app_dim:])**2*dmB).sum()/(dmB.sum()*3+1e-8)
    oB.zero_grad(); lossB.backward()
    torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad,mB.parameters()),1.0)
    oB.step()
    
    if step%20==0:
        print(f"step {step:3d}: A(pos)={lossA.item():.6f} B(all)={lossB.item():.6f}")

# eval
print("\n=== Eval ===")
mB.eval(); mA.eval()
torch.manual_seed(99)
ev = next(iter(ds.get_dataloader(batch_size=4,shuffle=False,num_workers=0)))['video'].cuda()
for label,m in [("A(pos)",mA),("B(all)",mB),("Base",make())]:
    m.eval()
    with torch.no_grad():
        o = m(ev)
    pd=o['slots']['predicted'][:, :rollout,:,app_dim:]
    td=o['slots']['target'][:, :rollout,:,app_dim:]
    mk=o['depth_mask'][:, :rollout].unsqueeze(-1).float()
    pm=((pd[...,:2]-td[...,:2])**2*mk).sum()/(mk.sum()*2+1e-8)
    dm=((pd[...,2:3]-td[...,2:3])**2*mk).sum()/(mk.sum()+1e-8)
    print(f"  {label}: pos={pm.item():.6f} depth={dm.item():.6f}")
