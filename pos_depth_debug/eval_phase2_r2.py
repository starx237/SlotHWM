"""快速评估phase2 step_14000的R2"""
import torch, yaml, sys, warnings
import numpy as np
from numpy.polynomial import polynomial as P
warnings.filterwarnings('ignore')
sys.path.insert(0, '..')
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset
from types import SimpleNamespace

device = torch.device('cuda')
app_dim = 64

def load_model(ckpt_path, cfg):
    model = SlotDynamicsModel(cfg).cuda()
    ckpt = torch.load(ckpt_path, map_location='cuda')
    model_sd = model.state_dict()
    ckpt_sd = ckpt['model']
    loaded = {}
    for ck_key, v in ckpt_sd.items():
        ck_clean = ck_key.replace('_orig_mod.', '')
        for mk_key in model_sd:
            if ck_clean == mk_key.replace('_orig_mod.', '') and v.shape == model_sd[mk_key].shape:
                loaded[mk_key] = v; break
    model.load_state_dict(loaded, strict=False)
    return model

with open('../config/pretrain_phase2.yaml') as f: cfg_dict = yaml.safe_load(f)
cfg = SimpleNamespace(**cfg_dict)
cfg.pretrain = True; cfg.freeze_slot = False; cfg.continue_pretrain = False

ds = OBJ3DDataset(data_path='../data/obj3d', subsample=2)
H, W = 64, 64

model = load_model('../experiments/phase2_depth_spread/checkpoints/step_14000.pt', cfg)
model.eval()

all_depth = []; all_spread = []; all_cov = []; all_depth_raw = []

with torch.no_grad():
    gy, gx = torch.meshgrid(torch.linspace(-1,1,H,device=device), torch.linspace(-1,1,W,device=device), indexing='ij')
    for si in range(100):
        sample = ds[si]
        video = sample['video'].unsqueeze(0).to(device)
        out = model(video)
        slots = out['slots']['corrected'][0]
        T = slots.shape[0]; N = slots.shape[1]
        for t in range(T):
            _, a_full, _ = model.decoder(slots[t].unsqueeze(0), return_alphas=True, return_rgb=True)
            a = a_full[0, :, 0]
            dominant_slot = a.argmax(dim=0)
            for s in range(N):
                alpha_cov = a[s].sum().item()
                pixel_cov = (dominant_slot == s).sum().item()
                a_max = a[s].max().item()
                depth = slots[t, s, app_dim+2].item()
                all_depth_raw.append(depth)
                if a_max > 0.3 and 50 < pixel_cov < 1200 and depth < 0.5:
                    a_s = a[s]; a_n = a_s/(alpha_cov+1e-8)
                    cx = (a_n*gx).sum().item(); cy = (a_n*gy).sum().item()
                    sp = np.sqrt((a_n*((gx-cx)**2+(gy-cy)**2)).sum().item())
                    all_depth.append(depth)
                    all_spread.append(sp)
                    all_cov.append(alpha_cov/(H*W))

d = np.array(all_depth); sp = np.array(all_spread); c = np.array(all_cov)
d_all = np.array(all_depth_raw)
print(f'FG points: {len(d)}, depth range (all): [{d_all.min():.4f}, {d_all.max():.4f}]')
print(f'FG depth range (filtered): [{d.min():.4f}, {d.max():.4f}]')
print(f'Points with depth>0.5 filtered out: {(d_all > 0.5).sum()} / {len(d_all)}')

co1 = P.polyfit(d, sp, 1)
r2_sp = 1 - np.sum((sp-P.polyval(d,co1))**2)/np.sum((sp-sp.mean())**2)
d2 = d**2
co2 = P.polyfit(d2, c, 1)
r2_cov = 1 - np.sum((c-P.polyval(d2,co2))**2)/np.sum((c-c.mean())**2)
print(f'R²(depth, spread) = {r2_sp:.4f}')
print(f'R²(depth², cov) = {r2_cov:.4f}')
