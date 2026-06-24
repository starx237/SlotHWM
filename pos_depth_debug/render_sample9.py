"""
渲染 sample 9 的16帧视频 + slot 0 的注意力权重热力图
"""
import sys, os
sys.path.insert(0, '/autodl-fs/data/SlotHWM')
import warnings; warnings.filterwarnings('ignore')
import torch
import numpy as np
import yaml
from types import SimpleNamespace
from models.dynamics import SlotDynamicsModel
from data.obj3d_dataset import OBJ3DDataset
from PIL import Image, ImageDraw, ImageFont

with open('config/obj3d.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))

app_dim = cfg.appearance_dim
depth_max = getattr(cfg, 'depth_max', 0.30)
device = torch.device('cuda')

model = SlotDynamicsModel(cfg).to(device)
ckpt = torch.load(cfg.pretrained_path, map_location=device)
sd = model.state_dict()
ld = {}
for mk in sd:
    mc = mk.replace('_orig_mod.','')
    for ck in ckpt['model']:
        cc = ck.replace('_orig_mod.','')
        if cc==mc and ckpt['model'][ck].shape==sd[mk].shape:
            ld[mk]=ckpt['model'][ck]; break
model.load_state_dict(ld, strict=False)
model.eval()

ds = OBJ3DDataset(data_path='./data/obj3d', num_frames=16, stride=4, subsample=2)

SI = 9
TARGET_SLOT = 0

# 收集注意力权重
all_attns = []
def attn_hook(module, input, output):
    if isinstance(output, tuple) and len(output) == 2:
        all_attns.append(output[1].detach().cpu())
handle = model.slot_attention.register_forward_hook(attn_hook)

sample = ds[SI]
frames = sample['video'].unsqueeze(0).to(device)

with torch.no_grad():
    out = model(frames)

handle.remove()

corrected = out['slots']['corrected'][0]   # (6, N, 67)
target = out['slots']['target'][0]          # (10, N, 67)
all_slots = torch.cat([corrected, target], dim=0)  # (16, N, 67)

N_slots = all_slots.shape[1]
burnin = 6

# 解码每帧的 slot alpha 和 rgb
all_alphas = []
all_rgbs = []
for t in range(16):
    _, alpha_t, rgb_t = model.decoder(all_slots[t:t+1], return_rgb=True)
    all_alphas.append(alpha_t[0, :, 0])   # (N, H, W)
    all_rgbs.append(rgb_t[0])             # (N, 3, H, W)

alphas = torch.stack(all_alphas)  # (16, N, H, W)
rgbs = torch.stack(all_rgbs)      # (16, N, 3, H, W)

# 注意力热力图尺寸: 从 encoder 输出推断
# encoder 输出 256 个像素 = 16x16
ATTN_H = ATTN_W = 16

S = 64   # slot 图大小
PA = 2   # padding
ATTN_S = 64  # 注意力图显示大小（缩放到64x64）
LABEL_W = 80

# 布局: 每帧一列，行: Video | Recon | Slot0..N | Attn0..N
n_rows = 2 + N_slots + 1  # video, recon, slot0..5, attn_slot0
n_cols = 16

canvas_w = LABEL_W + n_cols * (S + PA)
canvas_h = n_rows * (S + PA)
canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
except:
    font = ImageFont.load_default()

def put_image(img_pil, r, c):
    x = LABEL_W + c * (S + PA)
    y = r * (S + PA)
    canvas.paste(img_pil, (x, y))

def tensor_to_pil(t):
    arr = t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype('uint8'))

# Row 0: Video
draw.text((2, 2), 'Video', fill=(0, 0, 0), font=font)
for t in range(16):
    put_image(tensor_to_pil(frames[0, t]), 0, t)

# Row 1: Recon (all slots combined)
draw.text((2, (S+PA)+2), 'Recon', fill=(0, 0, 0), font=font)
recon_frames = out['outputs']['video_burnin'][0] if 'video_burnin' in out['outputs'] else None
# 用 pred recons
if 'video_pred' in out['outputs']:
    burnin_recon = out['outputs']['video_burnin'][0]
    pred_recon = out['outputs']['video_pred'][0]
    full_recon = torch.cat([burnin_recon, pred_recon], dim=0)
else:
    full_recon = out['outputs']['video_burnin'][0]

for t in range(min(16, full_recon.shape[0])):
    put_image(tensor_to_pil(full_recon[t]), 1, t)

# Row 2-7: Slot 分解
for j in range(N_slots):
    y_text = (2 + j) * (S + PA) + 2
    depth_val = all_slots[0, j, app_dim+2].item()
    is_fg = depth_val < depth_max
    label = f'Slot {j}' + ('' if is_fg else ' BG')
    draw.text((2, y_text), label, fill=(0, 0, 0) if is_fg else (150, 150, 150), font=font)
    for t in range(16):
        arr_rgb = rgbs[t, j].detach().cpu().permute(1, 2, 0).numpy()
        arr_alpha = alphas[t, j].detach().cpu().numpy()
        arr_alpha = np.expand_dims(arr_alpha, axis=-1)
        display = arr_rgb * arr_alpha + (1.0 - arr_alpha)
        img = Image.fromarray((display * 255).astype('uint8'))
        put_image(img, 2 + j, t)

# Row 8: Slot 0 的注意力权重热力图
row_attn = 2 + N_slots
draw.text((2, row_attn * (S+PA) + 2), f'Attn s{TARGET_SLOT}', fill=(0, 128, 0), font=font)

for t in range(min(16, len(all_attns))):
    attn_t = all_attns[t][0, TARGET_SLOT].numpy()  # (256,)
    attn_2d = attn_t.reshape(ATTN_H, ATTN_W)
    
    # 缩放到 64x64
    attn_img = Image.fromarray((attn_2d * 255).astype('uint8')).resize((S, S), Image.NEAREST)
    
    # 用 colormap (简单: 黑=0, 绿=max)
    attn_norm = attn_2d / (attn_2d.max() + 1e-8)
    # 创建彩色版本
    colored = np.zeros((ATTN_H, ATTN_W, 3), dtype=np.uint8)
    colored[:, :, 1] = (attn_norm * 255).astype(np.uint8)  # green channel
    colored[attn_norm > 0.5, 0] = ((attn_norm[attn_norm > 0.5] - 0.5) * 2 * 255).astype(np.uint8)  # red for high
    attn_pil = Image.fromarray(colored).resize((S, S), Image.NEAREST)
    
    put_image(attn_pil, row_attn, t)

# 标注 burnin/rollout 分界线
for row in range(n_rows):
    x = LABEL_W + burnin * (S + PA) - 1
    y0 = row * (S + PA)
    draw.line([(x, y0), (x, y0 + S)], fill=(255, 0, 0), width=2)

# 标注 depth 值
depth_vals = all_slots[:, TARGET_SLOT, app_dim+2].cpu().numpy()
for t in range(16):
    x = LABEL_W + t * (S + PA) + 2
    y = row_attn * (S + PA) + S - 14
    draw.text((x, y), f'{depth_vals[t]:.3f}', fill=(255, 0, 0), font=font)

out_dir = '/autodl-fs/data/SlotHWM/pos_depth_debug'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f'sample{SI}_slots_attn.png')
canvas.save(out_path)
print(f'Saved: {out_path}')

# 也保存视频帧为 gif
video_frames = []
for t in range(16):
    frame = frames[0, t].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    video_frames.append(Image.fromarray((frame * 255).astype('uint8')))

gif_path = os.path.join(out_dir, f'sample{SI}_video.gif')
video_frames[0].save(gif_path, save_all=True, append_images=video_frames[1:], duration=200, loop=0)
print(f'Saved: {gif_path}')

# 打印 depth 信息
print(f'\nSample {SI}, Slot {TARGET_SLOT} depth:')
print(f'  {[f"{d:.4f}" for d in depth_vals]}')
print(f'  pos_x range: [{all_slots[:, TARGET_SLOT, app_dim].min():.3f}, {all_slots[:, TARGET_SLOT, app_dim].max():.3f}]')
print(f'  pos_y range: [{all_slots[:, TARGET_SLOT, app_dim+1].min():.3f}, {all_slots[:, TARGET_SLOT, app_dim+1].max():.3f}]')
