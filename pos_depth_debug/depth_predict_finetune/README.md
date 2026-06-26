# Depth Spread Predictor Finetune

## 存档点

- **路径**: `pos_depth_debug/depth_predict_finetune/depth_predict_2000.pt`
- **来源**: 从 `good_checkpoints/isa_single_poscosloss_40000.pt` 单帧微调 2000 步
- **训练配置**: burnin=1, detach_cospos=False, continue_pretrain=False, depth_weight=0.5

## 存档内容

```python
ckpt = torch.load('pos_depth_debug/depth_predict_finetune/depth_predict_2000.pt')
# ckpt['model']: SlotDynamicsModel 的 state_dict (key 已去掉 _orig_mod 前缀)
# ckpt['predictor']: DepthSpreadPredictor 的 state_dict
# ckpt['config']: 训练配置信息
```

## 使用方法

### 加载模型和 predictor

```python
import torch
from models.dynamics import SlotDynamicsModel
from types import SimpleNamespace
import yaml

# 加载配置
with open('config/pretrain_obj3d.yaml') as f:
    cfg = SimpleNamespace(**yaml.safe_load(f))
cfg.continue_pretrain = False
cfg.freeze_slot = False

# 创建模型
m = SlotDynamicsModel(cfg).cuda()

# 加载存档点
ckpt = torch.load('pos_depth_debug/depth_predict_finetune/depth_predict_2000.pt')

# 加载模型权重
sd = m.state_dict()
ld = {}
for mk in sd:
    mc = mk.replace('_orig_mod.', '')
    if mc in ckpt['model'] and ckpt['model'][mc].shape == sd[mk].shape:
        ld[mk] = ckpt['model'][mc]
m.load_state_dict(ld, strict=False)
m.eval()

# 加载 predictor
from pos_depth_debug.exp_depth_predict_finetune import DepthSpreadPredictor
predictor = DepthSpreadPredictor(hidden=32).cuda()
predictor.load_state_dict(ckpt['predictor'])
predictor.eval()
```

### 使用 predictor 预测 spread

```python
# slots: (1, N, D) 从 ISA 输出
depth = slots[0, :, app_dim + 2]  # (N,)
pred_spread = predictor(depth.unsqueeze(0))  # (1, N)
```

## DepthSpreadPredictor 结构

```python
class DepthSpreadPredictor(nn.Module):
    # 只看 depth (1维) -> 预测 alpha spread (1维)
    # Linear(1, 32) -> ReLU -> Linear(32, 32) -> ReLU -> Linear(32, 1)
```

## 训练效果

| 指标 | Baseline (40000.pt) | After 2000 steps |
|------|---------------------|-------------------|
| R²(depth→spread) | 0.504 | **0.897** |
| R²(predictor) | -0.574 | **0.822** |
| Recon loss | 0.000115 | 0.000239 |

关键: depth 和 alpha spread 的对齐从 R²=0.50 提升到 0.90, predictor 只看 depth 就能预测 spread

## 训练脚本

`pos_depth_debug/exp_depth_predict_finetune.py`

- Phase 1: 预训练 predictor (ISA 冻结, 500步)
- Phase 2: 联合训练 ISA + predictor (2000步, depth_weight=0.5)
