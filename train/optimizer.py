import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(model_params, config):
    lr = getattr(config, 'learning_rate', 2e-4)
    wd = getattr(config, 'weight_decay', 0.0)
    optimizer = AdamW(model_params, lr=lr, weight_decay=wd)

    warmup_steps = getattr(config, 'warmup_steps', 2500)
    total_steps = getattr(config, 'num_steps', 160000)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(
            (step - warmup_steps) / max(1, total_steps - warmup_steps) * 3.14159)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
