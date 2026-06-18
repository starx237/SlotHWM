import torch


def create_coordinate_grid(h, w, device):
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing='ij',
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid
