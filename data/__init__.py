from .base_dataset import BaseVideoDataset
from .obj3d_dataset import OBJ3DDataset

_DATASET_MAP = {
    "obj3d": OBJ3DDataset,
    "clevrer": OBJ3DDataset,
    "physion": OBJ3DDataset,
    "fluid": OBJ3DDataset,
    "realworld": OBJ3DDataset,
}

def get_dataset(name, **kwargs):
    cls = _DATASET_MAP.get(name, OBJ3DDataset)
    return cls(**kwargs)
