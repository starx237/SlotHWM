from .base_dataset import BaseVideoDataset
from .obj3d_dataset import OBJ3DDataset
from .clevrer_dataset import CLEVRERDataset

_DATASET_MAP = {
    "obj3d": OBJ3DDataset,
    "clevrer": CLEVRERDataset,
    "physion": OBJ3DDataset,
    "fluid": OBJ3DDataset,
    "realworld": OBJ3DDataset,
}

def get_dataset(name, **kwargs):
    cls = _DATASET_MAP.get(name, OBJ3DDataset)
    return cls(**kwargs)
