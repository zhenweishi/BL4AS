from pathlib import Path
import monai
import monai.transforms as transforms
import pandas as pd
import torch
monai.data.set_track_meta(False)
from timm.layers.helpers import to_3tuple


class CacheCSVDataset(monai.data.CacheDataset):
    def __init__(self, src, transform=None, cache_rate=1.0, num_workers=1, **kwargs):
        data = monai.data.CSVDataset(src, **kwargs)
        # data = pd.read_csv(src).to_dict(orient='records')
        super().__init__(data, transform=transform, cache_rate=cache_rate, num_workers=num_workers)
        
