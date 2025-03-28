import monai
import numpy as np
from pathlib import Path


class LoadImage(monai.transforms.LoadImage):
    def __init__(
        self,
        reader=None,
        image_only: bool = True,
        dtype=np.float32,
        ensure_channel_first=False,
        simple_keys=False,
        prune_meta_pattern=None,
        prune_meta_sep=".",
        expanduser=True,
        *args,
        **kwargs,
    ) -> None:
        
        if "base_dir" in kwargs:
            self.base_dir = kwargs.pop("base_dir")
        else:
            self.base_dir = None
        
        super().__init__(
            reader=reader,
            image_only=image_only,
            dtype=dtype,
            ensure_channel_first=ensure_channel_first,
            simple_keys=simple_keys,
            prune_meta_pattern=prune_meta_pattern,
            prune_meta_sep=prune_meta_sep,
            expanduser=expanduser,
            *args,
            **kwargs,
        )
    def __call__(self, filename, reader=None):
        if self.base_dir is not None:
            filename = Path(self.base_dir) / filename
        return super().__call__(filename, reader)
    

class LoadImaged(monai.transforms.LoadImaged):
    def __init__(
        self,
        keys,
        reader=None,
        dtype=np.float32,
        meta_keys=None,
        meta_key_postfix=monai.utils.enums.PostFix.meta(),
        overwriting=False,
        image_only=True,
        ensure_channel_first=False,
        simple_keys=False,
        prune_meta_pattern=None,
        prune_meta_sep=".",
        allow_missing_keys=False,
        expanduser=True,
        *args,
        **kwargs,
    ) -> None:
        if "base_dir" in kwargs:
            self.base_dir = kwargs.pop("base_dir")
        else:
            self.base_dir = None

        super().__init__(
            keys=keys,
            reader=reader,
            dtype=dtype,
            meta_keys=meta_keys,
            meta_key_postfix=meta_key_postfix,
            overwriting=overwriting,
            image_only=image_only,
            ensure_channel_first=ensure_channel_first,
            simple_keys=simple_keys,
            prune_meta_pattern=prune_meta_pattern,
            prune_meta_sep=prune_meta_sep,
            allow_missing_keys=allow_missing_keys,
            expanduser=expanduser,
            *args,
            **kwargs,
        )

    def __call__(self, data, reader=None):
        d = dict(data)
        if self.base_dir is not None:
            for key in self.keys:
                d[key] = Path(self.base_dir) / d[key]
        return super().__call__(d, reader)
    
LoadImageD = LoadImageDict = LoadImaged