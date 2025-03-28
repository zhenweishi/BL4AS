import os
from .io import *
from .intensity import *
# from .med_aug import *
# from .from_fmcib import *
# from .from_SelfMedMAE import *
# # from torchvision.transforms import Compose, Lambda

from monai.transforms.transform import Transform, MapTransform, RandomizableTransform
import SimpleITK as sitk

class Lambda(Transform):
    """
    Transform to apply a lambda function.
    """
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def __call__(self, data):
        return self.func(data)
    
class LambdaD(MapTransform):
    """
    Transform to apply a lambda function to a dictionary.
    """
    def __init__(self, keys, func):
        super().__init__(keys)
        self.func = func
    
    def __call__(self, data):
        for key in self.keys:
            data[key] = self.func(data[key])
        return data

class StopCachingFromHere(RandomizableTransform):
    """
    Transform to stop caching from here.
    """
    def __init__(self):
        super().__init__()
    
    def __call__(self, data):
        return data

class UnpackDict(Transform):
    """
    Transform to unpack the dictionary from a list.
    """
    
    def __call__(self, data):
        # Assuming data is a list of dictionaries
        if isinstance(data, list) and len(data) == 1:
            return data[0]
        return data
    
class GetFirstSampleOnly(RandomizableTransform):
    """
    Transform to get the first item from a list.
    """
    def __init__(self, image_key="image", transform=None):
        super().__init__()
        self.transform = transform
        self.image_key = image_key
    
    def __call__(self, data):
        data = self.transform(data)
        if isinstance(data, list):
            return data[0][self.image_key]
        return data[self.image_key]
    
class OnlyOneSample:
    """
    Transform to get only one sample from the dataset.
    """
    def __init__(self):
        super().__init__()
    
    def __call__(self, batch):
        # Assumes each item in the batch is a dictionary
        reorganized = {}
        for data in batch:
            for key, value in data.items():
                if key not in reorganized:
                    reorganized[key] = []
                reorganized[key].append(value)
        # Convert lists to tensors if necessary
        # This can be customized depending on your specific needs
        return reorganized

class PathJoin(Transform):
    """
    Transform to join the image path with the root path.
    """
    def __init__(self, base_dir, name_key, path_key):
        super().__init__()
        self.name_key = name_key
        self.base_dir = base_dir
        self.path_key = path_key

    def __call__(self, data: dict):
        data[self.path_key] = os.path.join(self.base_dir, data[self.name_key])
        return data
    

class ReadNPY(Transform):
    def __init__(self, key):
        super().__init__()
        self.key = key
    def __call__(self, data):
        nii_path = Path(data[self.key])
        npy_path = nii_path.parent / nii_path.name.replace(".nii.gz", ".npy")

        if not npy_path.exists():
            img = sitk.GetArrayFromImage(sitk.ReadImage(nii_path))
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, img)
        data[self.key] = npy_path

        return data