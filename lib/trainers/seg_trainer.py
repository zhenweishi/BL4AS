import os
import math
from pathlib import Path
import time
from functools import partial
from matplotlib.pyplot import grid
import numpy as np
from numpy import nanmean, nonzero, percentile
from torchprofile import profile_macs

import monai

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import lib.models as models
# from third_libs.SelfMedMAE.lib.utils import SmoothedValue, concat_all_gather, LayerDecayValueAssigner

import wandb

# from third_libs.SelfMedMAE.lib.data.med_transforms import get_scratch_train_transforms, get_val_transforms, get_post_transforms, get_vis_transforms, get_raw_transforms
# from third_libs.SelfMedMAE.lib.data.med_datasets import get_msd_trainset, get_train_loader, get_val_loader, idx2label_all, btcv_8cls_idx
# from third_libs.SelfMedMAE.lib.tools.visualization import patches3d_to_grid, images3d_to_grid
from lib.trainers.base_trainer import BaseTrainer

from timm.data import Mixup
from timm.utils import accuracy
from timm.layers.helpers import to_3tuple

from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import compute_dice, compute_hausdorff_distance, compute_iou, ConfusionMatrixMetric, get_confusion_matrix, compute_confusion_matrix_metric

from collections import defaultdict, OrderedDict

import pdb
import json
from monai.config import KeysCollection, PathLike
from collections import deque

from packaging import version
_persistent_workers = False if version.parse(torch.__version__) < version.parse('1.8.2') else True

__all__ = ['SegTrainer']

idx2label_all = {
    'btcv': ['tumor'],
    # 'msd_brats': ['TC', 'WT', 'ET']
}

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
        print(f'count: {self.count} | total: {self.total}')

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def _compute_path(base_dir, element, check_path=False):
    """
    Args:
        base_dir: the base directory of the dataset.
        element: file path(s) to append to directory.
        check_path: if `True`, only compute when the result is an existing path.

    Raises:
        TypeError: When ``element`` contains a non ``str``.
        TypeError: When ``element`` type is not in ``Union[list, str]``.

    """

    def _join_path(base_dir: PathLike, item: PathLike):
        result = os.path.normpath(os.path.join(base_dir, item))
        if check_path and not os.path.exists(result):
            # if not an existing path, don't join with base dir
            return f"{item}"
        return f"{result}"

    if isinstance(element, (str, os.PathLike)):
        return _join_path(base_dir, element)
    if isinstance(element, list):
        for e in element:
            if not isinstance(e, (str, os.PathLike)):
                return element
        return [_join_path(base_dir, e) for e in element]
    return element


def _append_paths(base_dir: PathLike, is_segmentation: bool, items: list[dict]) -> list[dict]:
    """
    Args:
        base_dir: the base directory of the dataset.
        is_segmentation: whether the datalist is for segmentation task.
        items: list of data items, each of which is a dict keyed by element names.

    Raises:
        TypeError: When ``items`` contains a non ``dict``.

    """
    for item in items:
        if not isinstance(item, dict):
            raise TypeError(f"Every item in items must be a dict but got {type(item).__name__}.")
        for k, v in item.items():
            if k == "image" or is_segmentation and k == "label":
                item[k] = _compute_path(base_dir, v, check_path=False)
            else:
                # for other items, auto detect whether it's a valid path
                item[k] = _compute_path(base_dir, v, check_path=True)
    return items


def load_decathlon_datalist(
    data_list_file_path: PathLike,
    is_segmentation: bool = True,
    data_list_key: str = "training",
    base_dir= None,
) -> list[dict]:
    """Load image/label paths of decathlon challenge from JSON file

    Json file is similar to what you get from http://medicaldecathlon.com/
    Those dataset.json files

    Args:
        data_list_file_path: the path to the json file of datalist.
        is_segmentation: whether the datalist is for segmentation task, default is True.
        data_list_key: the key to get a list of dictionary to be used, default is "training".
        base_dir: the base directory of the dataset, if None, use the datalist directory.

    Raises:
        ValueError: When ``data_list_file_path`` does not point to a file.
        ValueError: When ``data_list_key`` is not specified in the data list file.

    Returns a list of data items, each of which is a dict keyed by element names, for example:

    .. code-block::

        [
            {'image': '/workspace/data/chest_19.nii.gz',  'label': 0},
            {'image': '/workspace/data/chest_31.nii.gz',  'label': 1}
        ]

    """
    data_list_file_path = Path(data_list_file_path)
    if not data_list_file_path.is_file():
        raise ValueError(f"Data list file {data_list_file_path} does not exist.")
    with open(data_list_file_path) as json_file:
        json_data = json.load(json_file)
    if data_list_key not in json_data:
        raise ValueError(f'Data list {data_list_key} not specified in "{data_list_file_path}".')
    expected_data = json_data[data_list_key]
    if data_list_key == "test" and not isinstance(expected_data[0], dict):
        # decathlon datalist may save the test images in a list directly instead of dict
        expected_data = [{"image": i} for i in expected_data]

    if base_dir is None:
        base_dir = data_list_file_path.parent

    return _append_paths(base_dir, is_segmentation, expected_data)

def get_json_trainset(args, workers, train_transform=None):
    data_dir = args.data_path
    print("=> Base dir: ", data_dir)
    print(f'=> Get trainset from specified json file {args.json_list}')
    datalist_json = os.path.join(data_dir, args.json_list)

    datalist = load_decathlon_datalist(datalist_json,
                                        True,
                                        "training",
                                        base_dir=data_dir)
    train_ds = monai.data.CacheDataset(
        data=datalist,
        transform=train_transform,
        cache_num=len(datalist),
        cache_rate=args.get('cache_rate', 1.0),
        num_workers=workers,
    )
    return train_ds

def get_json_valset(args, val_transform=None):
    data_dir = args.data_path
    print(f'=> Get valset from specified json file {args.json_list}')
    datalist_json = os.path.join(data_dir, args.json_list)

    val_files = load_decathlon_datalist(datalist_json,
                                        True,
                                        "validation",
                                        base_dir=data_dir)
    # val_ds = data.Dataset(data=val_files, transform=val_transform)
    val_ds = monai.data.CacheDataset(
        data=val_files,
        transform=val_transform,
        cache_num=len(val_files),
        cache_rate=args.get('cache_rate', 1.0),
        num_workers=args.workers if hasattr(args, 'workers') else 4,
    )
    return val_ds

def get_train_loader(args, batch_size, workers, train_transform=None):
    if args.dataset in ['btcv']:
        train_ds = get_json_trainset(args, 
                                     workers=workers, 
                                     train_transform=train_transform)
    elif args.dataset == 'msd_brats':
        train_ds = get_msd_trainset(args, workers, 
                                    train_transform=train_transform, 
                                    task='Task01_BrainTumour', 
                                    ts_fold=args.ts_fold, 
                                    seed=args.data_seed)
    else:
        raise NotImplementedError(f"{args.dataset} is not supported yet.")

    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = monai.data.DataLoader(train_ds,
                                    batch_size=batch_size,
                                    shuffle=(train_sampler is None),
                                    num_workers=workers,
                                    sampler=train_sampler,
                                    pin_memory=True,
                                    persistent_workers=_persistent_workers)
    return train_loader

def get_val_loader(args, batch_size, workers, val_transform=None):
    if args.dataset in ['btcv']:
        val_ds = get_json_valset(args, val_transform=val_transform)
    elif args.dataset == 'msd_brats':
        val_ds = get_msd_valset(args, 
                                workers, 
                                val_transform=val_transform,
                                task='Task01_BrainTumour',
                                ts_fold=args.ts_fold, 
                                seed=args.data_seed)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = monai.data.DataLoader(val_ds,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=workers,
                                 sampler=val_sampler,
                                 pin_memory=True,
                                 persistent_workers=_persistent_workers)
    return val_loader

def get_scratch_train_transforms(args):
    if args.dataset == 'btcv':
        train_transform = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=["image", "label"]),
                monai.transforms.EnsureChannelFirstd(channel_dim='no_channel', keys=["image", "label"]),
                monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                monai.transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                monai.transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                # clip=True) if not args.get("z_score", False) else monai.transforms.ScaleIntensityd(keys=["image"]),
                                                clip=True) if not args.get("z_score", False) else monai.transforms.NormalizeIntensityd(keys=["image"]),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                monai.transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=1,
                    num_samples=args.num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                monai.transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=0),
                monai.transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=1),
                monai.transforms.RandFlipd(keys=["image", "label"],
                                    prob=args.RandFlipd_prob,
                                    spatial_axis=2),
                monai.transforms.RandRotate90d(
                    keys=["image", "label"],
                    prob=args.RandRotate90d_prob,
                    max_k=3,
                ),
                monai.transforms.RandScaleIntensityd(keys="image",
                                            factors=0.1,
                                            prob=args.RandScaleIntensityd_prob),
                monai.transforms.RandShiftIntensityd(keys="image",
                                            offsets=0.1,
                                            prob=args.RandShiftIntensityd_prob),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'msd_brats':
        train_transform = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=["image", "label"]),
                monai.transforms.EnsureChannelFirstd(keys="image"),
                monai.transforms.EnsureChannelFirstd(channel_dim='no_channel', keys=["label"]),
                monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                monai.transforms.Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"], label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1, neg=1, num_samples=args.num_samples,
                    image_key="image", image_threshold=0),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                # monai.transforms.RandSpatialCropd(keys=["image", "label"], roi_size=(args.roi_x, args.roi_y, args.roi_z), random_size=False),
                monai.transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
                monai.transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
                monai.transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
                monai.transforms.RandRotate90d(
                    keys=["image", "label"],
                    prob=args.RandRotate90d_prob,
                    max_k=3,
                ),
                monai.transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                monai.transforms.ToTensord(keys=["image", "label"])
                # monai.transforms.EnsureTyped(keys=["image", "label"]),
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return train_transform

def get_val_transforms(args):
    if args.dataset == 'btcv':
        val_transform = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=["image", "label"]),
                monai.transforms.EnsureChannelFirstd(channel_dim='no_channel', keys=["image", "label"]),
                monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                monai.transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                monai.transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                # clip=True) if not args.get("z_score", False) else monai.transforms.ScaleIntensityd(keys=["image"]),
                                                clip=True) if not args.get("z_score", False) else monai.transforms.NormalizeIntensityd(keys=["image"]),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'msd_brats':
        val_transform = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=["image", "label"]),
                monai.transforms.EnsureChannelFirstd(keys="image"),
                monai.transforms.EnsureChannelFirstd(channel_dim='no_channel', keys=["label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                monai.transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(1.0, 1.0, 1.0),
                                    mode=("bilinear", "nearest")),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                monai.transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                monai.transforms.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return val_transform

def get_vis_transforms(args):
    if args.dataset == 'btcv':
        val_transform = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=["image", "label"]),
                monai.transforms.EnsureChannelFirstd(channel_dim='no_channel', keys=["image", "label"]),
                monai.transforms.Orientationd(keys=["image", "label"],
                                        axcodes="RAS"),
                monai.transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(args.space_x, args.space_y, args.space_z),
                                    mode=("bilinear", "nearest")),
                monai.transforms.ScaleIntensityRanged(keys=["image"],
                                                a_min=args.a_min,
                                                a_max=args.a_max,
                                                b_min=args.b_min,
                                                b_max=args.b_max,
                                                # clip=True) if not args.get("z_score", False) else monai.transforms.ScaleIntensityd(keys=["image"]),
                                                clip=True) if not args.get("z_score", False) else monai.transforms.NormalizeIntensityd(keys=["image"]),
                monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                # monai.transforms.RandCropByPosNegLabeld(
                #     keys=["image", "label"],
                #     label_key="label",
                #     spatial_size=(args.roi_x, args.roi_y, args.roi_x),
                #     pos=1,
                #     neg=1,
                #     num_samples=1,
                #     image_key="image",
                #     image_threshold=0,
                # ),
                monai.transforms.CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(args.roi_x, args.roi_y, args.roi_z)
                ),
                monai.transforms.ToTensord(keys=["image", "label"]),
            ]
        )
    elif args.dataset == 'msd_brats':
        val_transform = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=["image", "label"]),
                monai.transforms.EnsureChannelFirstd(keys="image"),
                monai.transforms.EnsureChannelFirstd(channel_dim='no_channel', keys=["label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                monai.transforms.Spacingd(keys=["image", "label"],
                                    pixdim=(1.0, 1.0, 1.0),
                                    mode=("bilinear", "nearest")),
                # monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                # monai.transforms.RandCropByPosNegLabeld(
                #     keys=["image", "label"],
                #     label_key="label",
                #     spatial_size=(args.roi_x, args.roi_y, args.roi_x),
                #     pos=1,
                #     neg=1,
                #     num_samples=1,
                #     image_key="image",
                #     image_threshold=0,
                # ),
                monai.transforms.CenterSpatialCropd(
                    keys=["image", "label"],
                    roi_size=(args.roi_x, args.roi_y, args.roi_z)
                ),
                monai.transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                monai.transforms.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        raise ValueError(f"Only support BTCV transforms for medical images")
    return val_transform


def get_vit_layer_id(var_name, num_max_layer, prefix=''):
    if var_name in (prefix + "cls_token", prefix + "mask_token", prefix + "pos_embed"):
        return 0
    elif var_name.startswith(prefix + "patch_embed"):
        return 0
    elif var_name.startswith(prefix + "rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith(prefix + "blocks"):
        names = var_name.split('.')
        anchor_ind = names.index('blocks') # 'blocks' is an anchor
        block_id = int(names[anchor_ind + 1])
        return block_id + 1
    else:
        return num_max_layer - 1

class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        if layer_id is not None:
            return self.values[layer_id]
        else:
            return 1

    def get_layer_id(self, var_name, prefix=''):
        return get_vit_layer_id(var_name, len(self.values), prefix)

def get_post_transforms(args):
    if args.dataset == 'btcv':
        if args.test:
            post_pred = monai.transforms.Compose([monai.transforms.EnsureType(),
                                            # Resize(scale_params=(args.space_x, args.space_y, args.space_z)),
                                            monai.transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = monai.transforms.Compose([monai.transforms.EnsureType(),
                                            # Resize(scale_params=(args.space_x, args.space_y, args.space_z)),
                                            monai.transforms.AsDiscrete(to_onehot=args.num_classes)])
        else:
            post_pred = monai.transforms.Compose([monai.transforms.EnsureType(),
                                            monai.transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
            post_label = monai.transforms.Compose([monai.transforms.EnsureType(),
                                            monai.transforms.AsDiscrete(to_onehot=args.num_classes)])
    elif args.dataset == 'msd_brats':
        post_pred = monai.transforms.Compose([monai.transforms.EnsureType(), monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)])
        post_label = monai.transforms.Identity()
    return post_pred, post_label

def compute_confusion_metric(metric_name, y_pred, y, include_background=True):
    confusion_matrix = get_confusion_matrix(y_pred, y, include_background=include_background)
    return compute_confusion_matrix_metric(confusion_matrix=confusion_matrix, metric_name=metric_name)

def compute_avg_metric(metric, meters, metric_name, batch_size, args):
    assert len(metric.shape) == 2
    # if args.dataset == 'btcv':
    #     # cls_avg_metric = np.nanmean(np.nanmean(metric, axis=0))
    #     cls_avg_metric = np.mean(np.ma.masked_invalid(np.nanmean(metric, axis=0)))
    #     # cls8_avg_metric = np.nanmean(np.nanmean(metric[..., btcv_8cls_idx], axis=0))
    #     cls8_avg_metric = np.nanmean(np.ma.masked_invalid(np.nanmean(metric[..., btcv_8cls_idx], axis=0)))
    #     meters[metric_name].update(value=cls_avg_metric, n=batch_size)
    #     meters[f'cls8_{metric_name}'].update(value=cls8_avg_metric, n=batch_size)
    if args.dataset == 'btcv':
        avg_metric = np.nanmean(np.nanmean(metric, axis=0))
        # print("avg_metric:",avg_metric)
        if not np.isnan(avg_metric).any():
            meters[metric_name].update(value=avg_metric, n=batch_size)
        else:
            print("avg_metric is nan")
    else:
        cls_avg_metric = np.nanmean(np.nanmean(metric, axis=0))
        if not np.isnan(cls_avg_metric).any():
            meters[metric_name].update(value=cls_avg_metric, n=batch_size)
        else:
            print("cls_avg_metric is nan")

class SegTrainer(BaseTrainer):
    r"""
    General Segmentation Trainer
    """
    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.proj_name
        self.scaler = torch.cuda.amp.GradScaler()
        if args.test:
            self.metric_funcs = OrderedDict([
                                        ('Dice', 
                                          compute_dice),
                                        ('IoU',
                                          compute_iou),
                                        ('HD',
                                          partial(compute_hausdorff_distance, percentile=95)),
                                        ('PPV',
                                          partial(compute_confusion_metric, metric_name='precision')),
                                        ('SEN',
                                         partial(compute_confusion_metric, metric_name='sensitivity')),
                                        # ('HitRate',
                                        #  partial(compute_confusion_metric, metric_name='hit_rate')),
                                        ])
        else:
            self.metric_funcs = OrderedDict([
                                        ('Dice', 
                                          compute_dice)
                                        ])

    def build_model(self, *args, **kwargs):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name}")

            if args.dataset == 'btcv':
                # args.num_classes = 2 # WZT
                args.num_classes = args.get("num_classes", 2)
                self.loss_fn = DiceCELoss(to_onehot_y=True,
                                          softmax=True,
                                          squared_pred=True,
                                          smooth_nr=args.smooth_nr,
                                          smooth_dr=args.smooth_dr)
            elif args.dataset == 'msd_brats':
                args.num_classes = 3
                self.loss_fn = DiceLoss(to_onehot_y=False, 
                                        sigmoid=True, 
                                        squared_pred=True, 
                                        smooth_nr=args.smooth_nr, 
                                        smooth_dr=args.smooth_dr)
            else:
                raise ValueError(f"Unsupported dataset {args.dataset}")
            self.post_pred, self.post_label = get_post_transforms(args)

            # setup mixup and loss functions
            if args.mixup > 0:
                raise NotImplemented("Mixup for segmentation has not been implemented.")
            else:
                self.mixup_fn = None

            self.model = getattr(models, self.model_name)(encoder=getattr(models, args.enc_arch),
                                                          decoder=getattr(models, args.dec_arch),
                                                          args=args)

            # load pretrained weights

            if args.get("debug", False) or hasattr(args, 'test') and args.test and args.pretrain is not None and os.path.exists(args.pretrain):
                print(f"=> Start loading the model weights from {args.pretrain} for test")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                state_dict = checkpoint['state_dict']
                msg = self.model.load_state_dict(state_dict, strict=False)
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")
            elif args.pretrain is not None: 
                if not os.path.exists(args.pretrain):
                    raise ValueError(f"Pretrained weights {args.pretrain} does not exist.")
                print(f"=> Start loading pretrained weights from {args.pretrain}")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                # import pdb
                # pdb.set_trace()
                if self.model_name == 'UNETR3D':
                    for key in list(state_dict.keys()):
                        if key.startswith('encoder.'):
                            state_dict[key[len('encoder.'):]] = state_dict[key]
                            del state_dict[key]
                        if key.startswith("base_encoder."):
                            state_dict[key[len("base_encoder."):]] = state_dict[key]
                            del state_dict[key]
                        # need to concat and load pos embed. too
                        # TODO: unify the learning of pos embed of pretraining and finetuning
                        if key == 'encoder_pos_embed':
                            pe = torch.zeros([1, 1, state_dict[key].size(-1)])
                            state_dict['pos_embed'] = torch.cat([pe, state_dict[key]], dim=1)
                            del state_dict[key]
                        if key == 'patch_embed.proj.weight' and \
                            state_dict['patch_embed.proj.weight'].shape != self.model.encoder.patch_embed.proj.weight.shape:
                            del state_dict['patch_embed.proj.weight']
                            del state_dict['patch_embed.proj.bias']
                        if key == 'pos_embed' and \
                            state_dict['pos_embed'].shape != self.model.encoder.pos_embed.shape:
                            del state_dict[key]
                        if key.startswith('momentum_encoder') or key.startswith('predictor') or key.startswith('predictor2'):
                            del state_dict[key]
                        if key.startswith('decoder'):
                            del state_dict[key]
                    msg = self.model.encoder.load_state_dict(state_dict, strict=False)
                elif self.model_name == 'DynSeg3d':
                    if args.pretrain_load == 'enc+dec':
                        for key in list(state_dict.keys()):
                            if key.startswith('decoder.head.') or (key.startswith('decoder.blocks.') and int(key[15]) > 7):
                                del state_dict[key]
                    elif args.pretrain_load == 'enc':
                        for key in list(state_dict.keys()):
                            if key.startswith('decoder.'):
                                del state_dict[key]
                    msg = self.model.load_state_dict(state_dict, strict=False)
                # self.model.load(state_dict)
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")

            if args.get("freeze_encoder", False):
                print("=> Freeze encoder weights")
                for name, param in self.model.encoder.named_parameters():
                    param.requires_grad = False

            self.wrap_model()
        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        assert(self.model is not None and self.wrapped_model is not None), \
                "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args
        model = self.model

        num_layers = model.get_num_layers()
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))

        # optim_params = self.group_params(model)
        optim_params = self.get_parameter_groups(get_layer_id=partial(assigner.get_layer_id, prefix='encoder.'), 
                                                 get_layer_scale=assigner.get_scale, 
                                                 verbose=False)
        # TODO: create optimizer factory
        self.optimizer = torch.optim.AdamW(optim_params, 
                                            lr=args.lr,
                                            betas=(args.beta1, args.beta2),
                                            weight_decay=args.weight_decay)

    def build_dataloader(self, fold=0):
        if self.dataloader is None:
            print("=> creating train dataloader")
            args = self.args

            if args.dataset in ['btcv', 'msd_brats']:
                # build train dataloader
                if not args.test:
                    train_transform = get_scratch_train_transforms(args)
                    self.dataloader = get_train_loader(args, 
                                                    batch_size=self.batch_size, 
                                                    workers=self.workers, 
                                                    train_transform=train_transform)
                    self.iters_per_epoch = len(self.dataloader)
                    print(f"==> Length of train dataloader is {self.iters_per_epoch}")
                else:
                    self.iters_per_epoch = 0
                    self.dataloader = None
                # build val dataloader
                val_transform = get_val_transforms(args)
                self.val_dataloader = get_val_loader(args, 
                                                     batch_size=args.val_batch_size, # batch per gpu
                                                     workers=self.workers, 
                                                     val_transform=val_transform)
                # build vis dataloader
                vis_transform = get_vis_transforms(args)
                self.vis_dataloader = get_val_loader(args,
                                                    batch_size=args.vis_batch_size,
                                                    workers=self.workers,
                                                    val_transform=vis_transform)
            elif args.dataset == 'brats20':
                raise NotImplementedError("brats20 transforms and dataloaders on MONAI has not been implemented yet.")
            else:
                raise ValueError("Currently only support brats2020 dataset")
        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")
        print("=> finish creating dataloader")

    def run(self):
        args = self.args
        # Compute iterations when resuming
        niters = args.start_epoch * self.iters_per_epoch

        best_metric = 0
        best_ts_metric = 0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                self.dataloader.sampler.set_epoch(epoch)
                torch.distributed.barrier()
            
            if epoch == args.start_epoch:
                if args.get('test', False):
                    self.evaluate(epoch=epoch, niters=niters)
                    break

            # train for one epoch
            niters = self.epoch_train(epoch, niters)

            # evaluate after each epoch training
            if (epoch + 1) % args.eval_freq == 0:
                start_eval_epoch = args.get("start_eval_epoch", -1)

                if start_eval_epoch > 0 and (epoch + 1) < start_eval_epoch:
                    print(f"Skip eval at epoch {epoch + 1} because start_eval_epoch is {start_eval_epoch}")
                else:
                    metric_list = self.evaluate(epoch=epoch, niters=niters)
                    metric = metric_list[0]
                    if len(metric_list) == 2:
                        ts_metric = metric_list[1]
                    else:
                        ts_metric = None
                    if metric > best_metric:
                        print(f"=> New val best metric: {metric} | Old val best metric: {best_metric}!")
                        best_metric = metric
                        #if metric > best_metric or (ts_metric is not None and ts_metric > best_ts_metric):
                        if ts_metric is not None:
                            print(f"=> New ts best metric: {ts_metric} | Old ts best metric: {best_ts_metric}!")
                            best_ts_metric = ts_metric
                        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                            self.save_checkpoint(
                                {
                                    'epoch': epoch + 1,
                                    'arch': args.arch,
                                    'state_dict': self.model.state_dict(),
                                    'optimizer' : self.optimizer.state_dict(),
                                    'scaler': self.scaler.state_dict(), # additional line compared with base imple
                                    'metric':metric
                                }, 
                                is_best=False, 
                                filename=f'{args.ckpt_dir}/best_model.pth.tar'
                            )
                            print("=> Finish saving best model.")
                    else:
                        print(f"=> Still old val best metric: {best_metric}")
                        if ts_metric is not None:
                            print(f"=> Still old ts best metric: {best_ts_metric}")

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if (epoch + 1) % args.save_freq == 0:
                    #TODO: save the best
                    self.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': self.model.state_dict(),
                            'optimizer' : self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(), # additional line compared with base imple
                        }, 
                        is_best=False, 
                        filename=f'{args.ckpt_dir}/checkpoint_{epoch:04d}.pth.tar'
                    )

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler
        mixup_fn = self.mixup_fn
        loss_fn = self.loss_fn

        # switch to train mode
        model.train()
        if args.get("freeze_encoder", False):
            model.encoder.eval()

        load_start_time = time.time()
        for i, batch_data in enumerate(train_loader):
            load_time = time.time() - load_start_time
            # adjust learning at the beginning of each iteration
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            image = batch_data['image']
            target = batch_data['label']

            # print(image.shape)

            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            if mixup_fn is not None:
                image, target = mixup_fn(image, target)

            # compute output and loss
            forward_start_time = time.time()
            # forward_start_time_1 = time.perf_counter()
            with torch.cuda.amp.autocast(True):
                loss = self.train_class_batch(model, image, target, loss_fn)
            forward_time = time.time() - forward_start_time

            # compute gradient and do SGD step
            bp_start_time = time.time()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bp_time = time.time() - bp_start_time

            # torch.cuda.synchronize()
            # print(f"training iter time is {time.perf_counter() - forward_start_time_1}")

            if 'lr_scale' in optimizer.param_groups[0]:
                last_layer_lr = optimizer.param_groups[0]['lr'] / optimizer.param_groups[0]['lr_scale']
            else:
                last_layer_lr = optimizer.param_groups[0]['lr']

            # Log to the screen
            if i % args.print_freq == 0:
                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.iters_per_epoch} | "
                      f"TotalIter: {niters:06d} | "
                      f"Init Lr: {self.lr:.05f} | "
                      f"Lr: {last_layer_lr:.05f} | "
                      f"Load Time: {load_time:.03f}s | "
                      f"Forward Time: {forward_time:.03f}s | "
                      f"Backward Time: {bp_time:.03f}s | "
                      f"Loss: {loss.item():.03f}")
                
            if args.rank == 0 and args.get("use_tensorboard", False):
                args.summary_writer.add_scalar("loss", loss.item(), epoch * self.iters_per_epoch + i)
                args.summary_writer.add_scalar("lr", last_layer_lr, epoch * self.iters_per_epoch + i)
                args.summary_writer.add_scalar("epoch", epoch, epoch * self.iters_per_epoch + i)



                # if args.rank == 0 and not args.disable_wandb:
                #     wandb.log(
                #         {
                #         "lr": last_layer_lr,
                #         "Loss": loss.item(),
                #         },
                #         step=niters,
                #     )

            niters += 1
            load_start_time = time.time()
        return niters

    @staticmethod
    def train_class_batch(model, samples, target, criterion):
        outputs = model(samples)
        loss = criterion(outputs, target)
        return loss

    @torch.no_grad()
    def evaluate(self, epoch=0, niters=0):
        print("=> Start Evaluating")
        args = self.args
        model = self.wrapped_model
        val_loader = self.val_dataloader
        
        if args.spatial_dim == 3:
            roi_size = (args.roi_x, args.roi_y, args.roi_z)
        elif args.spatial_dim == 2:
            roi_size = (args.roi_x, args.roi_y)
        else:
            raise ValueError(f"Do not support this spatial dimension (={args.spatial_dim}) for now")

        meters = defaultdict(SmoothedValue)
        if hasattr(args, 'ts_ratio') and args.ts_ratio != 0:
            assert args.batch_size == 1, "Test mode requires batch size 1"
            ts_samples = int(len(val_loader) * args.ts_ratio)
            val_samples = len(val_loader) - ts_samples
            ts_meters = defaultdict(SmoothedValue)
        else:
            ts_samples = 0
            val_samples = len(val_loader)
            ts_meters = None
        print(f"val samples: {val_samples} and test samples: {ts_samples}")

        # switch to evaluation mode
        model.eval()
        for i, batch_data in enumerate(val_loader):
            image, target = batch_data['image'], batch_data['label']
            if args.gpu is not None:
                image = image.to(args.gpu, non_blocking=True)
                target = target.to(args.gpu, non_blocking=True)
            if target.sum() == 0:
                print(f"skip the {i}th batch because it has no foreground.")
                continue

            # compute output
            with torch.cuda.amp.autocast():
                output = sliding_window_inference(image,
                                                  roi_size=roi_size,
                                                  sw_batch_size=4,
                                                  predictor=model,
                                                  overlap=args.infer_overlap)

            target_convert = torch.stack([self.post_label(target_tensor) for target_tensor in decollate_batch(target)], dim=0)
            output_convert = torch.stack([self.post_pred(output_tensor) for output_tensor in decollate_batch(output)], dim=0)

            batch_size = image.size(0)
            if args.test:
                seg_dir = Path(args.ckpt_dir).parent / 'test_seg'
                seg_dir.mkdir(exist_ok=True, parents=True)
                for j in range(batch_size):
                    seg_path = seg_dir / f"{batch_data['filename'][j]}"
                    print(f"Saving {seg_path}")
                    import SimpleITK as sitk
                    # sitk.WriteImage(sitk.GetImageFromArray(output_convert[j, 1].permute(2, 1, 0).cpu().numpy()), str(seg_path))
                    sitk.WriteImage(sitk.GetImageFromArray(output_convert[j].argmax(dim=0).permute(2, 1, 0).cpu().numpy().astype(np.uint8)), str(seg_path))

            idx2label = idx2label_all[args.dataset]
            for metric_name, metric_func in self.metric_funcs.items():
                if i < val_samples:
                    log_meters = meters
                else:
                    log_meters = ts_meters
                metric = metric_func(y_pred=output_convert, y=target_convert, include_background=False if args.dataset == 'btcv' else True)
                metric = metric.cpu().numpy()
                print("metric:", metric_name)
                compute_avg_metric(metric, log_meters, metric_name, batch_size, args)
                for k in range(metric.shape[-1]):
                    cls_metric = np.nanmean(metric, axis=0)[k]
                    print(f"cls_{k}_{metric_name}:",cls_metric)
                    if np.isnan(cls_metric) or np.isinf(cls_metric):
                        continue
                    log_meters[f'{idx2label[k]}.{metric_name}'].update(value=cls_metric, n=batch_size)
                print(f"avg_{metric_name}:", log_meters[metric_name].global_avg)
            print(f'==> Evaluating on the {i+1}th batch is finished.\n')
            # break

        # gather the stats from all processes
        if args.distributed:
            for k, v in meters.items():
                print(f'==> start synchronizing meter {k}...')
                v.synchronize_between_processes()
                print(f'==> finish synchronizing meter {k}...')
            if ts_meters is not None:
                for k, v in ts_meters.items():
                    print(f'==> start synchronizing meter {k}...')
                    v.synchronize_between_processes()
                    print(f'==> finish synchronizing meter {k}...')
        # pdb.set_trace()
        log_string = f"==> Epoch {epoch:04d} val results: \n"
        for k, v in meters.items():
            global_avg_metric = v.global_avg
            new_line = f"===> {k}: {global_avg_metric:.05f} \n"
            log_string += new_line
        print(log_string)
        
        # csv output
        headers = ""
        values = ""
        for k, v in meters.items():
            if "." in k:
                continue
            headers = headers + "," + k
            values = values + "," + f"{v.global_avg}"
        headers = headers.strip(",")
        values = values.strip(",")
        print(f"{headers}\n{values}\n")

        # save the ts_meters: dict to json
        if args.rank == 0:
            results_dict = {}
            for k, v in meters.items():
                results_dict[k] = float(v.global_avg)
            
            if ts_meters is not None:
                ts_results_dict = {}
                for k, v in ts_meters.items():
                    ts_results_dict[k] = float(v.global_avg)
                results_dict["test_results"] = ts_results_dict
            
            with open(Path(args.ckpt_dir).parent / 'test_results.json', "w") as f:
                json.dump(results_dict, f, indent=4)

        if ts_meters is not None:
            log_string = f"==> Epoch {epoch:04d} test results: \n"
            for k, v in ts_meters.items():
                global_avg_metric = v.global_avg
                new_line = f"===> {k}: {global_avg_metric:.05f} \n"
                log_string += new_line
            print(log_string)


        if args.rank == 0 and args.get("use_tensorboard", False):
            args.summary_writer.add_scalar("val_dice", meters['Dice'].global_avg, epoch)
            if ts_meters is not None:
                args.summary_writer.add_scalar("ts_dice", ts_meters['Dice'].global_avg, epoch)
        # if args.rank == 0 and not args.disable_wandb:
        #     wandb_log_dict = {}
        #     for k, v in meters.items():
        #         wandb_log_dict[k] = v.global_avg
        #     wandb.log(wandb_log_dict, step=niters)
        print("=> Finish Evaluating")

        if args.dataset == 'btcv':
            #assert ts_meters is None
            #return [ts_meters['global_lung_dice'].global_avg]
            if ts_meters is None:
                return [meters['Dice'].global_avg]
            else:
                return [meters['Dice'].global_avg, ts_meters['Dice'].global_avg]
        elif args.dataset == 'msd_brats':
            if ts_meters is None:
                return [meters['Dice'].global_avg]
            else:
                return [meters['Dice'].global_avg, ts_meters['Dice'].global_avg]

    @torch.no_grad()
    def visualize(self, channel_ind=0, directory='seg_vis'):
        print("=> Start Visualization")
        args = self.args
        model = self.wrapped_model
        val_loader = self.val_dataloader
        
        directory = os.path.join(args.output_dir, directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if args.spatial_dim == 3:
            roi_size = (args.roi_x, args.roi_y, args.roi_z)
        elif args.spatial_dim == 2:
            roi_size = (args.roi_x, args.roi_y)
        else:
            raise ValueError(f"Do not support this spatial dimension (={args.spatial_dim}) for now")

        # switch to evaluation mode
        model.eval()
        for i, batch_data in enumerate(val_loader):
            if i > 10:
                break
            image, target = batch_data['image'], batch_data['label']
            if args.gpu is not None:
                image = image.to(args.gpu, non_blocking=True)
                target = target.to(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = sliding_window_inference(image,
                                                  roi_size=roi_size,
                                                  sw_batch_size=4,
                                                  predictor=model,
                                                  overlap=args.infer_overlap)
            image_list = [im for im in decollate_batch(image)]
            target_convert = [self.post_label(target_tensor) for target_tensor in decollate_batch(target)]
            output_convert = [self.post_pred(output_tensor) for output_tensor in decollate_batch(output)]

            from matplotlib import cm
            import matplotlib.pyplot as plt
            for image_t, target_t, output_t in zip(image_list, target_convert, output_convert):
                depth = target_t.size(3)
                for ratio in [1/5, 2/5, 3/5, 4/5]:
                    image = image_t.permute(1, 2, 3, 0)[:, :, int(depth*ratio), channel_ind]
                    target = target_t.permute(1, 2, 3, 0)[:, :, int(depth*ratio), :]
                    output = output_t.permute(1, 2, 3, 0)[:, :, int(depth*ratio), :]
                    if args.dataset == 'btcv':
                        vmin, vmax = 0, 12
                        target_mask = target.argmax(dim=-1)
                        output_mask = output.argmax(dim=-1)
                        target_alphas = 1 - target[..., 0]
                        output_alphas = 1 - output[..., 0]
                    elif args.dataset == 'msd_brats':
                        vmin, vmax = 0, 3
                        target_mask = torch.zeros(image.shape).int()
                        target_mask[target[:,:,1].bool()] = 1
                        target_mask[target[:,:,0].bool()] = 2
                        target_mask[target[:,:,2].bool()] = 3
                        target_alphas = (target_mask > 0).float()

                        output_mask = torch.zeros(image.shape).int()
                        output_mask[output[:,:,1].bool()] = 1
                        output_mask[output[:,:,0].bool()] = 2
                        output_mask[output[:,:,2].bool()] = 3
                        output_alphas = (output_mask > 0).float()

                    image = image.cpu().numpy()
                    target_mask = target_mask.cpu().numpy()
                    output_mask = output_mask.cpu().numpy()
                    target_alphas = target_alphas.cpu().numpy()
                    output_alphas = output_alphas.cpu().numpy()
                    # pdb.set_trace()
                    print("start saving")
                    # image
                    fig = plt.figure(frameon=False)
                    # fig.set_size_inches(16, 16)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(image, cmap='gray')
                    fig.savefig(os.path.join(directory, f'{args.dataset}_{i:02d}_image_depth{int(ratio*100):02d}.png'))
                    print(f"finish saving image {i}")

                    # target
                    fig = plt.figure(frameon=False)
                    # fig.set_size_inches(16, 16)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(image, cmap='gray')
                    ax.imshow(target_mask, alpha=target_alphas, vmin=vmin, vmax=vmax, cmap='viridis')
                    fig.savefig(os.path.join(directory, f'{args.dataset}_{i:02d}_gt_depth{int(ratio*100):02d}.png'))
                    print(f"finish saving gt {i}")

                    # output
                    fig = plt.figure(frameon=False)
                    # fig.set_size_inches(16, 16)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    ax.imshow(image, cmap='gray')
                    ax.imshow(output_mask, alpha=output_alphas, vmin=vmin, vmax=vmax, cmap='viridis')
                    fig.savefig(os.path.join(directory, f'{args.dataset}_{i:02d}_out_depth{int(ratio*100):02d}.png'))
                    print(f"finish saving output {i}")
    
    def mask2image(self, images, targets, outputs, id, channel_ind=0, directory='seg_vis_policy'):
        args = self.args

        directory = os.path.join(args.output_dir, directory)
        if not os.path.exists(directory):
            os.makedirs(directory)

        image_list = [im for im in decollate_batch(images)]
        target_convert = [self.post_label(target_tensor) for target_tensor in decollate_batch(targets)]
        output_convert = [self.post_pred(output_tensor) for output_tensor in decollate_batch(outputs)]
        from matplotlib import cm
        import matplotlib.pyplot as plt
        for image_t, target_t, output_t in zip(image_list, target_convert, output_convert):
            depth = target_t.size(3)
            for i, ratio in enumerate([0.3, 0.4, 0.5, 0.6, 0.7]):
                image = image_t.permute(1, 2, 3, 0)[:, :, int(depth*ratio), channel_ind]
                target = target_t.permute(1, 2, 3, 0)[:, :, int(depth*ratio), :]
                output = output_t.permute(1, 2, 3, 0)[:, :, int(depth*ratio), :]
                if args.dataset == 'btcv':
                    vmin, vmax = 0, 12
                    target_mask = target.argmax(dim=-1)
                    output_mask = output.argmax(dim=-1)
                    target_alphas = 1 - target[..., 0]
                    output_alphas = 1 - output[..., 0]
                elif args.dataset == 'msd_brats':
                    vmin, vmax = 0, 3
                    target_mask = torch.zeros(image.shape).int()
                    target_mask[target[:,:,1].bool()] = 1
                    target_mask[target[:,:,0].bool()] = 2
                    target_mask[target[:,:,2].bool()] = 3
                    target_alphas = (target_mask > 0).float()

                    output_mask = torch.zeros(image.shape).int()
                    output_mask[output[:,:,1].bool()] = 1
                    output_mask[output[:,:,0].bool()] = 2
                    output_mask[output[:,:,2].bool()] = 3
                    output_alphas = (output_mask > 0).float()

                image = image.cpu().numpy()
                target_mask = target_mask.cpu().numpy()
                output_mask = output_mask.cpu().numpy()
                target_alphas = target_alphas.cpu().numpy()
                output_alphas = output_alphas.cpu().numpy()
                # pdb.set_trace()
                print("start saving")

                # target
                fig = plt.figure(frameon=False)
                # fig.set_size_inches(16, 16)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(image, cmap='gray')
                ax.imshow(target_mask, alpha=target_alphas, vmin=vmin, vmax=vmax, cmap='viridis')
                fig.savefig(os.path.join(directory, f'{args.dataset}_bid{id}_depidx{i}_gt.png'))
                print(f"finish saving gt {i}")

                # output
                fig = plt.figure(frameon=False)
                # fig.set_size_inches(16, 16)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(image, cmap='gray')
                ax.imshow(output_mask, alpha=output_alphas, vmin=vmin, vmax=vmax, cmap='viridis')
                fig.savefig(os.path.join(directory, f'{args.dataset}_bid{id}_depidx{i}.png'))
                print(f"finish saving output {id}")

    
    def vis_policy(self, niters=0, modality_idx=0, image_alpha=0.5):
        args = self.args
        loader = self.vis_dataloader
        model = self.wrapped_model

        model.eval()

        for b_idx, batch_data in enumerate(loader):
            image = batch_data['image']
            target = batch_data['label']
            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True) # [B, 4, IH, IW, ID]

            # inference to get policy list
            # pdb.set_trace()
            output, policy_list = model(image, return_policy=True) # policy list of [num_layers+1] 0-or-1 tensors with shape of [B, L+1]

            print("start saving segmentation image")

            self.mask2image(image, target, output, b_idx)
            print("finish saving")


            image = image.to('cpu')
            
            grid_size = []
            input_size = (args.roi_x, args.roi_y, args.roi_z)
            for pa_size, in_size in zip(to_3tuple(args.patch_size), to_3tuple(input_size)):
                grid_size.append(in_size // pa_size)
            data_dim = len(grid_size)

            # build policy map
            color_stride = 1. / len(policy_list)
            B = policy_list[0].size(0)
            accum_policy_map = torch.zeros(B, 3, *input_size, device=torch.device('cpu'))
            # pdb.set_trace()
            for k in range(len(policy_list)):
                policy = policy_list[k]
                # if k == 0:
                #     policy = 1 - policy_list[k]
                # elif k == len(policy_list) - 1:
                #     policy = policy_list[k]
                # else:
                #     policy = policy_list[k - 1] - policy_list[k]
                policy = policy[:, 1:].cpu()
                # print(f"policy shape: {policy.size()}")
                B, L = policy.size()
                assert L == np.prod(grid_size), "policy length does not match predefined grid size"
                # expand policy to be full resolution
                policy = policy.reshape(B, 1, *grid_size, *[1 for _ in range(data_dim)]) # [B, 1, H, W, D, 1, 1, 1]
                expand_shape = [-1] * (2 + data_dim) + [args.patch_size] * data_dim
                policy = policy.expand(expand_shape) # [B, 1, H, W, D, h, w, d]
                # permute policy & reshape
                permute_idx = [0, 1]
                for i in range(data_dim):
                    permute_idx.append(i + 2)
                    permute_idx.append(i + 2 + data_dim)
                policy = policy.permute(permute_idx) # [B, 1, H, h, W, w, D, d]
                policy = policy.reshape(B, 1, *input_size)
                chan_pad = torch.zeros(B, 2, *input_size, dtype=policy.dtype, device=policy.device)
                policy = torch.cat([policy, chan_pad], dim=1) # [B, 3, IH, IW, ID]
                # colorize
                # if k < len(policy_list) - 1:
                # policy = policy * color_stride * (k + 1)
                # policy = policy * color_stride * (k + 1)
                policy = policy * color_stride * k
                accum_policy_map += policy

            image_slice = image[:, modality_idx : modality_idx + 1, ...]
            image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
            image_slice = image_slice.expand(-1, 3, *[-1 for _ in range(data_dim)])
            # pdb.set_trace()
            alpha_image_w_policy = image_alpha * image_slice + \
                (1 - image_alpha) * accum_policy_map
            
            vis_tensor = torch.cat([image_slice, alpha_image_w_policy], dim=0)
                
            # visualize
            list_vis_grid_hw = images3d_to_grid(vis_tensor, n_group=2, hidden_axis='d')
            # vis_grid_hd = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size, in_chans=args.in_chans, hidden_axis='w')
            # vis_grid_wd = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size, in_chans=args.in_chans, hidden_axis='h')

            print("wandb logging")
            vis_hw_03 = wandb.Image(list_vis_grid_hw[0], caption=f"hw03_iter{niters:06d}_{b_idx}")
            vis_hw_04 = wandb.Image(list_vis_grid_hw[1], caption=f"hw04_iter{niters:06d}_{b_idx}")
            vis_hw_05 = wandb.Image(list_vis_grid_hw[2], caption=f"hw05_iter{niters:06d}_{b_idx}")
            vis_hw_06 = wandb.Image(list_vis_grid_hw[3], caption=f"hw06_iter{niters:06d}_{b_idx}")
            vis_hw_07 = wandb.Image(list_vis_grid_hw[4], caption=f"hw07_iter{niters:06d}_{b_idx}")

            # wandb.log(
            #     {
            #         "vis_hw_03": vis_hw_03,
            #         "vis_hw_04": vis_hw_04,
            #         "vis_hw_05": vis_hw_05,
            #         "vis_hw_06": vis_hw_06,
            #         "vis_hw_07": vis_hw_07
            #     },
            #     step=b_idx,
            # )
            if b_idx > 6:
                break
        print("finish wandb logging")

    def speedometer(self, niters=0):
        args = self.args
        loader = self.vis_dataloader
        model = self.wrapped_model

        model.eval()

        time_meters = defaultdict(list)
        num_trials = 5
        for t in range(num_trials):
            len_ds = len(loader)
            for b_idx, batch_data in enumerate(loader):
                image = batch_data['image']
                # target = batch_data['label']
                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True) # [B, 4, IH, IW, ID]
                start_time = time.time()
                if b_idx > 1 and b_idx < len_ds - 1:
                    model(image, time_meters=time_meters)
                    end_time  = time.time()
                    time_meters['total'].append(end_time - start_time)
                else:
                    model(image)
            print(f"finish trial {t}")
        for key in time_meters.keys():
            # print(f'num of records in {key} is {len(time_meters[key])}')
            avg_time = np.mean(time_meters[key])
            print(f"=> averaged inference time for {key} is {avg_time}")
        # print(f"==> avg inference time over all trials is {np.mean(trials)}")

    def speedometerv2(self):
        args = self.args
        model = self.wrapped_model

        model.eval()

        time_meters = defaultdict(list)
        num_trials = 16
        for t in range(num_trials):
            image = torch.rand(args.batch_size, args.in_chans, args.roi_x, args.roi_y, args.roi_z)
            single_image = torch.rand(1, args.in_chans, args.roi_x, args.roi_y, args.roi_z)
            if args.gpu is not None:
                single_image = single_image.cuda(args.gpu, non_blocking=True) # [B, 4, IH, IW, ID]
                image = image.cuda(args.gpu, non_blocking=True) # [B, 4, IH, IW, ID]
            print(f"image shape is {image.shape}")
            if t == 0:
                try:
                    macs = profile_macs(model, single_image) * 1e-9
                except:
                    macs = -1
                print(f"MACS is {macs} G")
            # target = batch_data['label']
            if t > 2 and t < 13:
                start_time = time.perf_counter()
                model(image, time_meters=time_meters)
                torch.cuda.synchronize()
                end_time  = time.perf_counter()
                time_meters['total'].append(end_time - start_time)
            else:
                model(image)
            print(f"finish trial {t}")
        for key in time_meters.keys():
            # print(f'num of records in {key} is {len(time_meters[key])}')
            avg_time = np.mean(time_meters[key])
            print(f"=> averaged inference time for {key} is {avg_time}")
        print(f"MACS is {macs} G")
        print(f"{4 / np.mean(time_meters['total'])} total")
        print(f"{4 / np.mean(time_meters['enc'])} enc")
        # print(f"==> avg inference time over all trials is {np.mean(trials)}")

    def calc_sparsity(self):
        args = self.args
        raw_transform = get_raw_transforms(args)
        raw_dataloader = get_val_loader(args,
                                batch_size=1,
                                workers=1,
                                val_transform=raw_transform)
        if args.dataset == 'msd_brats':
            cls_sparsity_sum = [0] * 3
        elif args.dataset == 'btcv':
            cls_sparsity_sum = [0] * 14
        for b_idx, batch_data in enumerate(raw_dataloader):
            print("=========================================")
            print(f"the {b_idx} sample")
            # image = batch_data['image']
            target = batch_data['label']
            # pdb.set_trace()
            # print(f"{b_idx}th sample")
            volume_shape = target.shape
            num_voxels = volume_shape[-3] * volume_shape[-2] * volume_shape[-1]
            if args.dataset == 'msd_brats':
                for c in range(3):
                    cls_target = target[0, c]
                    num_target = cls_target.sum()
                    sparsity = (num_voxels - num_target) / num_voxels
                    cls_sparsity_sum[c] += sparsity
                    print("-----------------------")
                    print(f"current class {c} sparsity is ({num_voxels} - {num_target})/{num_voxels} = {sparsity}")
                    print(f"current class {c} accum sparsity is {cls_sparsity_sum[c]/(b_idx+1)}")
            elif args.dataset == 'btcv':
                for c in range(14):
                    if c == 0:
                        num_target = num_voxels - (target[0, 0] == c).sum()
                    else:
                        num_target = (target[0, 0] == c).sum()
                    sparsity = (num_voxels - num_target) / num_voxels
                    cls_sparsity_sum[c] += sparsity
                    print("-----------------------")
                    print(f"current class {c} sparsity is ({num_voxels} - {num_target})/{num_voxels} = {sparsity}")
                    print(f"current class {c} accum sparsity is {cls_sparsity_sum[c]/(b_idx+1)}")
                

    def resume(self, *args, **kwargs):
        args = self.args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            
            if not args.test:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.scaler.load_state_dict(checkpoint['scaler'])
                    print("=> loaded optimizer and scaler state")
                except ValueError as e:
                    print(f"=> Warning: {str(e)}")
                    print("=> Could not load optimizer state (parameter groups mismatch). Continuing with initialized optimizer.")
            else:
                print("=> Test mode: skipping optimizer and scaler loading")
            
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise FileNotFoundError


    def adjust_learning_rate(self, epoch, args):
        """Base schedule: CosineDecay with warm-up."""
        init_lr = self.lr
        if epoch < args.warmup_epochs:
            cur_lr = init_lr * epoch / args.warmup_epochs
        else:
            cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if 'lr_scale' in param_group:
                param_group['lr'] = cur_lr * param_group['lr_scale']
            else:
                param_group['lr'] = cur_lr
