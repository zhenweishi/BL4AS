import os
from pathlib import Path
import monai
import pandas as pd
import SimpleITK as sitk
from tqdm.auto import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import label, find_objects


def worker(func, kwargs_list, use_multiprocessing, desc=""):
    desc = desc or func.__name__ + "_with multiprocessing" if use_multiprocessing else func.__name__
    pbar = tqdm(kwargs_list, desc=desc) 
    if use_multiprocessing:
        with ProcessPoolExecutor() as executor:
            futures = []
            futures_dict = {}
            for kwargs in kwargs_list:
                _kwargs = {k: v for k, v in kwargs.items() if not (k.startswith('__') and k.endswith('__'))}
                f = executor.submit(func, **_kwargs)
                futures.append(f)
                futures_dict[f] = kwargs.get('__name__')
            for f in as_completed(futures):
                name = futures_dict[f]
                if name:
                    pbar.set_postfix_str(f'{name} is finished') # 并行时，tqdm只能显示完成的名字，不能显示正在处理的名字
                pbar.update(1)
    else:
        for kwargs in pbar:
            _kwargs = {k: v for k, v in kwargs.items() if not (k.startswith('__') and k.endswith('__'))}
            name = kwargs.get('__name__')
            if name:
                pbar.set_postfix_str(f'{name} is processing')
            func(**_kwargs)

def generate_breast_region(img, mask):
    return img * mask

def filter_connected_components(mask_array, cc_threshold):
    """
    根据连通域中的像素点个数阈值过滤连通域。
    
    参数:
        mask_array (np.array): 输入的二维或三维分割图像（假设为0和1的数组）。
        cc_threshold (int): 连通域大小的阈值。
        
    返回:
        np.array: 过滤后的分割图像。
    """
    # 找到所有连通域
    labeled_array, num_features = label(mask_array)

    if num_features == 1:
        return mask_array
    
    # 创建一个新的数组用于保存过滤后的连通域
    filtered_mask = np.zeros_like(mask_array)
    
    # 遍历每个连通域
    for i in range(1, num_features + 1):
        # 计算当前连通域的大小
        cc_size = np.sum(labeled_array == i)
        
        # 如果连通域大小大于阈值，则保留该连通域
        if cc_size > cc_threshold:
            filtered_mask[labeled_array == i] = 1
            
    return filtered_mask.astype(np.uint8)


def crop_center(img, mask, size):
    roi_start, roi_end = monai.transforms.generate_spatial_bounding_box(np.expand_dims(mask, axis=0), margin=0, allow_smaller=False)
    roi_center = [(roi_start[i] + roi_end[i]) // 2 for i in range(3)]

    cropper = monai.transforms.Compose([
        monai.transforms.ToTensor(track_meta=False),
        monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
        monai.transforms.SpatialCrop(roi_center=roi_center, roi_size=(size, size, size)),
        monai.transforms.SqueezeDim(dim=0),
    ])
    return cropper(img), cropper(mask)

def crop_bbox(img, mask):
    roi_start, roi_end = monai.transforms.generate_spatial_bounding_box(np.expand_dims(mask, axis=0), margin=0, allow_smaller=False)
    cropper = monai.transforms.Compose([
        monai.transforms.ToTensor(track_meta=False),
        monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
        monai.transforms.SpatialCrop(roi_start=roi_start, roi_end=roi_end),
        monai.transforms.SqueezeDim(dim=0),
    ])
    return cropper(img), cropper(mask)

def crop_resize_roi(img, mask, size, method="all_roi_resize"):
    roi_start, roi_end = monai.transforms.generate_spatial_bounding_box(np.expand_dims(mask, axis=0), margin=0, allow_smaller=False)
    roi_center = [(roi_start[i] + roi_end[i]) // 2 for i in range(3)]
    roi_size = [roi_end[i] - roi_start[i] for i in range(3)]

    if method == "all_roi_resize":
        crop_size = max(roi_size)
        resizer = monai.transforms.Resize(spatial_size=(size, size, size))
    elif method == "big_roi_resize":
        crop_size = max(max(roi_size), size)
        resizer = monai.transforms.Resize(spatial_size=(size, size, size)) if crop_size > size else monai.transforms.Lambda(lambda x: x)
    else:
        raise ValueError(f"Invalid method: {method}")
    
    cropper = monai.transforms.Compose([
        monai.transforms.ToTensor(track_meta=False),
        monai.transforms.EnsureChannelFirst(channel_dim='no_channel'),
        monai.transforms.SpatialCrop(roi_center=roi_center, roi_size=crop_size),
        resizer,
        monai.transforms.SqueezeDim(dim=0),
    ])
    return cropper(img), cropper(mask)

def run_crop(img_path, mask_path, method, dst_img_path, dst_mask_path, size, largest, cc_threshold=0, norm=False):
    if dst_img_path.exists() and dst_mask_path.exists():
        return
    img = sitk.ReadImage(str(img_path))
    old_img_type = img.GetPixelID()
    img = sitk.GetArrayFromImage(img)

    if norm:
        img = (img - img.mean()) / (img.std() + 1e-8)

    mask = sitk.ReadImage(str(mask_path))
    old_mask_type = mask.GetPixelID()
    mask = sitk.GetArrayFromImage(mask)

    if mask.sum() < 10:
        print(f"Skip {mask_path.name} because of small mask")
        return
    if largest:
        mask = monai.transforms.utils.get_largest_connected_component_mask(mask).astype("int32")
    
    if cc_threshold > 0:
        mask = filter_connected_components(mask, cc_threshold)

    save_mask = True
    if method == "crop_bbox":
        new_img, new_mask = crop_bbox(img, mask)
    elif method in ["all_roi_resize", "big_roi_resize"]:
        new_img, new_mask = crop_resize_roi(img, mask, size, method)
    elif method == "crop_center":
        new_img, new_mask = crop_center(img, mask, size)
    elif method == "generate_breast_region":
        new_img = generate_breast_region(img, mask)
        new_mask = mask
        save_mask = False
    elif method == "crop_breast_region":
        new_img = generate_breast_region(img, mask)
        new_img, new_mask = crop_bbox(new_img, mask)
    else:
        raise ValueError(f"Invalid method: {method}")
    
    new_img = sitk.GetImageFromArray(new_img)
    new_img = sitk.Cast(new_img, old_img_type)
    sitk.WriteImage(new_img, str(dst_img_path))
    if save_mask:
        new_mask[new_mask > 0] = 1
        new_mask = sitk.GetImageFromArray(new_mask)
        new_mask = sitk.Cast(new_mask, sitk.sitkUInt8)
        sitk.WriteImage(new_mask, str(dst_mask_path))

def main(args):
    root = Path(args.root)
    img_dir_name = args.img_dir_name
    mask_dir_name = args.mask_dir_name
    methods = args.method.split(",")
    size = args.size

    kwargs_list = []
    for cx in sorted(os.listdir(root)):
        if args.cx_list is not None and cx not in args.cx_list:
            continue
        img_dir = root / cx / img_dir_name
        mask_dir = root / cx / mask_dir_name

        if not img_dir.exists() or not mask_dir.exists():
            continue

        img_path_list = sorted(list(img_dir.rglob("*.nii.gz")))
        mask_path_list = sorted(list(mask_dir.rglob("*.nii.gz")))
        assert len(img_path_list) == len(mask_path_list)

        for img_path, mask_path in zip(img_path_list, mask_path_list):
            # if "SY_d0_CX_11533113#R.nii.gz" not in img_path.name:
            #     continue
            assert img_path.name == mask_path.name

            for method in methods:
                if args.dst_img_dir_name is not None:
                    dst_img_dir_name = args.dst_img_dir_name
                else:
                    dst_img_dir_name = img_dir_name + "@" + method

                if args.largest:
                    dst_img_dir_name += "@largest"

                if args.cc_threshold > 0:
                    dst_img_dir_name += f"@cc{args.cc_threshold}"
                
                if args.norm:
                    dst_img_dir_name += "@norm"

                dst_img_path = root / cx / dst_img_dir_name / img_path.name
                dst_img_path.parent.mkdir(parents=True, exist_ok=True)

                if args.dst_mask_dir_name is not None:
                    dst_mask_dir_name = args.dst_mask_dir_name
                else:
                    dst_mask_dir_name = mask_dir_name + "@" + method

                if args.largest:
                    dst_mask_dir_name += "@largest"

                if args.cc_threshold > 0:
                    dst_mask_dir_name += f"@cc{args.cc_threshold}"

                if args.norm:
                    dst_mask_dir_name += "@norm"

                dst_mask_path = root / cx / dst_mask_dir_name / mask_path.name
                dst_mask_path.parent.mkdir(parents=True, exist_ok=True)

                kwargs = {
                    "img_path": img_path,
                    "mask_path": mask_path,
                    "method": method,
                    "dst_img_path": dst_img_path,
                    "dst_mask_path": dst_mask_path,
                    "size": size,
                    "largest": args.largest,
                    "cc_threshold": args.cc_threshold,
                    "norm": args.norm,
                    "__name__": img_path.name
                }
                kwargs_list.append(kwargs)
    
    # kwargs_list = kwargs_list[101:]
    worker(run_crop, kwargs_list, args.use_multiprocessing, desc="Cropping and resizing") 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--img_dir_name", type=str, required=True)
    parser.add_argument("--mask_dir_name", type=str, required=True)
    parser.add_argument("--method", type=str, help="crop methods, e.g. 'crop_1,crop_2'", required=True)
    parser.add_argument("--dst_img_dir_name", type=str, default=None)
    parser.add_argument("--dst_mask_dir_name", type=str, default=None)

    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("-m", "--use_multiprocessing", action="store_true")
    parser.add_argument("--largest", action="store_true", required=False)
    parser.add_argument("--norm", action="store_true", required=False)
    parser.add_argument("--cc_threshold", type=int, default=0)
    parser.add_argument("--cx_list", type=str, default=None)
    args = parser.parse_args()
    if args.cx_list is not None:
        args.cx_list = [x.strip() for x in args.cx_list.split(",")]

    main(args)