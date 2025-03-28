import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import shutil

df = pd.read_csv("table.csv")

ds_dir = Path(".")
for idx, row in df.iterrows():
    c0_img = sitk.Cast(sitk.ReadImage(ds_dir / "C0" / "image" / row["filename"]), sitk.sitkFloat32)
    c2_img = sitk.Cast(sitk.ReadImage(ds_dir / "C2" / "image" / row["filename"]), sitk.sitkFloat32)
    c5_img = sitk.Cast(sitk.ReadImage(ds_dir / "C5" / "image" / row["filename"]), sitk.sitkFloat32)

    c2_c0_img = sitk.SubtractImageFilter().Execute(c2_img, c0_img)
    c5_c2_img = sitk.SubtractImageFilter().Execute(c5_img, c2_img)
    dst_c2_c0_img = ds_dir / "C2-C0" / "image" / row["filename"]
    dst_c5_c2_img = ds_dir / "C5-C2" / "image" / row["filename"]
    dst_c2_c0_img.parent.mkdir(parents=True, exist_ok=True)
    dst_c5_c2_img.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(c2_c0_img, str(dst_c2_c0_img))
    sitk.WriteImage(c5_c2_img, str(dst_c5_c2_img))

    (ds_dir / "C2-C0" / "tumor_mask").mkdir(parents=True, exist_ok=True)
    (ds_dir / "C5-C2" / "tumor_mask").mkdir(parents=True, exist_ok=True)

    shutil.copy2(ds_dir / "C2" / "tumor_mask" / row["filename"], ds_dir / "C2-C0" / "tumor_mask" / row["filename"])
    shutil.copy2(ds_dir / "C5" / "tumor_mask" / row["filename"], ds_dir / "C5-C2" / "tumor_mask" / row["filename"])

import subprocess
cmd = "python roi.py --root . --img_dir_name image --mask_dir_name tumor_mask --method all_roi_resize --cx_list C2,C2-C0,C5-C2 --cc_threshold 15"
subprocess.run(cmd, shell=True)

new_df = []
for idx, row in df.iterrows():
    row["mask"] = f"C2/tumor_mask@all_roi_resize@cc15/{row['filename']}"
    row["C2"] = f"C2/image@all_roi_resize@cc15/{row['filename']}"
    row["C2-C0"] = f"C2-C0/image@all_roi_resize@cc15/{row['filename']}"
    row["C5-C2"] = f"C5-C2/image@all_roi_resize@cc15/{row['filename']}"
    new_df.append(row)

new_df = pd.DataFrame(new_df)
new_df.to_csv("cls_demo.csv", index=False)