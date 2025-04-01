#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optimized data loader for daily satellite time series (RGB+NIR),
including optional alignment shifts and multi-band label conversion.
Includes semi-supervised aspects (mask_superv).
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import rasterio
from datetime import datetime
from torch.utils.data import Dataset
import torch.nn.functional as F
from functools import lru_cache
import albumentations as A

###############################################################################
# 1) Caching TIF file I/O
###############################################################################
@lru_cache(maxsize=4096)
def read_tiff(path):
    """
    Cached read of a TIFF file. 
    Returns a NumPy array with shape (bands, height, width).
    """
    with rasterio.open(path) as src:
        return src.read()

###############################################################################
# 2) Safer random cropping
###############################################################################
def safe_random_crop(
    image, mask, crop_w, crop_h, min_annot_ratio=0.05, max_attempts=5, rng=None
):
    """
    Tries multiple random crops so that at least min_annot_ratio
    fraction of the crop is annotated (mask != 255). If no attempt works,
    returns the last attempted crop.
    """
    if rng is None:
        rng = np.random  
    H, W, _ = image.shape
    last_crop_img, last_crop_mask = None, None
    for _ in range(max_attempts):
        # random x,y
        x = rng.randint(0, max(1, W - crop_w + 1)) if (W > crop_w) else 0
        y = rng.randint(0, max(1, H - crop_h + 1)) if (H > crop_h) else 0
        crop_img = image[y : y + crop_h, x : x + crop_w, :]
        crop_mask = mask[y : y + crop_h, x : x + crop_w]
        ratio = np.mean(crop_mask != 255)
        if ratio >= min_annot_ratio:
            return crop_img, crop_mask
        last_crop_img, last_crop_mask = crop_img, crop_mask
    return last_crop_img, last_crop_mask

###############################################################################
# 3) Data augmentation pipelines with Albumentations
###############################################################################
def safe_sequence_transform(
    list_img, list_mask=None, crop_w=512, crop_h=512, min_annot_ratio=0.05, rng=None
):
    """
    Applies the same Albumentations transforms to each frame in the sequence,
    then a "safe" random crop that ensures enough labeled pixels.
    """
    if rng is None:
        rng = np.random  

    aug = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ],
        additional_targets={"mask": "mask"},
    )

    out_imgs, out_masks = [], [] if list_mask is not None else None
    for i in range(len(list_img)):
        img_chw = list_img[i]
        # shape: (C, H, W) -> Albumentations (H, W, C)
        img_hwc = np.transpose(img_chw, (1, 2, 0))
        m = list_mask[i] if list_mask is not None else None

        # apply transform
        augmented = aug(image=img_hwc, mask=m)
        img_tf = augmented["image"]
        mask_tf = augmented["mask"] if m is not None else None

        # safe random crop on the mask
        if mask_tf is not None:
            img_c, mask_c = safe_random_crop(
                img_tf,
                mask_tf,
                crop_w,
                crop_h,
                min_annot_ratio,
                rng=rng,
            )
        else:
            center_aug = A.CenterCrop(height=crop_h, width=crop_w, p=1.0)
            _res = center_aug(image=img_tf)
            img_c = _res["image"]
            mask_c = None

        # revert (H,W,C) -> (C,H,W)
        out_imgs.append(np.transpose(img_c, (2, 0, 1)))
        if list_mask is not None:
            out_masks.append(mask_c)

    return out_imgs, out_masks

# transform for validation
def validation_transform(list_img, list_mask=None, size=512):
    """
    Center-crop transform for validation. No randomness, same shape on all frames.
    """
    center_aug = A.Compose(
        [A.CenterCrop(height=size, width=size, p=1.0)],
        additional_targets={"mask": "mask"},
    )
    out_imgs, out_masks = [], [] if list_mask is not None else None

    for i in range(len(list_img)):
        img_chw = list_img[i]
        img_hwc = np.transpose(img_chw, (1, 2, 0))
        m = list_mask[i] if list_mask is not None else None

        if m is not None:
            m = np.ascontiguousarray(m)
        res = center_aug(image=img_hwc, mask=m)
        img_c = np.transpose(res["image"], (2, 0, 1))
        out_imgs.append(img_c)
        if list_mask is not None:
            out_masks.append(res["mask"])

    return out_imgs, out_masks

validation_fn = validation_transform
default_augmentation_fn = safe_sequence_transform

###############################################################################
# 4) Shifting images/masks in CPU with grid_sample
###############################################################################
import torch.nn.functional as F

def apply_shift_multi_channel(arr_img, shift_xy):
    """
    arr_img shape: (C, H, W). shift_xy is (shiftX, shiftY) in pixels.
    We treat shift as a pure translation, using bilinear grid_sample.
    """
    if shift_xy is None:
        return arr_img 
    device = "cpu"
    C, H, W = arr_img.shape
    # to [1,C,H,W]
    t = torch.tensor(arr_img, dtype=torch.float32, device=device).unsqueeze(0) / 255.0

    # build sampling grid
    shift_x, shift_y = shift_xy
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )

    shift_x_norm = 2.0 * shift_x / max(W - 1, 1)
    shift_y_norm = 2.0 * shift_y / max(H - 1, 1)

    grid_x_norm = (2.0 * grid_x / (W - 1)) - 1.0 - shift_x_norm
    grid_y_norm = (2.0 * grid_y / (H - 1)) - 1.0 - shift_y_norm

    sampling_grid = torch.stack((grid_x_norm, grid_y_norm), dim=-1).unsqueeze(0)
    warped = F.grid_sample(
        t, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    out = (warped.squeeze(0) * 255.0).cpu().numpy()
    return out

def apply_shift_mask(mask, shift_xy):
    """
    mask shape: (H,W). shift_xy is (shiftX, shiftY). nearest-neighbor sampling.
    """
    if shift_xy is None:
        return mask
    device = "cpu"
    H, W = mask.shape
    # shape: [1,1,H,W]
    t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    shift_x, shift_y = shift_xy
    shift_x_norm = 2.0 * shift_x / max(W - 1, 1)
    shift_y_norm = 2.0 * shift_y / max(H - 1, 1)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij',
    )
    grid_x_norm = (2.0 * grid_x / (W - 1)) - 1.0 - shift_x_norm
    grid_y_norm = (2.0 * grid_y / (H - 1)) - 1.0 - shift_y_norm
    grid = torch.stack((grid_x_norm, grid_y_norm), dim=-1).unsqueeze(0)

    warped = F.grid_sample(t, grid, mode="nearest", padding_mode="zeros", align_corners=True)
    out = warped.squeeze().cpu().numpy().astype(np.uint8)
    return out

###############################################################################
# 5) The main Dataset
###############################################################################
class TemporalSatDataset(Dataset):
    """
    Dataset for time-series satellite images (4 bands) plus multi-channel (8-band)
    label TIFF that we convert to a single-class-per-pixel segmentation mask (0..7, with 255=ignore).
    The 'split' argument is just used to filter subsets by key in a simple random manner.
    """

    def __init__(
        self,
        csv_path,
        alignment_csv,
        transform_fn=None,
        seq_length=None,
        random_subseq=True,
        split="train",  # 'train','val','test'
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        split_seed=42,
    ):
        super().__init__()
        self.transform_fn = transform_fn
        self.seq_length = seq_length
        self.random_subseq = random_subseq

        
        df = pd.read_csv(csv_path)
        # Filter 
        df = df[df["planet_path"].apply(os.path.exists)].reset_index(drop=True)
        if "labels_path" not in df.columns:
            df["labels_path"] = None

        # Basic checks
        for col in ("key", "date", "planet_path"):
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in CSV")

        # Shuffle
        unique_keys = df["key"].unique()
        rng = np.random.default_rng(split_seed)
        rng.shuffle(unique_keys)
        n = len(unique_keys)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        if split == "train":
            chosen_keys = unique_keys[:n_train]
        elif split == "val":
            chosen_keys = unique_keys[n_train : n_train + n_val]
        elif split == "test":
            chosen_keys = unique_keys[n_train + n_val :]
        else:
            raise ValueError("split must be 'train','val' or 'test'")

        df = df[df["key"].isin(chosen_keys)].copy()

        df["date_dt"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

 
        self.samples_by_key = []
        for key_val, g in df.groupby("key"):
            g = g.sort_values("date_dt")
            seq_list = []
            for _, row in g.iterrows():
                seq_list.append(
                    {
                        "planet_path": row["planet_path"],
                        "labels_path": row["labels_path"] if pd.notna(row["labels_path"]) else None,
                        "date": row["date_dt"],
                    }
                )
            self.samples_by_key.append({"key": key_val, "sequence": seq_list})

        #  alignment CSV 
        self.alignment_dict = {}
        align_df = pd.read_csv(alignment_csv)
        for _, rw in align_df.iterrows():
            zpath = rw["zone_path"]
            fname = rw["image_name"]
            shiftxy = (rw["shift_x"], rw["shift_y"])
            self.alignment_dict[(zpath, fname)] = shiftxy

    def __len__(self):
        return len(self.samples_by_key)

    def _pick_subsequence(self, sequence):
        """
        Given the full sorted sequence for one key, either take the entire thing
        or choose a sub-range of length self.seq_length that ensures at least 1 annotated frame.
        """
        T_all = len(sequence)
        if self.seq_length is None or self.seq_length >= T_all:
         
            if not any(s["labels_path"] is not None and os.path.exists(s["labels_path"]) for s in sequence):
                raise ValueError("No annotated frames in entire sequence.")
            return sequence

        # find indices that have annotation
        ann_idx = [
            i for i, s in enumerate(sequence)
            if s["labels_path"] is not None and os.path.exists(s["labels_path"])
        ]
        if not ann_idx:
            raise ValueError("No annotated frames found in that sequence.")


        chosen_i = random.choice(ann_idx)
        start_min = max(0, chosen_i - self.seq_length + 1)
        start_max = min(chosen_i, T_all - self.seq_length)
        if start_min > start_max:
            start_idx = 0
        else:
            start_idx = random.randint(start_min, start_max)
        return sequence[start_idx : start_idx + self.seq_length]

    def __getitem__(self, idx):
        item = self.samples_by_key[idx]
        seq_list = item["sequence"]
        # choose frames
        if self.seq_length is not None:
            if self.random_subseq:
                chosen_seq = self._pick_subsequence(seq_list)
            else:
                n = len(seq_list)
                start_i = max(0, (n - self.seq_length) // 2)
                chosen_seq = seq_list[start_i : start_i + self.seq_length]
        else:
            chosen_seq = seq_list

        T = len(chosen_seq)
        # read images & masks
        images = []
        masks = []
        for info in chosen_seq:
            planet_p = info["planet_path"]
            
            arr = read_tiff(planet_p).astype(np.float32)  # shape (bands,H,W)
            base_name = os.path.basename(planet_p)
            zone_path = os.path.dirname(planet_p)

            shift = self.alignment_dict.get((zone_path, base_name), None)
            # apply shift if present
            if shift is not None:
                # needs shape (C,H,W)
                arr = apply_shift_multi_channel(arr, shift)

            images.append(arr)

  
            labels_p = info["labels_path"]
            if labels_p is not None and os.path.exists(labels_p):
                lab_8b = read_tiff(labels_p)  # shape (8,H,W)
                # convert to single-class
                # each band is 0 or 255
                bin_mask = (lab_8b >= 1)  # True/False
                count_per_px = bin_mask.sum(axis=0)
                # compute weight
                global_counts = bin_mask.sum(axis=(1, 2)).astype(np.float32) + 1e-6
                weights = bin_mask.astype(np.float32) / global_counts[:, None, None]
                class_map = np.argmax(weights, axis=0)  # shape(H,W)
            
                class_map[count_per_px == 0] = 255

                if shift is not None:
                    class_map = apply_shift_mask(class_map, shift)
                masks.append(class_map)
            else:
                
                masks.append(None)

        
        if self.transform_fn is not None:
            images_out, masks_out = self.transform_fn(images, masks)
        else:
            images_out, masks_out = images, masks

    
        # shape => [T,4,H,W] float32, and Y => [T,H,W] long
        X_list, Y_list = [], []
        for i in range(T):
  
            x_t = torch.from_numpy(images_out[i])
            X_list.append(x_t)
            if masks_out[i] is not None:
                y_t = torch.from_numpy(masks_out[i])
            else:
                y_t = None
            Y_list.append(y_t)

        X = torch.stack(X_list, dim=0)  # shape [T, 4, H, W]


        T, _, H, W = X.shape
        Y = torch.full((T, H, W), 255, dtype=torch.long)
        mask_superv = torch.zeros((T,), dtype=torch.bool)
        for t in range(T):
            if Y_list[t] is not None:
                Y[t] = Y_list[t]
                mask_superv[t] = True

       
        device_cpu = torch.device("cpu")
        means = torch.tensor([1042.59, 915.62, 671.26, 2605.21], device=device_cpu, dtype=torch.float32)
        stds  = torch.tensor([957.96, 715.55, 596.94, 1059.90], device=device_cpu, dtype=torch.float32)
        # (T,4,H,W)
        X = (X - means.view(1, -1, 1, 1)) / stds.view(1, -1, 1, 1)

        return {
            "X": X,                  # [T,4,H,W], float
            "Y": Y,                  # [T,H,W], long
            "mask_superv": mask_superv,  # [T], bool
            "key": item["key"],      # str
        }
