#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DataLoader semi-supervisé pour séries temporelles d’images satellites (4 bandes : RGB+NIR)
avec recalage appliqué via un CSV de keyframes et conversion des masques multi‑canaux (8 bandes binaires)
en masque d'indices de classes. La data augmentation est réalisée avec Albumentations,
appliquée de façon identique à toute la séquence.
À placer dans src/temporal_sat_dataset.py
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import rasterio
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from functools import lru_cache
from tqdm import tqdm
import albumentations as A

###############################################################################
# Fonctions de lecture avec mise en cache
###############################################################################
@lru_cache(maxsize=1024)
def read_tiff(path):
    """Lit le fichier TIFF et renvoie le tableau numpy."""
    with rasterio.open(path) as src:
        return src.read()

###############################################################################
# Data Augmentation avec Albumentations
###############################################################################
def albumentations_sequence_transform(list_img, list_mask=None):
    """
    Applique une transformation Albumentations identique à toutes les frames d'une séquence.
    Les images sont supposées au format [C, H, W] (ici 4 canaux). On les convertit en HWC,
    on applique la transformation, puis on retransforme en CHW.
    La même transformation (fixée par un seed commun) est appliquée à chaque frame.
    """
    transform = A.Compose([
         A.HorizontalFlip(p=0.5),
         A.VerticalFlip(p=0.5),
         A.RandomRotate90(p=0.5),
         A.RandomBrightnessContrast(p=0.5),
         A.RandomCrop(width=512, height=512, p=1.0),
    ], additional_targets={'mask':'mask'})
    
    seed = random.randint(0, int(1e6))
    augmented_imgs = []
    augmented_masks = [] if list_mask is not None else None
    
    for idx, img in enumerate(list_img):
        img_hwc = np.transpose(img, (1, 2, 0))
        mask = None
        if list_mask is not None and list_mask[idx] is not None:
            mask = list_mask[idx]
        augmented = transform(image=img_hwc, mask=mask, seed=seed)
        aug_img = np.transpose(augmented['image'], (2, 0, 1))
        augmented_imgs.append(aug_img)
        if list_mask is not None:
            augmented_masks.append(augmented['mask'])
            
    return augmented_imgs, augmented_masks

default_augmentation_fn = albumentations_sequence_transform

###############################################################################
# Fonction pour appliquer le recalage sur une image multi-canaux (CPU uniquement)
###############################################################################
def apply_shift_multi_channel(arr_img, shift_params):
    """
    Applique le décalage (shift_x, shift_y) sur une image multi-canaux.
    arr_img : np.array de shape [C, H, W]
    shift_params : tuple (shift_x, shift_y)
    Utilise le CPU pour éviter de ré-initialiser CUDA dans les workers.
    Retourne l'image recalée de même shape [C, H, W].
    """
    device = "cpu"
    t_img = torch.tensor(arr_img, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
    B, C, H, W = t_img.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    shift_x_norm = 2.0 * shift_params[0] / max(W - 1, 1.0)
    shift_y_norm = 2.0 * shift_params[1] / max(H - 1, 1.0)
    grid_x_norm = (2.0 * grid_x / (W - 1)) - 1.0 - shift_x_norm
    grid_y_norm = (2.0 * grid_y / (H - 1)) - 1.0 - shift_y_norm
    grid = torch.stack((grid_x_norm, grid_y_norm), dim=-1).unsqueeze(0)
    warped = F.grid_sample(t_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    warped_np = (warped.squeeze(0) * 255.0).detach().cpu().numpy()
    return warped_np

###############################################################################
# CLASSE TemporalSatDataset
###############################################################################
class TemporalSatDataset(Dataset):
    """
    Dataset pour séries temporelles d’images satellites avec recalage.
    
    - Images en 4 bandes (RGB+NIR).
    - Masques en 8 bandes binaires (une bande par classe) avec pixels à 0 ou 255.
      La conversion en masque d'indices de classes se fait ainsi :
         * Convertir chaque bande en booléen (True si pixel == 255).
         * Calculer le nombre de classes actives par pixel.
         * Calculer la fréquence globale pour chaque classe dans l'image.
         * Pour un pixel avec plusieurs classes actives, choisir la classe rare (celle dont le nombre total de pixels actifs est le plus faible).
         * Pour les pixels sans aucune classe active, assigner 255.
    - Regroupement par "key" (colonne du CSV) et tri par date.
    - Recalage appliqué via un CSV d'alignement (déjà généré).
    - Sélection intelligente du sous-intervalle temporel : on choisit celui avec le maximum d'annotations.
    """
    def __init__(self,
                 csv_path,
                 alignment_csv,
                 transform_fn=default_augmentation_fn,
                 seq_length=None,
                 random_subseq=True):
        """
        Args:
          csv_path      : Chemin vers df_merged.csv (doit contenir une colonne "key").
          alignment_csv : Chemin vers le CSV de recalage.
          transform_fn  : Fonction de data augmentation (pour la séquence entière).
          seq_length    : Nombre de frames à retourner (sinon toute la séquence).
          random_subseq : Si True, sélectionne un sous-intervalle avec le maximum d'annotations.
        """
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['planet_path'].apply(os.path.exists)].reset_index(drop=True)
        for col in ["key", "date", "planet_path"]:
            if col not in self.df.columns:
                raise ValueError(f"Le CSV doit contenir la colonne '{col}'.")
        if "labels_path" not in self.df.columns:
            self.df["labels_path"] = None

        self.df["date_dt"] = pd.to_datetime(self.df["date"], format="%Y-%m-%d")
        
        # Groupement par "key" et tri par date
        self.grouped_key = []
        for key_val, group in self.df.groupby("key"):
            group_sorted = group.sort_values("date_dt")
            seq = []
            for idx, row in group_sorted.iterrows():
                seq.append({
                    "planet_path": row["planet_path"],
                    "labels_path": row["labels_path"] if pd.notna(row["labels_path"]) else None,
                    "date": row["date_dt"]
                })
            self.grouped_key.append({
                "key": key_val,
                "sequence": seq
            })
        
        self.transform_fn = transform_fn
        self.seq_length = seq_length
        self.random_subseq = random_subseq
        
        # Charger le CSV d'alignement et construire un dictionnaire
        self.alignment_dict = {}
        df_align = pd.read_csv(alignment_csv)
        for idx, row in df_align.iterrows():
            align_key = (row["zone_path"], row["image_name"])
            self.alignment_dict[align_key] = (row["shift_x"], row["shift_y"])

    def __len__(self):
        return len(self.grouped_key)

    def select_subsequence(self, seq):
        """
        Sélection intelligente d'un sous-intervalle de longueur seq_length
        en choisissant celui avec le maximum de frames annotées.
        Si aucun intervalle ne contient d'annotations, renvoie un intervalle aléatoire.
        """
        T_all = len(seq)
        if self.seq_length is None or self.seq_length >= T_all:
            return seq
        best_interval = None
        best_count = -1
        for start in range(0, T_all - self.seq_length + 1):
            interval = seq[start:start+self.seq_length]
            count = sum(1 for item in interval if item["labels_path"] is not None and os.path.exists(item["labels_path"]))
            if count > best_count:
                best_count = count
                best_interval = interval
        if best_interval is not None and best_count > 0:
            return best_interval
        else:
            start = random.randint(0, T_all - self.seq_length)
            return seq[start:start+self.seq_length]

    def __getitem__(self, index):
        """
        Renvoie un dictionnaire contenant :
          "X": tenseur [T, 4, H, W] recalé et transformé,
          "Y": tenseur [T, H, W] (masque d'indices de classe) où
               - -1 signifie pas d'annotation,
               - 255 signifie pixel sans aucune classe active,
               - sinon l'indice de la classe choisie (en cas de conflit, la classe rare est choisie).
          "mask_superv": booléen [T] indiquant si la frame est annotée,
          "key": identifiant de la zone.
        """
        item = self.grouped_key[index]
        key_val = item["key"]
        seq_meta = item["sequence"]
        
        if self.random_subseq and self.seq_length is not None:
            chosen_meta = self.select_subsequence(seq_meta)
        else:
            chosen_meta = seq_meta
        T = len(chosen_meta)
        
        list_img = []
        list_mask = []
        for info in chosen_meta:
            planet_path = info["planet_path"]
            labels_path = info["labels_path"]
            arr = read_tiff(planet_path).astype(np.float32)  # [4, H, W]
            base_name = os.path.basename(planet_path)
            zone_path = os.path.dirname(planet_path)
            align_key = (zone_path, base_name)
            if align_key in self.alignment_dict:
                shift_params = self.alignment_dict[align_key]
                arr = apply_shift_multi_channel(arr, shift_params)
            list_img.append(arr)
            
            if labels_path is not None and os.path.exists(labels_path):
                try:
                    label_8bands = read_tiff(labels_path)  # [8, H, W] avec 0 ou 255
                except Exception:
                    mask_data = None
                else:
                    bin_mask = (label_8bands == 255)
                    count_per_pixel = bin_mask.sum(axis=0)
                    global_counts = bin_mask.sum(axis=(1,2)).astype(np.float32) + 1e-6
                    weights = bin_mask.astype(np.float32) / global_counts[:, None, None]
                    class_indices = np.argmax(weights, axis=0)
                    no_class = (count_per_pixel == 0)
                    class_indices[no_class] = 255
                    mask_data = class_indices.astype(np.uint8)
            else:
                mask_data = None
            list_mask.append(mask_data)
            
        if self.transform_fn is not None:
            list_img, list_mask = self.transform_fn(list_img, list_mask)
            
        X_list = []
        Y_list = []
        for i in range(T):
            x_t = torch.from_numpy(list_img[i])
            X_list.append(x_t)
            if list_mask[i] is not None:
                y_t = torch.from_numpy(list_mask[i])
                Y_list.append(y_t)
            else:
                Y_list.append(None)
        X = torch.stack(X_list, dim=0)  # [T, 4, H, W]
        T, _, H, W = X.shape
        Y = torch.full((T, H, W), fill_value=-1, dtype=torch.long)
        mask_superv = torch.zeros((T,), dtype=torch.bool)
        for t in range(T):
            if Y_list[t] is not None:
                Y[t] = Y_list[t]
                mask_superv[t] = True
        sample = {"X": X, "Y": Y, "mask_superv": mask_superv, "key": key_val}

        # Normalize for ResNet50 (Planet‑specific mean/std)
        mean = torch.tensor([1042.59,  915.62,  671.26, 2605.21], dtype=torch.float32, device=X.device)
        std  = torch.tensor([ 957.96,  715.55,  596.94, 1059.90], dtype=torch.float32, device=X.device)
        # X shape = [T, 4, H, W] → normalize channel‑wise
        X = (X - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
        sample["X"] = X
        # -----------------------------------------
        return sample
