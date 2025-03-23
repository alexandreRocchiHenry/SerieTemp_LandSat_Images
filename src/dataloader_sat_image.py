#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exemple de DataLoader semi-supervisé pour séries temporelles satellites (8 bandes).
A placer dans src/temporal_sat_dataset.py (par ex.).
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

#####################
# 1) Transforms pour la cohérence temporelle
#####################

def random_horizontal_flip_sequence(list_img, list_mask=None):
    """Applique un flip horizontal identique à toutes les frames."""
    if random.random() < 0.5:
        # list_img[i] shape = [8, H, W]
        new_imgs = [im[:, :, ::-1].copy() for im in list_img]
        if list_mask is not None:
            new_masks = []
            for m in list_mask:
                if m is not None:
                    # m shape = [H, W]
                    new_masks.append(m[:, ::-1].copy())
                else:
                    new_masks.append(None)
            return new_imgs, new_masks
        else:
            return new_imgs, None
    return list_img, list_mask


def random_vertical_flip_sequence(list_img, list_mask=None):
    """Applique un flip vertical identique à toutes les frames."""
    if random.random() < 0.5:
        new_imgs = [im[:, ::-1, :].copy() for im in list_img]
        if list_mask is not None:
            new_masks = []
            for m in list_mask:
                if m is not None:
                    new_masks.append(m[::-1, :].copy())
                else:
                    new_masks.append(None)
            return new_imgs, new_masks
        else:
            return new_imgs, None
    return list_img, list_mask


def random_crop_sequence(list_img, list_mask=None, crop_size=(512, 512)):
    """Exemple de recadrage identique sur toutes les frames."""
    _, H, W = list_img[0].shape  #  (8, H, W) => (H, W) in indices [1,2]
    ch, cw = crop_size

    if (H <= ch) or (W <= cw):
        # Pas de recadrage possible => on retourne tel quel
        return list_img, list_mask

    # Coordonnées de la zone croppée
    y0 = random.randint(0, H - ch)
    x0 = random.randint(0, W - cw)

    new_imgs = []
    new_masks = []
    for i, im in enumerate(list_img):
        # im shape = [8, H, W]
        im_cropped = im[:, y0:y0+ch, x0:x0+cw]
        new_imgs.append(im_cropped)
        if list_mask is not None:
            m = list_mask[i]
            if m is not None:
                m_cropped = m[y0:y0+ch, x0:x0+cw]
            else:
                m_cropped = None
            new_masks.append(m_cropped)
    return new_imgs, new_masks


def default_augmentation_fn(list_img, list_mask=None):
    """
    Exemple de "transform" combinant flips + crop.
    """
    list_img, list_mask = random_horizontal_flip_sequence(list_img, list_mask)
    list_img, list_mask = random_vertical_flip_sequence(list_img, list_mask)
    list_img, list_mask = random_crop_sequence(list_img, list_mask, crop_size=(512, 512))
    return list_img, list_mask


#####################
# 2) Dataset
#####################
class TemporalSatDataset(Dataset):
    """
    Dataset pour séries temporelles quotidiennes (8 bandes).
    On part d'un df_merged.csv indiquant:
        - AOI ou identifiant de zone
        - Chemin de l'image .tif (8 bandes)
        - Chemin du masque (ou None)
        - Date (YYYY-MM-DD)
      (Les noms de colonnes doivent être adaptés.)
    """

    def __init__(self,
                 csv_path,
                 transform_fn=None,
                 seq_length=None,
                 random_subseq=True):
        """
        Args:
          csv_path       : Chemin vers df_merged.csv
          transform_fn   : Fonction de data augmentation (sequence-level)
          seq_length     : Longueur de la séquence à retourner
          random_subseq  : si True, on pioche un sous-intervalle aléatoire
                           dans la suite temporelle pour ne pas tout charger
        """
        super().__init__()
        self.csv = pd.read_csv(csv_path)
        # On suppose qu'il y a au moins les colonnes :
        #   "aoi" (ou "zone_id")
        #   "date" (str "YYYY-MM-DD")
        #   "img_path" (chemin vers .tif)
        #   "mask_path" (ou None / NaN si pas d'annotation)
        # Adaptez ici si vos noms de colonnes sont différents:
        if "aoi" not in self.csv.columns:
            raise ValueError("La CSV doit contenir une colonne 'aoi' (zone).")
        if "date" not in self.csv.columns:
            raise ValueError("La CSV doit contenir une colonne 'date'.")
        if "img_path" not in self.csv.columns:
            raise ValueError("La CSV doit contenir une colonne 'img_path' pour l'image.")
        if "mask_path" not in self.csv.columns:
            # On peut quand même continuer, mais la partie masques ne fonctionnera pas
            self.csv["mask_path"] = None

        # Convertit la colonne "date" en datetime pour tri
        self.csv["date_dt"] = pd.to_datetime(self.csv["date"], format="%Y-%m-%d")

        # On regroupe par aoi, et on stocke la suite par ordre chronologique
        self.grouped_aoi = []
        for aoi_name, group in self.csv.groupby("aoi"):
            # tri par date
            group_sorted = group.sort_values("date_dt")
            # liste de (img_path, mask_path, date_dt)
            seq = []
            for idx, row in group_sorted.iterrows():
                seq.append({
                    "img_path": row["img_path"],
                    "mask_path": row["mask_path"] if pd.notna(row["mask_path"]) else None,
                    "date": row["date_dt"]
                })
            self.grouped_aoi.append({
                "aoi_name": aoi_name,
                "sequence": seq
            })

        self.transform_fn = transform_fn
        self.seq_length = seq_length
        self.random_subseq = random_subseq

    def __len__(self):
        # Chaque entrée du dataset correspond à 1 AOI
        return len(self.grouped_aoi)

    def __getitem__(self, index):
        """
        Retourne un dictionnaire:
            "X": [T, 8, H, W]
            "Y": [T, H, W] avec -1 pour frames non-annotées
            "mask_superv": bool [T], True si frame t est annotée
            "aoi_name": str
        """
        item = self.grouped_aoi[index]
        aoi_name = item["aoi_name"]
        seq_meta = item["sequence"]  # liste d'infos sur la suite

        T_all = len(seq_meta)
        if T_all == 0:
            # AOI sans données, peu probable mais on gère
            return {
                "X": torch.empty(0),
                "Y": torch.empty(0),
                "mask_superv": torch.empty(0),
                "aoi_name": aoi_name
            }

        # Détermine la plage temporelle à charger
        if self.seq_length is not None and self.seq_length < T_all and self.random_subseq:
            max_start = T_all - self.seq_length
            start_id = random.randint(0, max_start)
            end_id = start_id + self.seq_length
        else:
            # On prend tout
            start_id = 0
            end_id = T_all

        chosen_meta = seq_meta[start_id:end_id]
        T = len(chosen_meta)

        # Chargement des images + masques
        list_img = []
        list_mask = []
        for info in chosen_meta:
            img_path = info["img_path"]
            mask_path = info["mask_path"]

            # Lecture .tif (8 bandes) => shape [8, H, W]
            with rasterio.open(img_path) as src:
                # on attend 8 canaux
                arr = src.read()  # shape (bands, H, W)
                # cast en float32
                arr = arr.astype(np.float32)

            # Lecture du masque (ou None)
            if mask_path is not None and os.path.exists(mask_path):
                with rasterio.open(mask_path) as msrc:
                    mask_data = msrc.read(1)  # on prend la 1ère bande
                    mask_data = mask_data.astype(np.int64)
            else:
                mask_data = None

            list_img.append(arr)
            list_mask.append(mask_data)

        # data augmentation (identique sur toute la séquence)
        if self.transform_fn is not None:
            list_img, list_mask = self.transform_fn(list_img, list_mask)

        # Conversion en tensors
        X_list = []
        Y_list = []
        for i in range(T):
            # arr shape [8, H, W]
            x_ten = torch.from_numpy(list_img[i])  # => [8, H, W]
            if list_mask[i] is None:
                # pas d'annotation => on mettra -1 plus tard
                Y_list.append(None)
            else:
                # shape [H, W]
                y_ten = torch.from_numpy(list_mask[i])
                Y_list.append(y_ten)
            X_list.append(x_ten)

        # Empile en [T, 8, H, W]
        X = torch.stack(X_list, dim=0)

        # Construit Y en [T, H, W], -1 si None
        # On suppose que toutes les frames ont la même taille H,W (après transform)
        T, _, H, W = X.shape
        Y = torch.full((T, H, W), fill_value=-1, dtype=torch.long)
        mask_superv = torch.zeros((T,), dtype=torch.bool)
        for t in range(T):
            if Y_list[t] is not None:
                Y[t] = Y_list[t]
                mask_superv[t] = True  # frame annotée

        sample = {
            "X": X,                # [T, 8, H, W]
            "Y": Y,                # [T, H, W], -1 => pas d'annotation
            "mask_superv": mask_superv,   # [T] booleen
            "aoi_name": aoi_name
        }
        return sample


#####################
# 3) Exemple d'utilisation
#####################
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Exemple minimal : charger le dataset
    csv_path = "df_merged.csv"  # adapter chemin
    dataset = TemporalSatDataset(
        csv_path=csv_path,
        transform_fn=default_augmentation_fn,
        seq_length=16,
        random_subseq=True
    )

    # DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    for batch_idx, batch_data in enumerate(loader):
        X = batch_data["X"]  # shape [B, T, 8, H, W]
        Y = batch_data["Y"]  # shape [B, T, H, W], -1 => non-annoté
        sup_mask = batch_data["mask_superv"]  # shape [B, T]
        print(f"Batch {batch_idx} => X={X.shape}, Y={Y.shape}, sup_mask={sup_mask.shape}")
        # ... ici on peut faire l'entraînement ...
        break
