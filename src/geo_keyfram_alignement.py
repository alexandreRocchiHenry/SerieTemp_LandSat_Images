import os
import cv2
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
import torch
from datetime import datetime
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling


class GeoKeyFrameAlign:
    """
    Combinaison de géoréférenciation (via Rasterio) et d'une approche Key-frame
    mensuelle pour aligner les images.
    """

    def __init__(self, dst_crs="EPSG:4326", res=None):
        """
        Args:
            dst_crs (str): Code EPSG cible pour reprojection (par défaut EPSG:4326).
            res (tuple ou None): Résolution souhaitée (ex: (0.0001, 0.0001)) 
                                 ou None pour laisser rasterio calculer.
        """
        self.dst_crs = dst_crs
        self.res = res

    # -------------------------------------------------------------------------
    # 1. Lecture et reprojection Rasterio
    # -------------------------------------------------------------------------
    def load_and_reproject(self, path):
        """
        Ouvre un GeoTIFF avec rasterio, reprojette (si nécessaire)
        vers self.dst_crs et retourne un array numpy (bands, height, width).

        Args:
            path (str): Chemin du fichier GeoTIFF.

        Returns:
            np.ndarray: L'image reprojetée sous forme (H, W) ou (C, H, W).
                        Par simplicité, on peut limiter à 1 ou 3 canaux.
        """
        with rasterio.open(path) as src:
            # Si la projection source est déjà la même que dst_crs,
            # on peut éviter la reprojection.
            if src.crs == self.dst_crs and not self.res:
                # Lecture brute
                data = src.read()
                # data shape = (nb_bandes, height, width)
                return data
            else:
                # On prépare la transformation
                transform, width, height = rasterio.warp.calculate_default_transform(
                    src.crs, self.dst_crs, src.width, src.height, *src.bounds, resolution=self.res
                )

                # Profil pour la reprojection
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': self.dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    # On garde le nb de canaux
                })

                # On crée un tableau pour accueillir les données reprojetées
                dest = np.zeros((src.count, height, width), dtype=src.dtypes[0])

                for i in range(1, src.count + 1):
                    rasterio.warp.reproject(
                        source=rasterio.band(src, i),
                        destination=dest[i-1],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=self.dst_crs,
                        resampling=Resampling.bilinear
                    )
                return dest

    # -------------------------------------------------------------------------
    # 2. Prétraitement d'image pour l'alignement SIFT (OpenCV)
    # -------------------------------------------------------------------------
    def convert_image_type(self, image):
        """
        Convertit le type en 'uint8' si nécessaire (normalisation).
        On suppose image en float ou autre.
        """
        if image.dtype != 'uint8':
            # ATTENTION : data max peut être très élevé si 16 bits ou float
            max_val = image.max() if image.max() != 0 else 1
            image = cv2.convertScaleAbs(image, alpha=(255.0 / max_val))
        return image

    def to_grayscale(self, data):
        """
        data : (C, H, W) ou (H, W)
        - Si 1 canal, on la laisse telle quelle.
        - Si 3 canaux, on convertit en gris.
        """
        if len(data.shape) == 3:
            C, H, W = data.shape
            if C == 1:
                # Rien à faire
                return data[0, :, :]
            elif C >= 3:
                # On prend seulement 3 canaux pour OpenCV
                # ou on calcule la moyenne comme grayscale
                # data[:3] => (3, H, W)
                rgb = np.transpose(data[:3], (1,2,0))  # => (H, W, 3)
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                return gray
        else:
            # Déjà 2D
            return data

    def preprocess(self, data):
        """
        Enchaîne la reprojection en np.uint8 + grayscale.
        data : (C, H, W) => (H, W)
        """
        data = self.convert_image_type(data)
        # convert_image_type renvoie un (C, H, W) si C>1 => besoin de reshape ?
        # On standardise : transformons data en (H, W) si possible.
        if len(data.shape) == 3:
            # => (C, H, W). On prend seulement la 1ère bande ?
            # Mieux: on passera to_grayscale pour 3 canaux
            pass
        data = self.to_grayscale(data)
        return data

    # -------------------------------------------------------------------------
    # 3. Détection et Matching SIFT
    # -------------------------------------------------------------------------
    def detect_keypoints(self, image):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors

    def match_keypoints(self, desc1, desc2, ratio=0.75):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)

        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        return good

    # -------------------------------------------------------------------------
    # 4. Calcul du décalage (shift) ou homographie
    # -------------------------------------------------------------------------
    def calculate_shift(self, arr_ref, arr_img, ratio=0.75):
        """
        Calcul d'un simple décalage en pixels (moyen et max),
        en se basant sur SIFT. On suppose que la transformation est proche d'une translation.
        """
        # Prétraitement
        ref = self.preprocess(arr_ref)
        img = self.preprocess(arr_img)

        kp1, desc1 = self.detect_keypoints(ref)
        kp2, desc2 = self.detect_keypoints(img)
        if desc1 is None or desc2 is None:
            return None, None, None  # Pas de keypoints

        matches = self.match_keypoints(desc1, desc2, ratio=ratio)
        if len(matches) < 4:
            # Trop peu de correspondances
            return None, None, None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Distances globales
        distances = np.linalg.norm(src_pts - dst_pts, axis=1)
        mean_shift = np.mean(distances)
        max_shift = np.max(distances)

        # On peut aussi calculer le shift_x, shift_y moyen
        shift_x = np.mean(src_pts[:, 0] - dst_pts[:, 0])
        shift_y = np.mean(src_pts[:, 1] - dst_pts[:, 1])

        return (mean_shift, max_shift, (shift_x, shift_y))

    # -------------------------------------------------------------------------
    # 5. Analyser un dossier en mode key-frame mensuelle
    # -------------------------------------------------------------------------
    def analyze_folder_keyframe(self, folder_path, ratio=0.75):
        """
        - Liste toutes les images
        - Classe par ordre chronologique (en supposant le nom 'YYYY-MM-DD.tif')
        - Détecte la key-frame (1er jour de chaque mois)
        - Pour chaque image du mois, calcule le décalage par rapport à la key-frame mensuelle
        - Retourne un DataFrame avec (image_name, keyframe_name, mean_shift, max_shift, shift_x, shift_y)
        """
        # Filtrage .tif
        tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
        # On trie par date
        tif_files.sort()

        # Détection 1er jour
        def is_first_of_month(fname):
            base = fname.rsplit('.', 1)[0]
            try:
                dt = datetime.strptime(base, '%Y-%m-%d')
                return dt, dt.day == 1
            except ValueError:
                return None, False

        dated_files = []
        for f in tif_files:
            dt, is_first = is_first_of_month(f)
            if dt is not None:
                dated_files.append((f, dt, is_first))
        # Tri par date
        dated_files.sort(key=lambda x: x[1])

        if not dated_files:
            print("Aucune image datée trouvée.")
            return pd.DataFrame()

        # On va parcourir l'année/mois en cours, stocker la key-frame
        results = []
        current_keyframe = None
        current_month = None

        for file_name, dt, is_first in dated_files:
            # Si c'est le 1er jour => key-frame
            if is_first or (current_keyframe is None):
                # Mettre à jour la key-frame
                current_keyframe = file_name
                current_month = (dt.year, dt.month)

            # Vérifier si on a changé de mois
            if (dt.year, dt.month) != current_month:
                # Nouvelle key-frame
                current_keyframe = file_name
                current_month = (dt.year, dt.month)

            # Calcul du shift => par rapport à la key-frame
            ref_path = os.path.join(folder_path, current_keyframe)
            img_path = os.path.join(folder_path, file_name)

            arr_ref = self.load_and_reproject(ref_path)
            arr_img = self.load_and_reproject(img_path)

            mean_s, max_s, shift_xy = self.calculate_shift(arr_ref, arr_img, ratio=ratio)

            # Sauvegarder dans la liste
            results.append({
                'image_name': file_name,
                'keyframe': current_keyframe,
                'mean_shift': mean_s if mean_s is not None else 'NA',
                'max_shift': max_s if max_s is not None else 'NA',
                'shift_x': shift_xy[0] if shift_xy else 'NA',
                'shift_y': shift_xy[1] if shift_xy else 'NA'
            })

        return pd.DataFrame(results)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geotorch

# --- Your SimpleFeatureNet definition remains unchanged ---
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import geotorch
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import geotorch
import numpy as np
import rasterio

import torch
import torch.nn as nn
import torch.nn.functional as F
import geotorch
import numpy as np
import rasterio

import torch
import torch.nn as nn
import torch.nn.functional as F
import geotorch
import numpy as np
import rasterio

class SimpleFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conçu pour des images mono-bande : in_channels=1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        return x

class GeoKeyFrameAlignGPUParallel:
    def __init__(self, dst_crs="EPSG:4326", res=None):
        self.dst_crs = dst_crs
        self.res = res
        self.device_count = torch.cuda.device_count()
        self.devices = [torch.device(f"cuda:{i}") for i in range(self.device_count)]
        print(f"Nombre de GPUs disponibles : {self.device_count}")
        if self.device_count == 0:
            print("Aucun GPU détecté, utilisation possible du CPU.")

        self.feature_extractor = SimpleFeatureNet()
        self.feature_extractor.to(self.devices[0] if self.device_count > 0 else "cpu")
        # Si GeoTorch >= 0.3.0, on peut activer la contrainte orthogonale :
        # geotorch.orthogonal(self.feature_extractor.conv1, "weight")

    def load_and_reproject(self, image_path):
        """
        Lit l'image depuis un GeoTIFF (mono-bande) et la convertit en np.float32.
        Si souhaité, on peut ajouter une reprojection vers self.dst_crs.
        """
        with rasterio.open(image_path) as src:
            arr = src.read()  # shape: [bands, H, W]
        arr = arr[0, ...].astype(np.float32)  # On prend la première bande
        return arr

    def shift_image_tensor(self, t_img, shift):
        """
        Déplace un tenseur 4D [B, C, H, W] de (shift_x, shift_y) via grid_sample.
        """
        B, C, H_, W_ = t_img.shape
        device = t_img.device

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H_, dtype=torch.float32, device=device),
            torch.arange(W_, dtype=torch.float32, device=device),
            indexing='ij'
        )
        shift_x_norm = 2.0 * shift[0] / max(W_ - 1, 1.0)
        shift_y_norm = 2.0 * shift[1] / max(H_ - 1, 1.0)

        grid_x_norm = (2.0 * grid_x / (W_ - 1)) - 1.0 - shift_x_norm
        grid_y_norm = (2.0 * grid_y / (H_ - 1)) - 1.0 - shift_y_norm

        grid = torch.stack((grid_x_norm, grid_y_norm), dim=-1).unsqueeze(0)
        warped = F.grid_sample(t_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return warped

    def apply_shift(self, arr_img, shift_params):
        """
        Applique (shift_x, shift_y) à une image NumPy en utilisant PyTorch 
        puis renvoie l'image décalée en NumPy.
        """
        device = self.devices[0] if self.device_count > 0 else "cpu"
        # Prépare l'image
        t_img = torch.tensor(arr_img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) / 255.0
        shift = torch.tensor(shift_params, dtype=torch.float32, device=device)
        # Warp
        warped_t = self.shift_image_tensor(t_img, shift)
        # Retour en NumPy
        warped_np = (warped_t.squeeze(0).squeeze(0) * 255.0).detach().cpu().numpy()
        return warped_np

    def calculate_transform_geotorch(self, arr_ref, arr_img, iterations=50, lr=1e-3):
        """
        Calcule le décalage optimal entre arr_ref et arr_img 
        via descente de gradient sur un paramètre shift (x, y).
        Retourne (shift_x, shift_y), final_loss.
        """
        device = self.devices[0] if self.device_count > 0 else "cpu"

        t_ref = torch.tensor(arr_ref, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) / 255.0
        t_img = torch.tensor(arr_img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) / 255.0

        shift_param = nn.Parameter(torch.zeros(2, device=device))
        optimizer = torch.optim.Adam([shift_param], lr=lr)

        for _ in range(iterations):
            optimizer.zero_grad()
            warped_img = self.shift_image_tensor(t_img, shift_param)
            loss = F.l1_loss(t_ref, warped_img)
            loss.backward()
            optimizer.step()

        shift_final = shift_param.detach().cpu().numpy()
        return tuple(shift_final), float(loss.item())
