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
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

class GeoKeyFrameAlignGPUParallel:
    """
    Version GPU parallèle utilisant PyTorch pour la détection et le matching de points d’intérêt.
    La répartition se fait sur plusieurs GPUs via torch.
    """
    def __init__(self, dst_crs="EPSG:4326", res=None):
        """
        Args:
            dst_crs (str): Code EPSG cible pour reprojection (par défaut EPSG:4326).
            res (tuple ou None): Résolution souhaitée (ex: (0.0001, 0.0001)).
        """
        self.dst_crs = dst_crs
        self.res = res
        self.device_count = torch.cuda.device_count()
        print(f"Nombre de GPUs disponibles : {self.device_count}")
        self.devices = [torch.device(f"cuda:{i}") for i in range(self.device_count)]

    def load_and_reproject(self, path):
        with rasterio.open(path) as src:
            try:
                data = src.read()
            except Exception as e:
                print(f"Erreur lors de la lecture de {path} : {e}")
                return None

            if data is None:
                print(f"Aucune donnée lue pour {path}")
                return None

            if src.crs == self.dst_crs and not self.res:
                return data
            else:
                transform, width, height = calculate_default_transform(
                    src.crs, self.dst_crs, src.width, src.height, *src.bounds, resolution=self.res
                )

                dest = np.zeros((src.count, height, width), dtype=src.dtypes[0])

                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=dest[i - 1],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=self.dst_crs,
                        resampling=Resampling.bilinear
                    )
                return dest

    def convert_image_type(self, image):
        """
        Convertit l'image en uint8 si nécessaire.
        """
        if image is None:
            return None
        if image.dtype != np.uint8:
            max_val = image.max() if image.max() != 0 else 1
            image = (image * (255.0 / max_val)).astype(np.uint8)
        return image

    def to_grayscale(self, data):
        """
        Convertit une image en niveaux de gris.
        Si l'image a plusieurs canaux, on suppose que les 3 premiers représentent une image couleur.
        """
        if data is None:
            return None
        if len(data.shape) == 3:
            C, H, W = data.shape
            if C == 1:
                return data[0, :, :]
            else:
                rgb = data[:3].transpose(1, 2, 0)
                gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
                return gray.astype(data.dtype)
        return data

    def preprocess(self, data):
        """
        Applique la conversion en uint8 et en niveaux de gris.
        """
        if data is None:
            return None
        data = self.convert_image_type(data)
        data = self.to_grayscale(data)
        return data

    def detect_keypoints_torch(self, image, device):
        """
        Détecte des points d’intérêt et extrait des descripteurs à l’aide de filtres Sobel.
        Cette implémentation convertit l'image en tenseur, calcule le gradient
        et sélectionne les 500 points ayant la plus grande magnitude.
        Pour chaque point, un patch de taille fixe est extrait et aplani.
        """
        if image is None:
            return [], None

        tensor_img = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

        sobel_x = torch.tensor([[[[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]]]], dtype=torch.float32, device=device)
        sobel_y = torch.tensor([[[[-1, -2, -1],
                                   [ 0,  0,  0],
                                   [ 1,  2,  1]]]], dtype=torch.float32, device=device)

        grad_x = torch.nn.functional.conv2d(tensor_img, sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(tensor_img, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        grad_mag_flat = grad_mag.view(-1)
        num_points = min(500, grad_mag_flat.numel())
        if num_points == 0:
            return [], None
        topk = torch.topk(grad_mag_flat, num_points)
        indices = topk.indices

        H, W = image.shape
        y_coords = (indices // W).float()
        x_coords = (indices % W).float()

        keypoints = [(float(x.item()), float(y.item())) for x, y in zip(x_coords, y_coords)]

        descriptors = []
        patch_size = 16
        pad = patch_size // 2
        padded = torch.nn.functional.pad(tensor_img, (pad, pad, pad, pad), mode='reflect')
        for (x, y) in keypoints:
            x_int = int(round(x))
            y_int = int(round(y))
            patch = padded[0, 0, y_int:y_int+patch_size, x_int:x_int+patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = torch.zeros(patch_size, patch_size, device=device)
            descriptor = patch.flatten()
            descriptors.append(descriptor)
        if len(descriptors) == 0:
            return keypoints, None
        descriptors = torch.stack(descriptors)
        return keypoints, descriptors

    def match_keypoints_torch(self, desc1, desc2, ratio=0.75):
        """
        Effectue le matching des descripteurs en calculant la distance euclidienne.
        Pour chaque descripteur de desc1, on sélectionne le meilleur match dans desc2
        si le ratio entre la première et la deuxième distance est inférieur à 'ratio'.
        """
        if desc1 is None or desc2 is None:
            return []
        distances = torch.cdist(desc1, desc2, p=2)
        good_matches = []
        for i in range(distances.shape[0]):
            dists = distances[i]
            sorted_vals, sorted_idx = torch.sort(dists)
            if sorted_vals[0] < ratio * sorted_vals[1]:
                match = {'queryIdx': i, 'trainIdx': sorted_idx[0].item(), 'distance': sorted_vals[0].item()}
                good_matches.append(match)
        return good_matches

    def calculate_shift_parallel(self, arr_ref, arr_img):
        """
        Calcule le décalage entre deux images en utilisant plusieurs GPUs pour la détection.
        Pour simplifier, l'image de référence est traitée sur tous les GPUs, 
        mais seul le résultat du premier est utilisé pour le matching avec l'image cible.
        """
        if arr_ref is None or arr_img is None:
            print("Erreur: une des images d'entrée est None, impossible de calculer le décalage.")
            return None, None, None

        ref = self.preprocess(arr_ref)
        img = self.preprocess(arr_img)
        if ref is None or img is None:
            print("Erreur: le prétraitement a échoué.")
            return None, None, None

        keypoints_list = []
        descriptors_list = []
        for device in self.devices:
            kp, desc = self.detect_keypoints_torch(ref, device)
            keypoints_list.append(kp)
            descriptors_list.append(desc)

        keypoints_ref = keypoints_list[0]
        descriptors_ref = descriptors_list[0]
        keypoints_img, descriptors_img = self.detect_keypoints_torch(img, self.devices[0])

        good_matches = self.match_keypoints_torch(descriptors_ref, descriptors_img)
        if len(good_matches) < 4:
            return None, None, None

        src_pts = np.float32([keypoints_ref[m['queryIdx']] for m in good_matches])
        dst_pts = np.float32([keypoints_img[m['trainIdx']] for m in good_matches])

        distances = np.linalg.norm(src_pts - dst_pts, axis=1)
        mean_shift = np.mean(distances)
        max_shift = np.max(distances)
        shift_x = np.mean(src_pts[:, 0] - dst_pts[:, 0])
        shift_y = np.mean(src_pts[:, 1] - dst_pts[:, 1])

        return mean_shift, max_shift, (shift_x, shift_y)
