import os
import pandas as pd
import rasterio
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
import re
from tqdm import tqdm

import sys

sys.path.append(os.path.abspath("src"))
from dataaugmentationsatellite import SatelliteAugmentation

###############################################################################
#                  FONCTION COLLATE POUR IGNORER LES ÉCHANTILLONS None
###############################################################################
def skip_none_collate_fn(batch):
    """
    Filtre les éléments None avant de les assembler en batch.
    Si tous les éléments du batch sont None, renvoie None.
    """
    filtered_batch = [x for x in batch if x is not None]
    return None if len(filtered_batch) == 0 else torch.utils.data.dataloader.default_collate(filtered_batch)


###############################################################################
#                  FONCTION UTILITAIRE POUR MODIFIER UNIQUEMENT LA DATE
###############################################################################
def replace_date_in_path(path):
    """
    Repère une date au format YYYY-MM-DD.tif dans un chemin et génère 
    une version où la date est remplacée par YYYY_MM_DD.tif.

    Si aucune date n'est trouvée, retourne le chemin d'origine.
    """
    date_pattern = re.compile(r"(\d{4})-(\d{2})-(\d{2})\.tif$")
    
    match = date_pattern.search(path)
    if match:
        date_hyphen = match.group(0)  # "YYYY-MM-DD.tif"
        date_underscore = date_hyphen.replace("-", "_")  # "YYYY_MM_DD.tif"
        return path.replace(date_hyphen, date_underscore)
    return path  # Retourne le chemin inchangé s'il n'y a pas de date

###############################################################################
#                         CLASSE FourBandSegDataset
###############################################################################

class FourBandSegDataset(Dataset):
    """
    Dataset pour la segmentation avec 4 canaux d'images (RGB+IR) et un masque de 8 classes.
    Seules les lignes avec alignment == True sont utilisées.
    """

    def __init__(self, dataframe, apply_augmentation=False):
        df_filtered = dataframe[dataframe['alignment'] == True].copy().reset_index(drop=True)
        self.df = df_filtered
        self.apply_augmentation = apply_augmentation

       
        if self.apply_augmentation:
            self.transform = SatelliteAugmentation()
        else:
            self.transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Récupération des chemins des fichiers
        planet_path_hyphen = row.get("planet_path", "")
        labels_path_hyphen = row.get("labels_path", "")

        # Gestion des versions avec underscore pour la date
        planet_path_underscore = replace_date_in_path(planet_path_hyphen)
        labels_path_underscore = replace_date_in_path(labels_path_hyphen)

        # Vérification de l'existence des fichiers
        planet_path = planet_path_hyphen if os.path.exists(planet_path_hyphen) else planet_path_underscore
        label_path = labels_path_hyphen if os.path.exists(labels_path_hyphen) else labels_path_underscore

        # Ignorer si les fichiers sont inexistants
        if not os.path.exists(planet_path) or not os.path.exists(label_path):
            return None

        # Lecture de l'image (4 canaux)
        try:
            with rasterio.open(planet_path) as src:
                image = src.read()  # [4, H, W]
        except:
            return None

        # Lecture du masque multi-classe (8 canaux binaires)
        try:
            with rasterio.open(label_path) as src:
                label_8bands = src.read()  # [8, H, W], valeurs {0, 255}
        except:
            return None
        
        # Conversion en masque unique d'indices de classe
        bin_mask = (label_8bands == 255)  # Convertir en booléen : 1 = présent, 0 = absent
        count_per_pixel = bin_mask.sum(axis=0)  # Nombre de classes présentes par pixel

        # Initialiser la carte de classes avec la première classe active par pixel
        class_indices = np.argmax(bin_mask, axis=0)

        # Pixels avec plusieurs classes actives (conflits)
        conflicts = (count_per_pixel > 1)
        class_indices[conflicts] = 255  # On ignore ces pixels

        # Pixels sans aucune classe
        no_class = (count_per_pixel == 0)
        class_indices[no_class] = 255  # On ignore ces pixels (ou mettre 0 si vous avez une classe "fond")

        # (Optionnel) transformations / augmentations

        data_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))
        ])

        
        image_tensor = torch.from_numpy(image).float()  # [4, H, W]
        label_tensor = torch.from_numpy(class_indices).long()  # [H, W]

        # Normalisation Min-Max entre 0 et 1
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

        # Normalisation standard pour ResNet
        mean = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(4, 1, 1)  # ImageNet + 4e bande
        std = torch.tensor([0.229, 0.224, 0.225, 0.5]).view(4, 1, 1)    # ImageNet + 4e bande
        image_tensor = (image_tensor - mean) / std

        # (Optionnel) transformations / augmentations
        if self.apply_augmentation and self.transform is not None:
            image_tensor, label_tensor = self.transform(image_tensor, label_tensor)

        return image_tensor, label_tensor


###############################################################################

###############################################################################
#                 Fonctions pour évaluer le modèle sur la validation
###############################################################################

def compute_iou_per_class(pred, target, num_classes=8, ignore_index=255):
    """
    Calcule l’IoU par classe pour une prédiction et un masque de vérité terrain.
    pred   : Tensor (H, W)
    target : Tensor (H, W)
    """
    ious = []
    for class_id in range(num_classes):
        # On crée un masque pour ignorer les pixels à 255
        valid_mask = (target != ignore_index)
        
        pred_class = (pred == class_id) & valid_mask
        target_class = (target == class_id) & valid_mask
        
        intersection = (pred_class & target_class).sum().item()
        union = (pred_class | target_class).sum().item()

        if union == 0:
            iou = np.nan  # aucune occurrence de cette classe
        else:
            iou = intersection / union
        
        ious.append(iou)
    return ious

def evaluate_model(model, val_loader, criterion, device, num_classes=8, ignore_index=255):
    """
    Évalue la perte moyenne (loss), l’IoU moyen (mIoU) et la précision (Accuracy)
    du modèle sur l’ensemble de validation.
    """
    model.eval()
    running_loss = 0.0
    all_ious = []
    total_correct = 0  # Compteur des pixels correctement classés
    total_pixels = 0   # Nombre total de pixels valides (excluant 255)

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)  # [batch_size, num_classes, H, W]
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Prédiction classe par pixel
            preds = torch.argmax(outputs, dim=1)  # [batch_size, H, W]

            # Calcul de l’IoU par classe pour chaque image du batch
            for i in range(len(preds)):
                ious = compute_iou_per_class(
                    preds[i], labels[i],
                    num_classes=num_classes,
                    ignore_index=ignore_index
                )
                all_ious.append(ious)

                # Masque des pixels valides (non 255)
                valid_mask = labels[i] != ignore_index

                # Calcul de l’Accuracy
                total_correct += torch.sum((preds[i] == labels[i]) * valid_mask).item()
                total_pixels += torch.sum(valid_mask).item()

    # Moyenne de la perte sur tout le jeu de validation
    val_loss = running_loss / len(val_loader) if len(val_loader) > 0 else 0.0

    # Calcul du mIoU
    all_ious = np.array(all_ious)  # shape = (total_images, num_classes)
    mean_iou_per_class = np.nanmean(all_ious, axis=0)  # Moyenne par classe
    miou = np.nanmean(mean_iou_per_class)               # Moyenne sur les classes

    # Calcul de l’Accuracy
    accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0  # Évite la division par zéro

    return val_loss, miou, accuracy
