import os
import sys
import pandas as pd
import numpy as np
import cv2
import torch
import rasterio
from rasterio.enums import Resampling
from datetime import datetime

sys.path.append("src")
# Import de la classe GPU parallèle (assure-toi que le chemin est correct)
from geo_keyfram_alignement import GeoKeyFrameAlignGPUParallel

def analyze_folder_keyframe_gpu(folder_path, aligner, ratio=0.65):
    """
    Parcourt le dossier (supposé contenir des images nommées 'YYYY-MM-DD.tif'),
    définit la key-frame mensuelle (le 1er du mois) et calcule pour chaque image
    le décalage par rapport à la key-frame en utilisant le recalage GPU parallèle.
    Retourne un DataFrame avec :
      - image_name : nom de l'image analysée
      - keyframe   : nom de l'image de référence du mois
      - mean_shift : décalage moyen
      - max_shift  : décalage maximum
      - shift_x    : décalage moyen en x
      - shift_y    : décalage moyen en y
    """
    # Liste des fichiers TIF dans le dossier
    tif_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    tif_files.sort()

    # Fonction pour déterminer si le fichier correspond au 1er jour du mois
    def is_first_of_month(fname):
        base = fname.rsplit('.', 1)[0]
        try:
            dt = datetime.strptime(base, '%Y-%m-%d')
            return dt, dt.day == 1
        except Exception:
            return None, False

    dated_files = []
    for f in tif_files:
        dt, is_first = is_first_of_month(f)
        if dt is not None:
            dated_files.append((f, dt, is_first))
    dated_files.sort(key=lambda x: x[1])
    if not dated_files:
        print(f"Aucune image datée trouvée dans {folder_path}")
        return pd.DataFrame()

    results = []
    current_keyframe = None
    current_month = None

    for file_name, dt, is_first in dated_files:
        # Définir la key-frame mensuelle
        if is_first or current_keyframe is None or ((dt.year, dt.month) != current_month):
            current_keyframe = file_name
            current_month = (dt.year, dt.month)
        keyframe_path = os.path.join(folder_path, current_keyframe)
        img_path = os.path.join(folder_path, file_name)
        try:
            # Charger et reprojeter les images via le GPU
            arr_ref = aligner.load_and_reproject(keyframe_path)
            arr_img = aligner.load_and_reproject(img_path)
            res = aligner.calculate_shift_parallel(arr_ref, arr_img)
            if res is None:
                mean_s, max_s, shift_xy = None, None, (None, None)
            else:
                mean_s, max_s, shift_xy = res
            results.append({
                "image_name": file_name,
                "keyframe": current_keyframe,
                "mean_shift": mean_s if mean_s is not None else "NA",
                "max_shift": max_s if max_s is not None else "NA",
                "shift_x": shift_xy[0] if shift_xy[0] is not None else "NA",
                "shift_y": shift_xy[1] if shift_xy[1] is not None else "NA"
            })
        except Exception as e:
            print(f"Erreur sur {img_path} : {e}")
            results.append({
                "image_name": file_name,
                "keyframe": current_keyframe,
                "mean_shift": "Error",
                "max_shift": "Error",
                "shift_x": "Error",
                "shift_y": "Error"
            })
    return pd.DataFrame(results)

def main():
    # Chargement du DataFrame fusionné contenant les chemins des zones
    df_merged_path = "dataframe/df_merged.csv"
    if not os.path.exists(df_merged_path):
        print(f"Fichier {df_merged_path} introuvable !")
        return
    df_merged = pd.read_csv(df_merged_path)
    df_zones = df_merged[["planet_path"]].copy()
    df_zones.rename(columns={"planet_path": "path"}, inplace=True)
    
    # Instanciation de la classe GPU parallèle
    aligner = GeoKeyFrameAlignGPUParallel(dst_crs="EPSG:4326", res=None)
    
    all_zone_dfs = []
    for zone_path in df_zones["path"]:
        print(f"🔍 Analyse de la zone : {zone_path}")
        if os.path.exists(zone_path):
            df_zone = analyze_folder_keyframe_gpu(zone_path, aligner, ratio=0.65)
            if not df_zone.empty:
                df_zone["zone_path"] = zone_path
                all_zone_dfs.append(df_zone)
            else:
                print(f"Aucun résultat pour {zone_path}")
        else:
            print(f"⚠️ Chemin non trouvé : {zone_path}")
    
    if all_zone_dfs:
        df_final = pd.concat(all_zone_dfs, ignore_index=True)
        output_path = "keyframes_alignment_gpu.csv"
        df_final.to_csv(output_path, index=False)
        print(f"✅ Fichier '{output_path}' généré avec succès !")
    else:
        print("⚠️ Aucun résultat à sauvegarder.")

if __name__ == '__main__':
    main()
