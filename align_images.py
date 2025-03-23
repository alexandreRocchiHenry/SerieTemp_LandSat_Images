import os
import sys
import pandas as pd
from datetime import datetime
from tqdm import tqdm  # Ajout de tqdm pour le suivi de progression

sys.path.append("src")  # adapter selon votre structure de dossiers
from geo_keyfram_alignement import GeoKeyFrameAlignGPUParallel

def analyze_folder_keyframe_gpu(folder_path, aligner):
    """
    Parcourt un dossier d'images .tif (nommées au format YYYY-MM-DD.tif).
    Aligne chaque mois sur le mois précédent, et à l'intérieur du mois 
    aligne chaque image sur la keyframe de ce mois.
    Retourne un DataFrame des résultats.
    """

    # Liste de tous les fichiers TIF
    tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    tif_files.sort()

    def parse_date_from_filename(fname):
        base = fname.rsplit('.', 1)[0]
        try:
            dt = datetime.strptime(base, '%Y-%m-%d')
            return dt
        except:
            return None

    # Construit une liste (fichier, dt)
    dated_files = []
    for f in tif_files:
        dt = parse_date_from_filename(f)
        if dt is not None:
            dated_files.append((f, dt))
    dated_files.sort(key=lambda x: x[1])  # tri par date

    if not dated_files:
        print(f"Aucune image datée trouvée dans {folder_path}")
        return pd.DataFrame()

    # Dictionnaire : (year, month) -> image NumPy déjà alignée globalement
    keyframe_storage = {}
    previous_month = None

    results = []

    print(f" Traitement des images ({len(dated_files)} fichiers)")
    for file_name, dt in tqdm(dated_files, desc="Alignement des images", unit="image"):
        month_tuple = (dt.year, dt.month)
        img_path = os.path.join(folder_path, file_name)

        # Si on entre dans un nouveau mois, on définit la première image comme keyframe
        # et on l'aligne sur le mois précédent si nécessaire
        if month_tuple not in keyframe_storage:
            # Load la nouvelle keyframe brute
            arr_new_keyframe = aligner.load_and_reproject(img_path)

            # Si on a un mois précédent, on aligne la nouvelle keyframe sur la précédente
            if previous_month is not None:
                arr_ref_prev = keyframe_storage[previous_month]
                shift_params, final_loss = aligner.calculate_transform_geotorch(
                    arr_ref_prev, arr_new_keyframe
                )
                # On applique ce shift pour corriger la keyframe du nouveau mois
                arr_new_keyframe_aligned = aligner.apply_shift(arr_new_keyframe, shift_params)
                keyframe_storage[month_tuple] = arr_new_keyframe_aligned
            else:
                # Premier mois de la série
                keyframe_storage[month_tuple] = arr_new_keyframe

            previous_month = month_tuple

        # On aligne l'image courante sur la keyframe déjà corrigée du mois
        arr_ref = keyframe_storage[month_tuple]
        arr_img = aligner.load_and_reproject(img_path)

        shift_params, final_loss = aligner.calculate_transform_geotorch(arr_ref, arr_img)
        shift_x, shift_y = shift_params

        results.append({
            "image_name": file_name,
            "year": dt.year,
            "month": dt.month,
            "final_loss": final_loss,
            "shift_x": shift_x,
            "shift_y": shift_y
        })

    return pd.DataFrame(results)

def main():
    # 1) Charger le DataFrame fusionné, qui contient 'planet_path'
    df_merged_path = "dataframe/df_merged.csv"
    if not os.path.exists(df_merged_path):
        print(f"Fichier {df_merged_path} introuvable.")
        return

    df_merged = pd.read_csv(df_merged_path)
    df_zones = df_merged[["planet_path"]].copy()
    df_zones.rename(columns={"planet_path": "path"}, inplace=True)

    # 2) Instancier la classe d'alignement
    aligner = GeoKeyFrameAlignGPUParallel(dst_crs="EPSG:4326", res=None)

    # 3) Itérer sur chaque zone avec une barre de progression globale
    all_zone_dfs = []
    print(f"🔍 Analyse de {len(df_zones)} zones...")
    for zone_path in tqdm(df_zones["path"], desc=" Traitement des zones", unit="zone"):
        print(f"\n Analyse de la zone : {zone_path}")
        if not os.path.exists(zone_path):
            print(f" Chemin non trouvé : {zone_path}")
            continue

        df_zone = analyze_folder_keyframe_gpu(zone_path, aligner)
        if not df_zone.empty:
            df_zone["zone_path"] = zone_path
            all_zone_dfs.append(df_zone)
        else:
            print(f"Aucun résultat pour {zone_path}")

    # 4) Concaténer et sauvegarder
    if all_zone_dfs:
        df_final = pd.concat(all_zone_dfs, ignore_index=True)
        output_path = "keyframes_alignment_geotorch.csv"
        df_final.to_csv(output_path, index=False)
        print(f"\n✅ Fichier '{output_path}' généré avec succès.")
    else:
        print("\n⚠️ Aucun résultat à sauvegarder.")

if __name__ == '__main__':
    main()
