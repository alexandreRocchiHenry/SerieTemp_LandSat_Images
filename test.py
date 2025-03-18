import rasterio

path = "../../../../tsi/data_education/fil_rouge_Airbus_2025/dynamic_earth_net/planet.18N/planet/18N/24E-187N/2415_3082_13/PF-SR/2018-04-29.tif"
try:
    with rasterio.open(path) as src:
        print("Métadonnées :", src.meta)
        print("Dimensions :", src.width, "x", src.height)
        # Lire quelques données pour vérifier
        data = src.read(1)
        print("Statistiques de la bande 1 :", data.min(), data.max())
except Exception as e:
    print("Erreur lors de l'ouverture :", e)
