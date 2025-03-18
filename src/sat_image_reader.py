import rasterio
import numpy as np
import matplotlib.pyplot as plt


class SatImageReader:
    """
    Classe pour lire et afficher des images satellitaires.
    """

    def __init__(self, file_path):
        """
        Initialise le lecteur d'image satellitaire.

        Parameters:
        - file_path (str): Chemin vers le fichier image raster.
        """
        self.file_path = file_path
        try:
            self.image = rasterio.open(self.file_path)
            self.bandes = self.image.count
            self.metadata = self.image.meta
        except rasterio.errors.RasterioIOError as e:
            print(f"Erreur lors de l'ouverture du fichier: {e}")
            raise SystemExit(e)

    def __del__(self):
        """
        Assure la fermeture correcte du fichier raster lors de la destruction de l'objet.

        """
        if self.image and not self.image.closed:
            self.image.close()

    def get_n_bands(self):
        """
        Retourne le nombre de bandes de l'image.
        """
        return self.bandes

    def show_band(self, band=1):
        """
        Affiche une bande spécifique de l'image.

        Parameters:
        - band (int): Numéro de la bande à afficher.
        - fig_size (tuple): Taille de la figure matplotlib.
        """
        if band < 1 or band > self.bandes:
            print(f"Bande {band} introuvable")
            raise SystemExit(f"Bande {band} introuvable")
        data = self.image.read(band)

        plt.imshow(data, cmap="gray")
        plt.colorbar()
        plt.title(f"Bande {band}")

    def show_rgb(self, bands_rgb=(3, 2, 1)):
        """
        Affiche une image RGB composée de trois bandes, normalisées 0–255.

        Parameters:
        - bands_rgb (tuple): Indices des bandes pour rouge, vert et bleu.
        """
        if self.bandes < 3:
            print("Nombre insuffisant de bandes pour afficher une image RGB")
            raise SystemExit("Nombre insuffisant de bandes pour afficher une image RGB")

        try:
            red = self.image.read(bands_rgb[0])
            green = self.image.read(bands_rgb[1])
            blue = self.image.read(bands_rgb[2])

            def normalize_to_255(band_array):
                bmin, bmax = band_array.min(), band_array.max()
                if bmax > bmin:
                    # Normalisation 0–1 puis mise à l’échelle 0–255
                    band_scaled = (band_array - bmin) / (bmax - bmin) * 255
                else:
                    # Cas d'une bande uniforme
                    band_scaled = np.zeros_like(band_array)
                return band_scaled.astype(np.uint8)

            red_255 = normalize_to_255(red)
            green_255 = normalize_to_255(green)
            blue_255 = normalize_to_255(blue)

            rgb_image = np.dstack((red_255, green_255, blue_255))

            plt.imshow(rgb_image)
            plt.title("Image RGB (0–255)")

        except rasterio.errors.RasterioIOError as e:
            print(f"Erreur lors de la lecture des bandes RGB: {e}")
            raise SystemExit(e)
        except Exception as e:
            print(f"Erreur lors de l'affichage de l'image RGB: {e}")
            raise SystemExit(e)

    def show_metadata(self):
        """
        Affiche les métadonnées de l'image.
        """
        print(self.metadata)

    def show_band_hist(self, band=1):
        """
        Affiche l'histogramme d'une bande spécifique.

        Parameters:
        - band (int): Numéro de la bande pour laquelle afficher l'histogramme.
        """
        if band < 1 or band > self.bandes:
            print(f"Bande {band} introuvable")
            raise SystemExit(f"Bande {band} introuvable")
        data = self.image.read(band)
        plt.hist(data.flatten(), bins=256, range=(data.min(), data.max()), color="gray")
        plt.title(f"Histogramme de la bande {band}")
        plt.xlabel("Valeur de pixel")
        plt.ylabel("Fréquence")

    def show_rgb_hist(self, bands_rgb=(3, 2, 1), show_infrared=True):
        """
        Affiche les histogrammes des bandes RGB et éventuellement de la bande infrarouge.

        Parameters:
        - bands_rgb (tuple): Indices des bandes pour rouge, vert et bleu.
        - show_infrared (bool): Indique s'il faut afficher l'histogramme de la bande infrarouge.
        """
        red_band, green_band, blue_band = bands_rgb

        # Vérification des bandes
        max_band = max(bands_rgb)
        if show_infrared:
            max_band = max(max_band, 4)
        if max_band > self.bandes:
            print("Les bandes spécifiées dépassent le nombre de bandes disponibles.")
            raise SystemExit(
                "Les bandes spécifiées dépassent le nombre de bandes disponibles."
            )

        # Lecture des données des bandes
        try:
            data_red = self.image.read(red_band).flatten()
            data_green = self.image.read(green_band).flatten()
            data_blue = self.image.read(blue_band).flatten()
        except rasterio.errors.RasterioIOError as e:
            print(f"Erreur lors de la lecture des bandes RGB: {e}")
            raise SystemExit(e)

        # optionnel : Bande infrarouge, lors de show_infrared=True
        data_infrared = None
        if show_infrared and self.bandes >= 4:
            try:
                data_infrared = self.image.read(4).flatten()
            except rasterio.errors.RasterioIOError as e:
                print(f"Erreur lors de la lecture de la bande infrarouge: {e}")
                raise SystemExit(e)

        bins = 256
        if not show_infrared:
            range_bins = (0, 255)
        else:
            if data_infrared is not None:
                max_val = max(
                    data_red.max(),
                    data_green.max(),
                    data_blue.max(),
                    data_infrared.max(),
                )
            else:
                max_val = max(data_red.max(), data_green.max(), data_blue.max())
            range_bins = (0, max_val)

        # histogrammes superposés

        plt.hist(
            data_red, bins=bins, range=range_bins, color="red", alpha=0.5, label="Rouge"
        )
        plt.hist(
            data_green,
            bins=bins,
            range=range_bins,
            color="green",
            alpha=0.5,
            label="Vert",
        )
        plt.hist(
            data_blue,
            bins=bins,
            range=range_bins,
            color="blue",
            alpha=0.5,
            label="Bleu",
        )

        if show_infrared and self.bandes >= 4 and data_infrared is not None:
            plt.hist(
                data_infrared,
                bins=bins,
                range=range_bins,
                color="black",
                alpha=0.5,
                label="Infrarouge",
            )

        plt.title("Histogramme des bandes RGB")
        plt.xlabel("Valeur des pixels")
        plt.ylabel("Fréquence")
        plt.legend()

    def calculate_ndvi(self, red_band_index=3, nir_band_index=4):
        """
        Calcule l'indice NDVI à partir des bandes rouge et infrarouge.

        Parameters:
        - red_band_index (int): Index de la bande rouge.
        - nir_band_index (int): Index de la bande infrarouge.

        Returns:
        - ndvi (np.ndarray): Indice NDVI calculé, ou None si les bandes sont invalides.
        """
        if self.bandes < max(red_band_index, nir_band_index):
            print("Nombre insuffisant de bandes pour calculer le NDVI")
            return None
        try:
            red = self.image.read(red_band_index).astype("float32")
            nir = self.image.read(nir_band_index).astype("float32")
            denominator = nir + red  # entre 0 et 2
            with np.errstate(divide="ignore", invalid="ignore"):
                ndvi = np.true_divide((nir - red), denominator)  # entre -1 et 1
                ndvi[denominator == 0] = 0
            return ndvi
        except rasterio.errors.RasterioIOError as e:
            print(f"Erreur lors de la lecture des bandes pour NDVI: {e}")
            return None

    def show_ndvi(self, red_band_index=3, nir_band_index=4, threshold=None):
        """
        Affiche l'indice NDVI, avec option de seuil. Les valeurs au-dessus du seuil sont binarisées.
        Se renseigner sur les seuils classique de NDVI (végétation, eau, sol, etc).

        Parameters:
        - red_band_index (int): Index de la bande rouge.
        - nir_band_index (int): Index de la bande infrarouge.
        - threshold (float, optional): Seuil pour binariser le NDVI.
        """
        ndvi = self.calculate_ndvi(red_band_index, nir_band_index)
        if ndvi is None:
            print("NDVI non calculé.")
            raise SystemExit("NDVI non calculé.")
        if threshold is not None:
            ndvi = np.where(ndvi > threshold, 1, 0)

        if threshold is not None:
            plt.imshow(ndvi, cmap="gray")
        else:
            plt.imshow(ndvi, cmap="RdYlGn")
        plt.colorbar()
        plt.title("NDVI")
