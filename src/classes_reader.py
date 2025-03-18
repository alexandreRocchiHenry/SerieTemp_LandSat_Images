import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt


class ClassesReader:
    def __init__(self, file_path):
        self.file_path = file_path
        try:
            self.image = rasterio.open(self.file_path)
            self.bandes = self.image.count
            self.metadata = self.image.meta
            self.dict_classes = {
                "Impervious surfaces": 1,
                "agriculture": 2,
                "forest": 3,
                "wetlands": 4,
                "soil": 5,
                "water": 6,
                "snow": 7,
            }
            self.reverse_dict_classes = {
                v: k for k, v in self.dict_classes.items()
            }  # Reversed dictionary

        except rasterio.errors.RasterioIOError as e:
            print(f"Erreur lors de l'ouverture du fichier: {e}")
            raise SystemExit(e)

    def __del__(self):
        if self.image and not self.image.closed:
            self.image.close()

    def show_band(self, band=1):
        """Affiche une bande spécifique de l'image.

        Args:
            band (int, optional): La bande à afficher. Defaults to 1.

        Raises:
            SystemExit: Si la bande n'existe pas.

        Returns:
            np.array: La bande affichée
        """
        if band < 1 or band > self.bandes:
            raise SystemExit(f"Bande {band} introuvable")
        data = self.image.read(band)

        plt.imshow(data, cmap="gray")
        plt.colorbar()
        plt.title(f"Bande {band}")
        return data

    def show_class(self, classe):
        data = self.image.read(classe)

        classe_name = self.reverse_dict_classes.get(classe, f"Unknown Class ({classe})")
        plt.imshow(data, cmap="BuGn")
        plt.colorbar()
        plt.title(f"Classe {classe_name}")
        return data

    def get_n_bands(self):
        """Permet de retourner le nombre de bandes de l'image.

        Returns:
            int:nombre de bandes de l'image
        """
        return self.bandes

    def detect_classes(self):
        """
        Detect the classes the rasterio file. Return the bands where not every pixel is 0
        """
        classes = []
        for i in range(1, self.bandes + 1):
            data = self.image.read(i)
            if np.any(data):
                classes.append(i)
        return classes

    def show_class_list(self, class_list=None):
        """Affiche les classes de l'image.

        Args:
            class_list (int list, optional): La liste des classes à afficher. Defaults to None.

        Returns:
            np.array: Les classes affichées
        """
        if class_list is None:
            class_list = self.detect_classes()
        class_list_data = []
        for i in range(len(class_list)):
            classe = class_list[i]
            plt.subplot(1, len(class_list), i + 1)
            self.show_class(classe)
            class_list_data.append(self.image.read(classe))
        return class_list_data

    # TODO : Creation de fonction pour les metadatas
    # TODO : Creation de fonction pour stats des classes
    # TODO : Lecture des classes sous forme de masque supperposables sur l'image
