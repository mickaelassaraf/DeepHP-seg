# data_preparation.py
import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_data(source_folder, train_dir, test_dir, test_size=0.2):
    """
    Prépare les données, sépare les images en train/test et organise les dossiers.

    Args:
        source_folder (str): Dossier source contenant les images.
        train_dir (str): Dossier où les données d'entraînement seront copiées.
        test_dir (str): Dossier où les données de test seront copiées.
        test_size (float): Proportion de données utilisées pour les tests.
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    source_dict = {}

    # Lire les fichiers dans le dossier source
    for filename in os.listdir(source_folder):
        if filename.endswith((".jpeg", ".png")):  # Adapter aux extensions d'images
            source = filename.rsplit('_', 1)[0]  # Extraire la source avant le dernier "_"
            if source not in source_dict:
                source_dict[source] = []
            source_dict[source].append(filename)

    # Séparer les sources en ensembles d'entraînement et de test
    train_sources, test_sources = train_test_split(list(source_dict.keys()), test_size=test_size, random_state=42)

    # Copier les fichiers vers les dossiers train/test en fonction de la source
    for source in train_sources:
        train_folder_path = os.path.join(train_dir, source)
        os.makedirs(train_folder_path, exist_ok=True)
        for filename in source_dict[source]:
            shutil.copy(os.path.join(source_folder, filename), os.path.join(train_folder_path, filename))

    for source in test_sources:
        test_folder_path = os.path.join(test_dir, source)
        os.makedirs(test_folder_path, exist_ok=True)
        for filename in source_dict[source]:
            shutil.copy(os.path.join(source_folder, filename), os.path.join(test_folder_path, filename))

    print(f"Train data: {len(train_sources)} sources, Test data: {len(test_sources)} sources.")
