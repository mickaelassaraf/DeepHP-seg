# config.py
import os

# --- Configuration des hyperparamètres ---
config = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 5,
    "model_type": "SimpleNN",
    "dataset": "sample_10000",
    "project_name": "PyTorch_WandB_Tqdm",
    "entity": "mickaelassaraf",  # Remplacez par votre nom d'utilisateur ou équipe W&B
    "wandb_mode": "online",  # "online", "offline", ou "disabled"
    "data_dir": "data/sampled_10000",  # Dossier source des données
    "train_dir": "data/sampled_grouped/train_data",  # Dossier des données d'entraînement
    "test_dir": "data/sampled_grouped/test_data",  # Dossier des données de test
    "num_classes": 2,
}
