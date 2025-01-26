# config.py
import os

# --- Configuration des hyperparamètres ---
config = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 5,
    "model_type": "ResNet18", # "SimpleNN" ou "VisionTransformer"
    "dataset": "sample_10000",
    "project_name": "PyTorch_WandB_Tqdm",
    "entity": "mickaelassaraf",  # Remplacez par votre nom d'utilisateur ou équipe W&B
    "wandb_mode": "online",  # "online", "offline", ou "disabled"
    "data_dir": "data/full",  # Dossier source des données
    "train_dir": "data/deepHP_100K/train",  # Dossier des données d'entraînement
    "test_dir": "data/deepHP_100K/val",  # Dossier des données de test
    "num_classes": 2
}
