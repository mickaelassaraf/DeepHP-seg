# main.py
import argparse
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import config
from trainers import train_model
from transformers import ViTFeatureExtractor

def main():
    # Initialisation de W&B
    wandb.init(project=config["project_name"], entity=config["entity"], config=config, mode=config["wandb_mode"])


    # Préparation des transformations et DataLoaders
    if config["model_type"] == "SimpleNN":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    elif config["model_type"] == "mobilenet_v2":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif config["model_type"] == "ResNet50":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    elif config["model_type"] == "ResNet18":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif config["model_type"] == "SmallCNN":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    import random
    from torch.utils.data import Subset

    # Charger le dataset complet
    dataset_train = datasets.ImageFolder(root=config["train_dir"], transform=transform)
    dataset = datasets.ImageFolder(root = config["data_dir"], transform=transform)

    # Nombre d'images à utiliser
    total_images = 1000  # Exemple : 10,000 images seulement

    # Sélectionner des indices aléatoires
    selected_indices = random.sample(range(len(dataset)), total_images)

    # Créer un sous-dataset
    subset = Subset(dataset, selected_indices)

    # Diviser le sous-dataset en entraînement et test

    # DataLoaders

    train_loader = DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(subset, batch_size=config["batch_size"], shuffle=False)


    # Entraînement du modèle
    train_model(train_loader, test_loader, config)

if __name__ == "__main__":
    main()
