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
    
    train_dataset = datasets.ImageFolder(root=config["train_dir"], transform=transform)
    test_dataset = datasets.ImageFolder(root=config["test_dir"], transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Entraînement du modèle
    train_model(train_loader, test_loader, config)

if __name__ == "__main__":
    main()
