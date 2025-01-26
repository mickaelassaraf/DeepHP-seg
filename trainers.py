import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from model import SimpleNN, SmallCNN
import os
import numpy as np
from transformers import ViTFeatureExtractor
import torchvision.models as models
from utils.metrics import compute_metrics
from torch import nn

def initialize_model(model_str, num_classes, device):
    if model_str == "SimpleNN":
        return SimpleNN().to(device)
    elif model_str == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True).to(device)
        model.load_state_dict(torch.load("models/mobilenet_v2.pth", map_location=device))
        
        return model.to(device)
    elif model_str == "SmallCNN":
        return SmallCNN(num_classes=num_classes).to(device)
    elif model_str == "ResNet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
        return model.to(device)
    elif model_str == "ResNet50":
        model = models.resnet50(pretrained=True)
       
        return model.to(device)
    else:
        raise ValueError(f"Unknown model: {model_str}")



def train_model(train_loader, test_loader, config):
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Initialisation du modèle
    model = initialize_model(config["model_type"], config["num_classes"], device)
    # Configuration de l'entraînement
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=config["learning_rate"]
)


    # Initialisation de W&B
    wandb.init(project=config["project_name"], config=config)
    wandb.watch(model, log="all", log_freq=100)

    # Variables pour le suivi des meilleures métriques
    best_test_accuracy = 0

    # --- Entraînement ---
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            progress_bar.set_postfix(loss=loss.item())

        # Calcul des métriques d'entraînement
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Log des métriques d'entraînement
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
        })

        print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        # --- Évaluation ---
        model.eval()
        correct_test = 0
        total_test = 0
        running_test_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Testing", leave=False)
            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                running_test_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Calcul des métriques de test
        avg_test_loss = running_test_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test

        # Calcul des métriques avancées
        metrics = compute_metrics(all_labels, all_predictions, num_classes=config["num_classes"])
        wandb.log({
            "epoch": epoch + 1,
            "test_loss": avg_test_loss,
            "test_accuracy": test_accuracy,
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        })

        # Log de la matrice de confusion sous forme de tableau
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_predictions,
            class_names=[str(i) for i in range(config["num_classes"])]
        )})

        # Mise à jour du meilleur score
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy

        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Test Loss: {avg_test_loss:.4f}")

    # Log des meilleures métriques
    wandb.log({"best_test_accuracy": best_test_accuracy})

    # Sauvegarde du modèle
    model_path = f"models/{config["model_type"].lower()}.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)  # Sauvegarde du modèle dans W&B
    print("Training complete, metrics and model logged in Weights & Biases.")
