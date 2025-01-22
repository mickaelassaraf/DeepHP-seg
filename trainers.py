import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from model import SimpleNN
import os
import numpy as np

def compute_metrics(labels, predictions, num_classes):
    """
    Calculer les métriques détaillées : précision, rappel, F1-score.
    """
    conf_matrix = confusion_matrix(labels, predictions, labels=range(num_classes))
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    
    f1_scores = [report[str(i)]["f1-score"] for i in range(num_classes)]
    precision_scores = [report[str(i)]["precision"] for i in range(num_classes)]
    recall_scores = [report[str(i)]["recall"] for i in range(num_classes)]
    
    metrics = {
        "confusion_matrix": conf_matrix,
        "precision_scores": precision_scores,
        "recall_scores": recall_scores,
        "f1_scores": f1_scores,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }
    return metrics

def train_model(train_loader, test_loader, config, model_str="SimpleNN"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialisation du modèle
    if model_str == "SimpleNN":
        model = SimpleNN().to(device)
    elif model_str == "VisionTransformer":
        model = SimpleNN().to(device)
    else:
        raise ValueError(f"Modèle inconnu : {model_str}")

    # Configuration de l'entraînement
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

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
    model_path = f"models/{model_str.lower()}.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)  # Sauvegarde du modèle dans W&B
    print("Training complete, metrics and model logged in Weights & Biases.")
