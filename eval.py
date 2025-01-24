import argparse
import os
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, f1_score, accuracy_score
from PIL import Image
import wandb
from tqdm import tqdm


def load_model(model_path, device="cuda"):
    """
    Charge un modèle PyTorch sauvegardé.
    Args:
        model_path (str): Chemin vers le fichier du modèle sauvegardé.
        device (str): Périphérique à utiliser (par défaut : cuda).
    Returns:
        model: Le modèle chargé.
    """
    model = models.mobilenet_v2(pretrained=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def process_input(input_path, transform):
    """
    Prépare les données (dossier ou image unique) pour l'évaluation.
    Args:
        input_path (str): Chemin vers un dossier ou une image.
        transform (torchvision.transforms.Compose): Transformations à appliquer.
    Returns:
        DataLoader: Un DataLoader prêt pour l'évaluation.
    """
    if os.path.isdir(input_path):
        print("Chargement des données depuis le dossier...")
        dataset = datasets.ImageFolder(root=input_path, transform=transform)
    elif os.path.isfile(input_path):
        class SingleImageDataset(torch.utils.data.Dataset):
            def __init__(self, image_path, transform):
                self.image_path = image_path
                self.transform = transform

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                image = Image.open(self.image_path).convert("RGB")
                return self.transform(image), 0  # Retourne un label factice

        dataset = SingleImageDataset(image_path=input_path, transform=transform)
    else:
        raise ValueError("Chemin spécifié non valide.")
    return DataLoader(dataset, batch_size=16, shuffle=False)


def evaluate(model, dataloader, results_file, save_every=1000, device="cuda"):
    """
    Évalue le modèle sur le DataLoader fourni.
    Args:
        model: Le modèle PyTorch à évaluer.
        dataloader: DataLoader contenant les données d'entrée.
        results_file: Fichier pour enregistrer les résultats.
        save_every (int): Nombre d'échantillons après lequel les résultats sont sauvegardés.
        device: Périphérique à utiliser (par défaut : cuda).
    Returns:
        list: Prédictions et labels réels.
    """
    all_preds, all_labels = [], []
    batch_count = 0
    sample_count = 0

    with results_file.open("a") as f:
        for images, targets in tqdm(dataloader, desc="Évaluation en cours", unit="batch"):
            images, targets = images.to(device), targets.to(device)

            with torch.no_grad():
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(targets.cpu().tolist())
            sample_count += len(images)

            # Sauvegarde et affichage périodiques
            if sample_count >= save_every:
                # Calcul des métriques pour les 1000 derniers échantillons
                recent_preds = all_preds[-save_every:]
                recent_labels = all_labels[-save_every:]
                precision = precision_score(recent_labels, recent_preds, average="weighted", zero_division=0)
                f1 = f1_score(recent_labels, recent_preds, average="weighted", zero_division=0)
                accuracy = accuracy_score(recent_labels, recent_preds)

                print(f"Metrics for last {save_every} samples - Precision: {precision:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
                f.write(f"Metrics for last {save_every} samples - Precision: {precision:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}\n")

                sample_count = 0

    # Calcul des métriques globales
    global_precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    global_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    global_accuracy = accuracy_score(all_labels, all_preds)

    # Affichage des métriques globales
    print(f"Global metrics - Precision: {global_precision:.4f}, F1 Score: {global_f1:.4f}, Accuracy: {global_accuracy:.4f}")

    # Retourner les prédictions et labels
    return all_preds, all_labels, global_precision, global_f1, global_accuracy


def main():
    parser = argparse.ArgumentParser(description="Évaluation d'un modèle PyTorch.")
    parser.add_argument("--input", type=str, required=True, help="Chemin vers une image ou un dossier.")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le modèle PyTorch sauvegardé.")
    parser.add_argument("--results_dir", type=str, default="results", help="Dossier pour sauvegarder les résultats.")
    parser.add_argument("--wandb_project", type=str, default="model-evaluation", help="Nom du projet W&B.")
    parser.add_argument("--device", type=str, default="mps", help="Périphérique à utiliser (cuda ou cpu).")
    args = parser.parse_args()

    # Initialisation de W&B
    wandb.init(project=args.wandb_project, name="evaluation_run", config=args)

    # Préparation des transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Chargement des données et du modèle
    dataloader = process_input(args.input, transform)
    model = load_model(args.model_path, device=args.device)

    # Résultats
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "results.txt"

    # Évaluation avec sauvegarde périodique
    predictions, labels, precision, f1, accuracy = evaluate(
        model, dataloader, results_file, save_every=1000, device=args.device
    )

    # Enregistrement des métriques globales dans W&B
    wandb.log({
        "global_precision": precision,
        "global_f1_score": f1,
        "global_accuracy": accuracy
    })
    print(f"Résultats sauvegardés dans {results_file}")

    wandb.finish()


if __name__ == "__main__":
    main()
