from torch.utils.data import Dataset, DataLoader

class SingleImageDataset(Dataset):
    """
    Dataset personnalisé pour une image unique.
    """
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # Label factice pour compatibilité

def load_data(input_path, transform):
    """
    Charge les données en fonction de l'entrée (dossier ou image unique).

    Args:
        input_path: Chemin vers une image ou un dossier.
        transform: Transformations à appliquer.

    Returns:
        DataLoader prêt à l'emploi.
    """
    if os.path.isdir(input_path):
        dataset = datasets.ImageFolder(root=input_path, transform=transform)
    elif os.path.isfile(input_path):
        dataset = SingleImageDataset(image_path=input_path, transform=transform)
    else:
        raise ValueError("Le chemin spécifié n'est ni un fichier ni un dossier valide.")
    return DataLoader(dataset, batch_size=1, shuffle=False)

