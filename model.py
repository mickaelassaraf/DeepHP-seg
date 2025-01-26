# model.py
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.model(x)


import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10, input_size=(3, 224, 224)):  # Specify input size (C, H, W)
        super(SmallCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Première couche convolutive
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Convolution (3 -> 16 canaux)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Réduction par 2

            # Deuxième couche convolutive
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Convolution (16 -> 32 canaux)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Réduction par 2
        )

        # Dynamically calculate the size of the flattened tensor
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)  # Create a dummy input tensor
            flattened_size = self.conv_layers(dummy_input).view(1, -1).size(1)

        # Couches entièrement connectées
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Aplatir la sortie pour les couches FC
            nn.Linear(flattened_size, 128),  # Fully connected with dynamically calculated input size
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Sortie : nombre de classes
        )
    
    def forward(self, x):
        x = self.conv_layers(x)  # Passer par les couches convolutives
        x = self.fc_layers(x)    # Passer par les couches entièrement connectées
        return x

    
