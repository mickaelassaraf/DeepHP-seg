import os
import re
from pathlib import Path
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
from torch.nn.functional import interpolate
from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib import pyplot as plt
import torch.nn as nn
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

# Charger le modèle
@st.cache_resource



def initialize_model(model_path, device="cuda"):
    """Load and return the pre-trained model and target layers."""
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    target_layers = [model.layer4[-1]]
    return model, target_layers


def generate_cam(method, model, processed_images, device, target_layer):
    """
    Generate Grad-CAM heatmaps for a batch of images and a model.

    Parameters:
    - method: Grad-CAM method to use (e.g., "GradCAM", "GradCAM++").
    - model: The PyTorch model for which Grad-CAM is applied.
    - processed_images: The preprocessed input image tensor (batch of images).
    - device: The device (CPU/GPU) to use for computation.
    - target_layer: The target layer for Grad-CAM.

    Returns:
    - visualizations: A list of visualized heatmaps (one per image in the batch).
    """
    batch_size = processed_images.size(0)  # Nombre d'images dans le batch
    targets = [ClassifierOutputTarget(1) for _ in range(batch_size)]  # Cible pour chaque image

    # Use `with` to ensure Grad-CAM resources are properly released
    with methods[method](model=model, target_layers=[model.layer4[-1]]) as cam:
        grayscale_cams = cam(input_tensor=processed_images, targets=targets)  # CAMs pour tout le batch

    visualizations = []
    for i in range(batch_size):
        grayscale_cam = grayscale_cams[i] # CAM pour l'image i
        rgb_img = processed_images[i].permute(1, 2, 0).cpu().numpy()  # Convertir en numpy (H, W, C)
        
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        visualizations.append(visualization)

    return visualizations





def overlay_cam_on_image(image, cam, cam_weight):
    cam_resized = interpolate(
        torch.tensor(cam).permute(2, 0, 1).unsqueeze(0).float(),
        size=image.size[::-1],
        mode='bilinear',
        align_corners=False
    )
    cam_resized = cam_resized.squeeze(0).permute(1, 2, 0).numpy()
    

    return cam_resized

from PIL import Image
import os
import torch

def process_sub_images_in_stack_with_batch_transform(input_dir, sub_images, transform, method, model, device, target_layer):
    images = []
    metadata = []

    # Charger toutes les images sans appliquer la transformation
    for file, (x_start, y_start) in sub_images.items():
        sub_image_path = os.path.join(input_dir, file)
        sub_image = Image.open(sub_image_path).convert("RGB")
        processed_image = transform(sub_image)  
        images.append(torch.tensor(np.array(processed_image)))  # [H, W, C] -> [C, H, W]

        metadata.append((file, x_start, y_start))

    # Empiler toutes les images en un seul tenseur (forme [batch_size, channels, height, width])
    images_tensor = torch.stack(images).to(device)

    # Générer les CAMs pour toutes les images
    cams = generate_cam(method, model, images_tensor, device, target_layer)

    # Associer les résultats aux métadonnées
    results = []
    for cam, (file, x_start, y_start) in zip(cams, metadata):
        results.append({
            'file': file,
            'x_start': x_start,
            'y_start': y_start,
            'cam': cam
        })
    
    return results



def reconstruct_grad_cam_image(method,files, input_dir, model, transform, device, output_image_path, target_layer):
    pattern = r"_(\d+)x(\d+)"
    x_max, y_max = 0, 0
    sub_images = {}

    for file in files:
        match = re.search(pattern, file)
        if match:
            y_start_y_end, x_start_x_end = map(str, match.groups())
            for i in range(1, len(x_start_x_end)):
                x_start = int(x_start_x_end[:i])
                x_end = int(x_start_x_end[i:])
                if x_end - x_start == 256:
                    break
            for i in range(1, len(y_start_y_end)):
                y_start = int(y_start_y_end[:i])
                y_end = int(y_start_y_end[i:])
                if y_end - y_start == 256:
                    break

            x_max = max(x_max, x_end)
            y_max = max(y_max, y_end)
            sub_images[file] = (x_start, y_start)

    final_image = Image.new('RGB', (x_max, y_max))
    cam_weight = 1

    generated_cams = process_sub_images_in_stack_with_batch_transform(input_dir, sub_images, transform, method, model, device, target_layer)
    # Fusionner les CAMs dans l'image finale
    for result in generated_cams:
        file = result['file']
        x_start = result['x_start']
        y_start = result['y_start']
        cam = result['cam']

        # Charger l'image originale correspondante
        sub_image_path = os.path.join(input_dir, file)
        sub_image = Image.open(sub_image_path).convert("RGB")

        # Appliquer le CAM à l'image
        cam_overlay = overlay_cam_on_image(sub_image, cam, cam_weight)

        # Convertir en PIL.Image et coller dans l'image finale
        cam_overlay_pil = Image.fromarray((cam_overlay ).astype(np.uint8))
        final_image.paste(cam_overlay_pil, (x_start, y_start))

    # Sauvegarder l'image finale
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    final_image.save(output_image_path)
    print(f"Image finale sauvegardée dans : {output_image_path}")
    
    final_image.save(output_image_path)
    return final_image

# Interface Streamlit
def main():
    st.title("Grad-CAM Reconstruction")
    st.sidebar.title("Options")
    

    model_path = st.sidebar.file_uploader("Charger le modèle PyTorch", type=["pth", "pt"])

    base_dir = os.path.join(os.getcwd(),"data/sampled_grouped/test_data/")
    subfolders = ["Positive", "Negative"]
    selected_folder = st.sidebar.selectbox("Choisissez un label", subfolders)
    folder_path = os.path.join(base_dir, selected_folder)

    if os.path.exists(folder_path):

        indices = os.listdir(folder_path)


        if indices:
            # Menu déroulant pour choisir un index
            selected_index = st.sidebar.selectbox("Choisissez un index", indices)
            st.write(f"Vous avez sélectionné : {selected_index}")
            input_dir = os.path.join(folder_path, selected_index)
        else:
            st.write("Aucun fichier trouvé dans le dossier sélectionné.")
    else:
        input_dir = ""
        st.write("Le dossier sélectionné n'existe pas.")



    output_dir = st.sidebar.text_input("Dossier de sortie", "output/")
    device = st.sidebar.selectbox("Choisir le périphérique", ["cpu", "cuda", "mps"], index=0)
    method = st.sidebar.selectbox("Choisir la méthode", methods.keys(), index=0)
    if model_path is not None:
        model,target_layers = initialize_model(model_path, device)
        st.sidebar.success("Modèle chargé avec succès.")

    else:
        st.error("Veuillez charger un modèle PyTorch.")
        return


    if not os.path.isdir(input_dir):
        st.error("Veuillez fournir un dossier valide contenant les patchs.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if st.button("Reconstruire l'image avec une methode CAM"):
        all_files = os.listdir(input_dir)
        pattern = r'(.+)_\w+\.jpeg'
        prefixes = {re.match(pattern, f).group(1) for f in all_files if re.match(pattern, f)}
        grouped_images = [[file for file in all_files if file.startswith(prefix)] for prefix in prefixes]

        for group in grouped_images:
            group_name = re.match(pattern, group[0]).group(1)
            output_path = os.path.join(output_dir, f"{group_name}_grad_cam.jpeg")
            final_image = reconstruct_grad_cam_image(method,group, input_dir, model, transform, device, output_path, target_layers)
            st.image(final_image, caption=f"Image reconstruite : {group_name}", use_column_width=True)


if __name__ == "__main__":
    main()
