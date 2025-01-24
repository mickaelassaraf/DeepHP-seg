import os
import re
from pathlib import Path
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
from torch.nn.functional import interpolate
from matplotlib import pyplot as plt


# Charger le modèle
@st.cache_resource
def load_model(model_path, device="cuda"):
    model = models.mobilenet_v2(pretrained=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Préparer l'image
def preprocess_image(image, transform):
    return transform(image).unsqueeze(0)

# Grad-CAM pour une visualisation des activations
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Enregistrement des gradients
        target_layer.register_backward_hook(self.save_gradient)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        features = None

        # Hook pour capturer les activations
        def hook(module, input, output):
            nonlocal features
            features = output
        handle = self.target_layer.register_forward_hook(hook)

        outputs = self.model(x)
        handle.remove()

        if class_idx is None:
            class_idx = torch.argmax(outputs, dim=1).item()

        self.model.zero_grad()
        outputs[0, class_idx].backward(retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(features.shape[1]):
            features[0, i, :, :] *= pooled_gradients[i]

        cam = torch.mean(features, dim=1).squeeze()
        cam = torch.clamp(cam, min=0).cpu().detach().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

def generate_cam(model, image, device, target_layer_name):
    target_layer = dict(model.named_modules())[target_layer_name]
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam(image)
    return cam

def overlay_cam_on_image(image, cam):
    cam_resized = interpolate(torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                              size=image.size[::-1],
                              mode='bilinear',
                              align_corners=False)
    cam_resized = cam_resized.squeeze().numpy()
    image_np = np.array(image) / 255.0
    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
    overlay = (heatmap * 0.4 + image_np) / (1.0 + 0.4)
    overlay = np.clip(overlay, 0, 1)
    return overlay

def reconstruct_grad_cam_image(files, input_dir, model, transform, device, output_image_path, target_layer):
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

    for file, (x_start, y_start) in sub_images.items():
        sub_image_path = os.path.join(input_dir, file)
        sub_image = Image.open(sub_image_path).convert("RGB")
        processed_image = preprocess_image(sub_image, transform).to(device)

        cam = generate_cam(model, processed_image, device, target_layer)
        cam_overlay = overlay_cam_on_image(sub_image, cam)

        cam_overlay_pil = Image.fromarray((cam_overlay * 255).astype('uint8'))
        final_image.paste(cam_overlay_pil, (x_start, y_start))

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    final_image.save(output_image_path)
    return final_image

# Interface Streamlit
def main():
    st.title("Grad-CAM Reconstruction")
    st.sidebar.title("Options")

    model_path = st.sidebar.file_uploader("Charger le modèle PyTorch", type=["pth", "pt"])
    input_dir = st.sidebar.text_input("Dossier contenant les patchs", "patches/")
    output_dir = st.sidebar.text_input("Dossier de sortie", "output/")
    device = st.sidebar.selectbox("Choisir le périphérique", ["cpu", "cuda", "mps"], index=0)

    if not os.path.isdir(input_dir):
        st.error("Veuillez fournir un dossier valide contenant les patchs.")
        return

    model = load_model(model_path, device=device)
    if model:
        st.sidebar.success("Modèle chargé avec succès.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if st.button("Reconstruire l'image avec Grad-CAM"):
        all_files = os.listdir(input_dir)
        pattern = r'(.+)_\w+\.jpeg'
        prefixes = {re.match(pattern, f).group(1) for f in all_files if re.match(pattern, f)}
        grouped_images = [[file for file in all_files if file.startswith(prefix)] for prefix in prefixes]

        for group in grouped_images:
            group_name = re.match(pattern, group[0]).group(1)
            output_path = os.path.join(output_dir, f"{group_name}_grad_cam.jpeg")
            final_image = reconstruct_grad_cam_image(group, input_dir, model, transform, device, output_path, "features.18.0")
            st.image(final_image, caption=f"Image reconstruite : {group_name}", use_column_width=True)


if __name__ == "__main__":
    main()
