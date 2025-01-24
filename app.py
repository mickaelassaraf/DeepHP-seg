import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image, ImageFile
import os
from sklearn.metrics import precision_score, f1_score, accuracy_score
from pathlib import Path

# Activer le chargement d'images tronquées
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Charger le modèle
@st.cache_resource
def load_model(model_path, device="cuda"):
    if not model_path:
        st.error("Veuillez uploader un modèle.")
        return None

    model = models.mobilenet_v2(pretrained=True)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None
    model.to(device)
    model.eval()
    return model

# Préparer l'image
def preprocess_image(image, transform):
    return transform(image).unsqueeze(0)

# Effectuer une prédiction
def predict(model, image, device="cuda"):
    if not model:
        st.error("Modèle non chargé.")
        return None
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
    return pred.item()

# Évaluer un dossier
def evaluate_folder(model, folder_path, transform, device="cuda"):
    true_labels = []
    predictions = []
    images = []

    class_names = sorted(os.listdir(folder_path))
    class_mapping = {name: idx for idx, name in enumerate(class_names)}

    for class_folder in class_names:
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                try:
                    image = Image.open(image_path).convert("RGB")
                    processed_image = preprocess_image(image, transform)
                    pred = predict(model, processed_image, device)
                    predictions.append(pred)
                    true_labels.append(class_mapping[class_folder])
                    images.append((image, pred, class_mapping[class_folder]))
                except Exception as e:
                    st.error(f"Erreur avec l'image {image_name}: {e}")

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)

    return accuracy, precision, f1, images

# Interface Streamlit
def main():
    st.title("Demo d'évaluation de modèle")
    st.sidebar.title("Options")

    # Charger le modèle
    model_path = st.sidebar.file_uploader("Charger le modèle PyTorch", type=["pth", "pt"])
    device = st.sidebar.selectbox("Choisir le périphérique", ["cpu", "cuda", "mps"], index=0)
    model = load_model(model_path, device=device)

    if model:
        st.sidebar.success("Modèle chargé avec succès.")

    # Transformation des images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Section pour une image unique
    st.header("Tester une image")
    uploaded_image = st.file_uploader("Uploader une image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Image chargée", use_column_width=True)
            processed_image = preprocess_image(image, transform).to(device)
            pred = predict(model, processed_image, device)
            if pred is not None:
                st.write(f"Prédiction : {pred}")
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image : {e}")

    # Section pour un dossier
    st.header("Évaluer un dossier")
    folder_path = st.text_input("Chemin vers le dossier contenant les images (par classe)")
    if folder_path and os.path.isdir(folder_path):
        if st.button("Évaluer le dossier"):
            st.write("Évaluation en cours...")
            accuracy, precision, f1, images = evaluate_folder(model, folder_path, transform, device)
            st.write(f"Précision : {precision:.4f}, F1 Score : {f1:.4f}, Accuracy : {accuracy:.4f}")
            st.subheader("Résultats par image")
            for img, pred, label in images[:10]:  # Afficher les 10 premières images
                st.image(img, caption=f"Prédiction: {pred}, Vérité: {label}")

if __name__ == "__main__":
    main()
