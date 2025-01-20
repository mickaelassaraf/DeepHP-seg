from PIL import Image
import re
import os

def reconstruct_image(files, input_dir, output_image_path):
    """
    Reconstruit une image à partir d'une liste de fichiers.

    :param files: Liste des fichiers contenant les sous-images.
    :param input_dir: Répertoire contenant les fichiers.
    :param output_image_path: Chemin pour sauvegarder l'image finale.
    """
    pattern = r"_(\d{1,8})x(\d{1,8})"
    x_max, y_max = 0, 0
    sub_images = {}
    c=0
    # Extraire les coordonnées et vérifier la validité des blocs
    for file in files:
        match = re.search(pattern, file)
        if match:
            y_start_y_end,x_start_x_end= map(str, match.groups())
            for i in range(1,len(x_start_x_end)):
                x_start = int(x_start_x_end[:i])
                x_end = int(x_start_x_end[i:])
                if x_end-x_start==256:
                    break
            for i in range(1,len(y_start_y_end)):
                y_start = int(y_start_y_end[:i])
                y_end = int(y_start_y_end[i:])
                if y_end-y_start==256:
                    break

            x_max = max(x_max, x_end)
            y_max = max(y_max, y_end)
            sub_images[file] = (x_start, y_start)
        else:
            print(f"Le fichier {file} ne correspond pas au format attendu.")    
    # Créer une image vide pour la reconstruction
    final_image = Image.new('RGB', (x_max, y_max))
    # Assembler l'image
    for file, (x_start, y_start) in sub_images.items():
        sub_image_path = os.path.join(input_dir, file)
        sub_image = Image.open(sub_image_path)
        final_image.paste(sub_image, (x_start, y_start))

    # Créer le répertoire cible si nécessaire
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    # Sauvegarder l'image finale
    final_image.save(output_image_path)
    print(f"Image reconstruite sauvegardée à : {output_image_path}")


# Chemins
input_dir = 'Experiment-676'
output_base_path = 'data/full_image'

# Lister les fichiers
all_files = os.listdir(input_dir)

# Grouper les fichiers par préfixe commun
pattern = r'(.+)_\w+\.jpeg'
prefixes = {re.match(pattern, f).group(1) for f in all_files if re.match(pattern, f)}
grouped_images = [[file for file in all_files if file.startswith(prefix)] for prefix in prefixes]
# Appliquer la fonction de reconstruction à chaque groupe
for group in grouped_images:
    group_name = re.match(pattern, group[0]).group(1)  # Utiliser le préfixe du groupe
    print(group)
    output_path = os.path.join(output_base_path, f"{group_name}_reconstructed.jpeg")
    reconstruct_image(group, input_dir, output_path)

