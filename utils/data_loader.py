import os
import re
import glob
from typing import List, Tuple, Dict, Optional

def load_lyrics_dataset(root_folder: str = "lyrics_dataset") -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Charge toutes les paroles du dataset avec leurs métadonnées

    Args:
        root_folder: Chemin vers le dossier racine contenant les paroles

    Returns:
        Tuple contenant:
            - liste des textes de paroles
            - liste des métadonnées pour chaque chanson (artiste, album, année, genre)
    """
    texts = []
    metadata_list = []

    for filepath in glob.glob(os.path.join(root_folder, '**', '*.txt'), recursive=True):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                # Extraire les métadonnées et les paroles
                metadata, lyrics = extract_metadata_and_lyrics(content)
                
                # Extraire des informations supplémentaires du chemin du fichier
                path_parts = filepath.split(os.path.sep)
                filename = os.path.basename(filepath)
                song_title = os.path.splitext(filename)[0]
                
                # Si le chemin a la structure expected (année/genre/album-artist-*)
                if len(path_parts) >= 4:
                    year_range = path_parts[-4] if len(path_parts) >= 4 else "Unknown"
                    genre = path_parts[-3] if len(path_parts) >= 3 else "Unknown"
                    album_artist_info = path_parts[-2] if len(path_parts) >= 2 else ""
                    
                    # Extraction de l'album et de l'artiste depuis le nom du dossier
                    album = album_artist_info.split('-artist-')[0] if '-artist-' in album_artist_info else album_artist_info
                    artist_from_path = album_artist_info.split('-artist-')[1] if '-artist-' in album_artist_info else "Unknown"
                    
                    # Si l'artiste n'est pas dans les métadonnées, utiliser celui du chemin
                    if "artiste" not in metadata or not metadata["artiste"]:
                        metadata["artiste"] = artist_from_path
                    
                    # Compléter les métadonnées avec les informations du chemin
                    metadata.update({
                        "titre": song_title,
                        "album": album,
                        "année": year_range,
                        "genre": genre
                    })
                
                # Ajouter les paroles et les métadonnées aux listes
                texts.append(lyrics)
                metadata_list.append(metadata)
                
        except Exception as e:
            print(f"Erreur lors du chargement de {filepath}: {e}")
    
    print(f"{len(texts)} chansons chargées avec succès.")
    return texts, metadata_list

def extract_metadata_and_lyrics(content: str) -> Tuple[Dict[str, str], str]:
    """
    Extrait les métadonnées et les paroles à partir du contenu du fichier

    Args:
        content: Contenu complet du fichier texte

    Returns:
        Tuple contenant:
            - dictionnaire des métadonnées
            - texte des paroles
    """
    metadata = {}
    lines = content.split('\n')
    
    # Trouver la fin des métadonnées (première ligne vide)
    header_end = 0
    for i, line in enumerate(lines):
        if not line.strip():
            header_end = i
            break
    
    # Extraire les métadonnées (si présentes)
    if header_end > 0:
        for i in range(header_end):
            line = lines[i].strip()
            if ':' in line:
                key, value = line.split(':', 1)
                # Convertir la clé en minuscules pour uniformisation
                metadata[key.strip().lower()] = value.strip()
    
    # Le reste est considéré comme les paroles
    lyrics = '\n'.join(lines[header_end:]).strip()
    
    return metadata, lyrics

def save_tokenized_lyrics(texts: List[List[str]], metadata_list: List[Dict[str, str]], 
                         output_dir: str = "tokenized_lyrics_dataset") -> None:
    """
    Sauvegarde les paroles tokenisées dans une structure de dossiers similaire à l'originale

    Args:
        texts: Liste des paroles tokenisées
        metadata_list: Liste des métadonnées correspondantes
        output_dir: Dossier de sortie
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (tokens, metadata) in enumerate(zip(texts, metadata_list)):
        # Recréer la structure de dossiers
        year_range = metadata.get("année", "Unknown")
        genre = metadata.get("genre", "Unknown")
        album = metadata.get("album", "Unknown")
        artist = metadata.get("artiste", "Unknown")
        title = metadata.get("titre", f"song_{i}")
        
        # Créer le chemin de sortie
        album_dir = f"{album}-artist-{artist}"
        output_path = os.path.join(output_dir, year_range, genre, album_dir)
        os.makedirs(output_path, exist_ok=True)
        
        # Joindre les tokens en une chaîne
        tokenized_text = " ".join(tokens)
        
        # Sauvegarder dans un fichier
        with open(os.path.join(output_path, f"{title}.txt"), "w", encoding="utf-8") as f:
            f.write(tokenized_text)
        
    print(f"{len(texts)} fichiers tokenisés sauvegardés dans {output_dir}")

def get_label_from_metadata(metadata_list: List[Dict[str, str]], 
                          label_type: str = "artiste") -> List[str]:
    """
    Extrait une liste d'étiquettes à partir des métadonnées

    Args:
        metadata_list: Liste des métadonnées
        label_type: Type d'étiquette à extraire (artiste, album, genre, année)

    Returns:
        Liste des étiquettes
    """
    return [metadata.get(label_type, "Unknown") for metadata in metadata_list] 
