import os
import re
import glob
from typing import List, Tuple, Dict, Optional

def load_lyrics_dataset(root_folder: str = "lyrics_dataset") -> Tuple[List[str], List[Dict[str, str]]]:
    texts = []
    metadata_list = []

    for filepath in glob.glob(os.path.join(root_folder, '**', '*.txt'), recursive=True):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                metadata, lyrics = extract_metadata_and_lyrics(content)
                
                path_parts = filepath.split(os.path.sep)
                filename = os.path.basename(filepath)
                song_title = os.path.splitext(filename)[0]
                
                if len(path_parts) >= 4:
                    year_range = path_parts[-4] if len(path_parts) >= 4 else "Unknown"
                    genre = path_parts[-3] if len(path_parts) >= 3 else "Unknown"
                    album_artist_info = path_parts[-2] if len(path_parts) >= 2 else ""
                    
                    album = album_artist_info.split('-artist-')[0] if '-artist-' in album_artist_info else album_artist_info
                    artist_from_path = album_artist_info.split('-artist-')[1] if '-artist-' in album_artist_info else "Unknown"
                    
                    if "artiste" not in metadata or not metadata["artiste"]:
                        metadata["artiste"] = artist_from_path
                    
                    metadata.update({
                        "titre": song_title,
                        "album": album,
                        "année": year_range,
                        "genre": genre
                    })
                
                texts.append(lyrics)
                metadata_list.append(metadata)
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    print(f"{len(texts)} songs loaded successfully.")
    return texts, metadata_list

def extract_metadata_and_lyrics(content: str) -> Tuple[Dict[str, str], str]:
    metadata = {}
    lines = content.split('\n')
    
    header_end = 0
    for i, line in enumerate(lines):
        if not line.strip():
            header_end = i
            break
    
    if header_end > 0:
        for i in range(header_end):
            line = lines[i].strip()
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip().lower()] = value.strip()
    
    lyrics = '\n'.join(lines[header_end:]).strip()
    
    return metadata, lyrics

def save_tokenized_lyrics(texts: List[List[str]], metadata_list: List[Dict[str, str]], 
                         output_dir: str = "tokenized_lyrics_dataset") -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (tokens, metadata) in enumerate(zip(texts, metadata_list)):
        year_range = metadata.get("année", "Unknown")
        genre = metadata.get("genre", "Unknown")
        album = metadata.get("album", "Unknown")
        artist = metadata.get("artiste", "Unknown")
        title = metadata.get("titre", f"song_{i}")
        
        album_dir = f"{album}-artist-{artist}"
        output_path = os.path.join(output_dir, year_range, genre, album_dir)
        os.makedirs(output_path, exist_ok=True)
        
        tokenized_text = " ".join(tokens)
        
        with open(os.path.join(output_path, f"{title}.txt"), "w", encoding="utf-8") as f:
            f.write(tokenized_text)
        
    print(f"{len(texts)} tokenized files saved in {output_dir}")

def get_label_from_metadata(metadata_list: List[Dict[str, str]], 
                          label_type: str = "artiste") -> List[str]:
    return [metadata.get(label_type, "Unknown") for metadata in metadata_list] 
