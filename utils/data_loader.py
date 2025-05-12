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

def combine_datasets(dataset_dirs: List[str], min_overlap_classes: int = 5) -> Tuple[List[str], List[Dict]]:
    """
    Combine multiple datasets and keep only classes that appear in at least N datasets
    
    Args:
        dataset_dirs: List of directories containing datasets
        min_overlap_classes: Minimum number of datasets a class must appear in
        
    Returns:
        Tuple of (combined_texts, combined_metadata)
    """
    all_texts = []
    all_metadata = []
    
    # Load all datasets
    for dataset_dir in dataset_dirs:
        texts, metadata = load_lyrics_dataset(dataset_dir)
        
        # Store dataset source in metadata
        for m in metadata:
            m['dataset_source'] = dataset_dir
            
        all_texts.extend(texts)
        all_metadata.extend(metadata)
        
        print(f"Loaded {len(texts)} songs from {dataset_dir}")
    
    # Count class occurrences across datasets if we have more than one dataset
    if len(dataset_dirs) > 1 and min_overlap_classes > 1:
        # Get list of unique sources
        sources = set(m['dataset_source'] for m in all_metadata)
        
        # Count artist occurrences in each source
        artist_sources = {}
        for i, m in enumerate(all_metadata):
            artist = m.get('artist', 'unknown')
            if artist not in artist_sources:
                artist_sources[artist] = set()
            artist_sources[artist].add(m['dataset_source'])
        
        # Filter to keep only classes that appear in multiple datasets
        keep_indices = []
        for i, m in enumerate(all_metadata):
            artist = m.get('artist', 'unknown')
            if len(artist_sources[artist]) >= min_overlap_classes:
                keep_indices.append(i)
                
        # Filter texts and metadata
        filtered_texts = [all_texts[i] for i in keep_indices]
        filtered_metadata = [all_metadata[i] for i in keep_indices]
        
        print(f"Kept {len(filtered_texts)} songs with artists appearing in at least {min_overlap_classes} datasets")
        
        return filtered_texts, filtered_metadata
    
    return all_texts, all_metadata

def cross_dataset_validation(dataset_dirs: List[str], vectorizer_type: str = "tfidf", 
                           classifier_type: str = "logistic", label_type: str = "artiste"):
    """
    Perform cross-dataset validation (train on one dataset, test on another)
    
    Args:
        dataset_dirs: List of directories containing datasets
        vectorizer_type: Type of vectorizer to use
        classifier_type: Type of classifier to use
        label_type: Type of label to use
        
    Returns:
        Dictionary of results
    """
    from utils.vectorizers import TextVectorizer
    from utils.models import TextClassifier
    from sklearn.metrics import accuracy_score, f1_score
    
    if len(dataset_dirs) < 2:
        raise ValueError("Need at least 2 datasets for cross-dataset validation")
    
    results = {}
    
    # For each combination of train/test datasets
    for train_idx, train_dir in enumerate(dataset_dirs):
        for test_idx, test_dir in enumerate(dataset_dirs):
            if train_idx == test_idx:
                continue  # Skip same dataset
                
            key = f"{train_dir}_to_{test_dir}"
            print(f"\nValidation croisée: entraînement sur {train_dir}, test sur {test_dir}")
            
            # Load datasets
            train_texts, train_metadata = load_lyrics_dataset(train_dir)
            test_texts, test_metadata = load_lyrics_dataset(test_dir)
            
            # Get labels
            train_labels = get_label_from_metadata(train_metadata, label_type)
            test_labels = get_label_from_metadata(test_metadata, label_type)
            
            # Find common classes
            train_classes = set(train_labels)
            test_classes = set(test_labels)
            common_classes = train_classes.intersection(test_classes)
            
            print(f"Classes communes: {len(common_classes)} sur {len(train_classes)} (train) et {len(test_classes)} (test)")
            
            if len(common_classes) < 2:
                print("Pas assez de classes communes, on passe cette combinaison")
                results[key] = {"error": "Not enough common classes"}
                continue
                
            # Filter to keep only common classes
            train_indices = [i for i, label in enumerate(train_labels) if label in common_classes]
            test_indices = [i for i, label in enumerate(test_labels) if label in common_classes]
            
            train_filtered_texts = [train_texts[i] for i in train_indices]
            train_filtered_labels = [train_labels[i] for i in train_indices]
            test_filtered_texts = [test_texts[i] for i in test_indices]
            test_filtered_labels = [test_labels[i] for i in test_indices]
            
            # Vectorize
            vectorizer = TextVectorizer(method=vectorizer_type)
            X_train = vectorizer.fit_transform(train_filtered_texts)
            X_test = vectorizer.transform(test_filtered_texts)
            
            # Train classifier
            classifier = TextClassifier(model_type=classifier_type)
            clf = classifier.model
            clf.fit(X_train, train_filtered_labels)
            
            # Predict and evaluate
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(test_filtered_labels, y_pred)
            macro_f1 = f1_score(test_filtered_labels, y_pred, average='macro')
            weighted_f1 = f1_score(test_filtered_labels, y_pred, average='weighted')
            
            print(f"Précision: {accuracy:.3f}")
            print(f"F1-score macro: {macro_f1:.3f}")
            
            results[key] = {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "train_size": len(train_filtered_texts),
                "test_size": len(test_filtered_texts),
                "common_classes": len(common_classes)
            }
    
    return results 
