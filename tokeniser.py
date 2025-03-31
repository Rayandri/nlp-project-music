import os
import re
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def tokenize_text(text):
    """
    Tokenise le texte en français :
      - Segmentation en phrases puis en mots
      - Conversion en minuscules
      - Suppression de la ponctuation
      - Suppression des stopwords français
    Renvoie la liste des tokens.
    """
    text = text.replace("j'", "je ")
    text = text.replace("l'", "le ")
    text = text.replace("d'", "de ")
    text = text.replace("c'", "ce ")
    text = text.replace("s'", "se ")
    text = text.replace("n'", "ne ")
    text = text.replace("t'", "te ")
    text = text.replace("qu'", "que ")
    text = text.replace("m'", "me ")
    text = text.replace("'", " ")
    sentences = sent_tokenize(text, language='french')
    tokens = []
    french_stopwords = set(stopwords.words('french'))
    
    for sentence in sentences:
        words = word_tokenize(sentence, language='french')
        for word in words:
            word = word.lower()
            if word not in string.punctuation and word not in french_stopwords:
                tokens.append(word)
    return tokens

input_root = "lyrics_dataset"
output_root = "tokenized_lyrics_dataset"

for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            tokens = tokenize_text(text)
            tokenized_text = " ".join(tokens)
            
            relative_path = os.path.relpath(root, input_root)
            output_dir = os.path.join(output_root, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            output_file_path = os.path.join(output_dir, file)
            
            with open(output_file_path, "w", encoding="utf-8") as out_f:
                out_f.write(tokenized_text)
            
            print("Fichier tokenisé enregistré :", output_file_path)
