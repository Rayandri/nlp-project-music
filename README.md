# Classification de Paroles Musicales

Projet de traitement du langage naturel pour la classification et la génération de paroles de chansons en français.

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/Rayandri/nlp-project-music.git
cd nlp-project-music

# Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les ressources NLTK nécessaires
python download_nltk.py
```

## Structure du dataset

Le projet utilise un corpus de paroles de chansons françaises organisées selon cette structure:

```
lyrics_dataset/
├── [année]/
│   ├── [genre]/
│   │   ├── [album]-artist-[artiste]/
│   │   │   ├── [titre].txt
```

## Utilisation

### Interface Web

```bash
python web_app.py
```

Puis accédez à: http://127.0.0.1:5000

### Classification

Pour identifier l'artiste d'un texte:

```bash
# À partir d'un texte
python predict.py --text "J'ai demandé à la lune si tu voulais encore de moi"

# À partir d'un fichier
python predict.py --file chemin/vers/chanson.txt
```

### Analyse du dataset

```bash
python analyze_dataset.py
```

## Script principal (run.py)

Le script `run.py` est le cœur du projet et permet d'exécuter toutes les fonctionnalités principales avec différentes configurations.

### Modes d'exécution

```bash
# Classification de paroles
python run.py --mode classify

# Augmentation de données
python run.py --mode augment

# Interprétation des modèles
python run.py --mode interpret

# Génération de texte
python run.py --mode generate

# Validation croisée entre datasets
python run.py --mode cross_validate
```

### Options communes

```bash
# Spécifier le répertoire de sortie des résultats
python run.py --output_dir results_personnalises

# Définir une graine aléatoire pour la reproductibilité
python run.py --random_seed 42

# Définir le type de label à prédire
python run.py --label artiste

# Filtrer les artistes avec au moins N échantillons
python run.py --min_samples 5
```

### Exemples d'utilisation complète

```bash
# Classification avec TF-IDF et analyse des résultats
python run.py --mode classify --vectorizers tfidf --classifier logistic --confusion_matrix

# Augmentation de données avec diverses techniques
python run.py --mode augment --augmentation random_deletion random_swap synonym_replacement --augmentation_factor 0.5

# Interprétation d'un modèle pré-entraîné
python run.py --mode interpret --interpretation coefficients permutation --model_path results/models/best_artiste.pkl

# Génération de paroles dans le style d'un artiste
python run.py --mode generate --generator ngram --artist "Nekfeu" --max_length 200
```

### Génération des résultats pour le rapport

Pour générer toutes les données et visualisations du rapport en une seule commande:

```bash
bash generate_results.sh
```

## Options principales

### Méthodes de vectorisation
- `bow` - Bag of Words
- `tfidf` - Term Frequency-Inverse Document Frequency
- `word2vec` - Embeddings Word2Vec
- `fasttext` - Embeddings FastText
- `transformer` - Modèles Transformer

### Algorithmes de classification
- `logistic` - Régression logistique (défaut)
- `svm` - Support Vector Machine
- `random_forest` - Forêts aléatoires

### Types de labels
- `artiste` - Prédiction de l'artiste (défaut)
- `album` - Prédiction de l'album
- `genre` - Prédiction du genre musical
- `année` - Prédiction de la période

## Fonctionnalités principales

- Classification des paroles selon leur artiste
- Augmentation de données textuelles
- Interprétation des modèles
- Génération de nouvelles paroles
- Interface web pour tester les modèles

## Technologies utilisées

- scikit-learn
- NLTK
- Matplotlib/Seaborn
- Word2Vec/FastText
- Flask

## Auteurs
- Rayan Drissi
- Emre Ulusoy
- Marc Guillemot

