# Classification de Paroles Musicales

Ce projet utilise des techniques de NLP pour classifier des paroles de chansons selon différents critères (artiste, album, genre, année).

## Guide de démarrage rapide

### Étape 1: Installation

```bash
# Cloner le dépôt
git clone https://github.com/Rayandri/nlp-project-music.git
cd nlp-project-music

# Créer et activer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le modèle spaCy français
python -m spacy download fr_core_news_sm
```

### Étape 2: Préparer les données

Le projet nécessite des paroles de chansons françaises organisées selon une structure spécifique :

```
lyrics_dataset/
├── [année]/
│   ├── [genre]/
│   │   ├── [album]-artist-[artiste]/
│   │   │   ├── [titre].txt
```

#### Sources de données

Vous avez plusieurs options pour obtenir des paroles:

1. **APIs de paroles**: Plusieurs services proposent des APIs pour récupérer des paroles:
   - [Genius API](https://docs.genius.com/) (gratuit avec inscription)
   - [Musixmatch API](https://developer.musixmatch.com/) (gratuit avec limitations)

2. **Datasets publics**: Certains ensembles de données de recherche sont disponibles:
   - [MTG-Jamendo Dataset](https://mtg.github.io/mtg-jamendo-dataset/) (paroles multilingues)
   - Datasets sur Kaggle comme "French Song Lyrics"

3. **Web scraping**: Script pour récupérer les paroles de sites spécialisés (attention aux droits d'auteur)

4. **Exemples de test**: Pour un test rapide, vous pouvez créer quelques fichiers manuellement:

```bash
# Créer une structure minimale pour tester
mkdir -p lyrics_dataset/2000s/pop/album-artist-Indochine
echo "J'ai demandé à la lune si tu voulais encore de moi, elle m'a dit oui mais toi je ne sais pas" > lyrics_dataset/2000s/pop/album-artist-Indochine/lune.txt

mkdir -p lyrics_dataset/2010s/rap/album-artist-Nekfeu
echo "On écrit l'histoire sans effaceur, le temps est compté comme dans Tetris" > lyrics_dataset/2010s/rap/album-artist-Nekfeu/onpark.txt

mkdir -p lyrics_dataset/1980s/chanson/album-artist-JacquesBrel
echo "Ne me quitte pas, il faut oublier, tout peut s'oublier" > lyrics_dataset/1980s/chanson/album-artist-JacquesBrel/nequittepas.txt
```

Pour un dataset plus complet, vous devrez soit collecter vous-même les paroles, soit utiliser une des sources ci-dessus.

### Étape 3: Charger les données

Avant de lancer l'application ou d'entraîner un modèle, il faut charger les données:

```bash
python data_loader.py
```

Cela préparera vos fichiers de paroles pour l'utilisation dans les étapes suivantes.

### Étape 4: Lancer l'application

Plusieurs options s'offrent à vous :

#### Interface Web (recommandée)

```bash
python web_app.py
```
Ensuite, ouvrez votre navigateur à l'adresse : http://127.0.0.1:5000

#### Entraîner le modèle avec les paramètres optimaux

```bash
python run.py
```

#### Prédire l'artiste d'une chanson

```bash
# À partir d'un texte
python predict.py --text "J'ai demandé à la lune si tu voulais encore de moi"

# À partir d'un fichier
python predict.py --file chemin/vers/chanson.txt
```

## Guide complet de main.py

Le script `main.py` est le cœur du projet et offre de nombreuses fonctionnalités avancées. Voici un guide détaillé:

### Modes d'exécution

```bash
# Mode "best" - utilise les meilleurs paramètres
python main.py --mode best

# Mode "tokenize" - uniquement tokeniser le dataset
python main.py --mode tokenize

# Mode "classify" - utiliser des tokens existants pour la classification
python main.py --mode classify

# Mode "all" - tokenisation puis classification (défaut)
python main.py --mode all
```

### Options de classification

```bash
# Changer le type de label à prédire
python main.py --label artiste
python main.py --label album
python main.py --label genre
python main.py --label année

# Filtrer les classes par nombre d'exemples minimum
python main.py --min_samples 5

# Limiter aux N classes les plus fréquentes
python main.py --top_classes 15

# Définir la graine aléatoire pour la reproductibilité
python main.py --random_seed 42
```

### Options de tokenisation

```bash
# Modifier le nombre de fusions BPE
python main.py --bpe_merges 1000

# Activer la suppression des mots vides
python main.py --use_stopwords

# Spécifier les dossiers d'entrée/sortie
python main.py --input_dir mon_dataset --output_dir mes_tokens
```

### Méthodes de vectorisation

```bash
# Utiliser une méthode spécifique
python main.py --vectorizers bow
python main.py --vectorizers tfidf
python main.py --vectorizers word2vec
python main.py --vectorizers fasttext
python main.py --vectorizers transformer

# Combiner plusieurs méthodes
python main.py --vectorizers bow tfidf

# Tester toutes les méthodes
python main.py --vectorizers all
```

### Algorithmes de classification

```bash
# Régression logistique (défaut)
python main.py --classifier logistic

# Machine à vecteurs de support
python main.py --classifier svm

# Forêts aléatoires
python main.py --classifier random_forest
```

### Réduction de dimensionnalité et visualisation

```bash
# Appliquer une réduction PCA
python main.py --pca 100

# Générer une matrice de confusion
python main.py --confusion_matrix

# Sauvegarder les vecteurs pour analyse externe
python main.py --save_vectors
```

### Entraînement et sauvegarde de modèles

```bash
# Sauvegarder les modèles entraînés pour prédiction future
python main.py --save_models models_dir
```

### Exemple complet avancé

```bash
python main.py \
  --mode all \
  --label artiste \
  --min_samples 5 \
  --top_classes 10 \
  --vectorizers bow \
  --classifier svm \
  --bpe_merges 2000 \
  --use_stopwords \
  --pca 200 \
  --confusion_matrix \
  --save_models my_models
```

Cette commande va:
1. Filtrer le dataset pour garder les 10 artistes avec au moins 5 chansons chacun
2. Tokeniser les paroles avec 2000 fusions BPE et suppression des mots vides
3. Vectoriser avec Bag-of-Words et réduire à 200 dimensions par PCA
4. Classifier avec SVM
5. Générer une matrice de confusion et sauvegarder le modèle

## Options avancées

### Personnalisation des paramètres

```bash
# Interface interactive
python run.py --mode custom

# Paramètres spécifiques
python main.py --min_samples 5 --top_classes 10 --vectorizers bow --classifier svm
```

### Analyse du dataset

```bash
python analyze_dataset.py
```
## Structure du projet

- `main.py`: Script principal pour la tokenisation et classification
- `run.py`: Interface simplifiée pour lancer le projet
- `predict.py`: Script pour prédire l'artiste d'une nouvelle chanson
- `web_app.py`: Interface web pour tester le classifieur
- `analyze_dataset.py`: Utilitaire pour analyser la distribution des classes
- `utils/`: Dossier contenant les modules utilitaires
  - `tokenizer.py`: Implémentation du tokenizer BPE
  - `data_loader.py`: Chargement des données et extraction des métadonnées
  - `vectorizers.py`: Différentes méthodes de vectorisation de texte
  - `models.py`: Modèles de classification et évaluation

## Performances

Les meilleurs résultats sont obtenus avec:
- Vectorisation: Bag-of-Words (BOW)
- Classification: Régression logistique avec équilibrage des classes
- Précision: ~60% sur la classification des 15 artistes les plus fréquents

## Améliorations possibles

- Augmentation des données pour les classes sous-représentées
- Extraction de caractéristiques spécifiques aux paroles (rimes, structure)
- Modèles plus complexes comme BERT fine-tuné sur le français
- Approche hiérarchique (classification par genre puis par artiste)

