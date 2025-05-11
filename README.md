# Classification de Paroles Musicales

Ce projet utilise des techniques de NLP pour classifier des paroles de chansons selon différents critères (artiste, album, genre, année).

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/RayanDri/nlp-project-music.git
cd nlp-project-music

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation rapide

Pour lancer la classification avec les paramètres optimaux:

```bash
python run.py
```

Ce script utilise les meilleurs paramètres déterminés empiriquement:
- Filtrage des classes avec moins de 5 exemples
- Conservation des 15 artistes les plus fréquents
- Tokenisation BPE avec 1500 fusions
- Suppression des mots vides
- Vectorisation Bag-of-Words (BOW)
- Classification par régression logistique avec équilibrage des classes

## Options personnalisées

Pour personnaliser les paramètres interactivement:

```bash
python run.py --mode custom
```

Pour spécifier directement les paramètres:

```bash
python main.py --min_samples 5 --top_classes 10 --vectorizers bow --classifier svm
```

## Structure du projet

- `main.py`: Script principal pour la tokenisation et classification
- `run.py`: Interface simplifiée pour lancer le projet
- `analyze_dataset.py`: Utilitaire pour analyser la distribution des classes
- `utils/`: Dossier contenant les modules utilitaires
  - `tokenizer.py`: Implémentation du tokenizer BPE
  - `data_loader.py`: Chargement des données et extraction des métadonnées
  - `vectorizers.py`: Différentes méthodes de vectorisation de texte
  - `models.py`: Modèles de classification et évaluation

## Fonctionnalités

- **Tokenisation BPE**: Apprentissage de sous-mots spécifiques au corpus
- **Vectorisation**: Plusieurs méthodes (BOW, TF-IDF, Word2Vec, FastText, Transformer)
- **Classification**: Différents algorithmes (Régression logistique, SVM, Random Forest)
- **Analyse de données**: Outils pour comprendre la distribution des classes
- **Évaluation**: Matrices de confusion, F1-score, précision, validation croisée

## Résultats

Les meilleurs résultats ont été obtenus avec:
- Vectorisation: Bag-of-Words (BOW)
- Classification: Régression logistique avec équilibrage des classes
- Précision: ~60% sur la classification des 15 artistes les plus fréquents

## Améliorations possibles

- Augmentation des données pour les classes sous-représentées
- Extraction de caractéristiques spécifiques aux paroles (rimes, structure)
- Modèles plus complexes comme BERT fine-tuné sur le français
- Approche hiérarchique (classification par genre puis par artiste)
