# Classification de paroles de chansons francophones

Ce projet analyse un corpus de paroles de chansons francophones et utilise des techniques de NLP pour prédire des métadonnées comme l'artiste, l'album, le genre ou l'année à partir du texte des paroles.

## Structure du projet

```
.
├── lyrics_dataset/            # Données brutes (paroles avec métadonnées) 
├── tokenized_lyrics_dataset/  # Données tokenisées (généré automatiquement)
├── vectors/                   # Représentations vectorielles (généré automatiquement)
├── utils/                     # Modules utilitaires
│   ├── tokenizer.py           # Tokenizer BPE (Byte Pair Encoding)
│   ├── data_loader.py         # Chargement et manipulation des données
│   ├── vectorizers.py         # Différentes méthodes de vectorisation
│   └── models.py              # Modèles de classification et évaluation
└── main.py                    # Script principal
```

## Installation

1. Clonez le dépôt :
```bash
git clone <URL_du_repo>
cd nlp-project-music
```

2. Créez et activez un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Installez le modèle français spaCy :
```bash
python -m spacy download fr_core_news_sm
```

## Utilisation

Le script principal `main.py` offre plusieurs options pour contrôler le processus d'analyse :

### Exécution complète (tokenisation + classification)

```bash
python main.py --input_dir lyrics_dataset --output_dir tokenized_lyrics_dataset
```

### Tokenisation uniquement

```bash
python main.py --mode tokenize
```

### Classification uniquement (sans retokeniser)

```bash
python main.py --mode classify
```

### Choisir l'étiquette à prédire

```bash
python main.py --label artiste  # Options: artiste, album, genre, année
```

### Choisir les méthodes de vectorisation

```bash
python main.py --vectorizers tfidf transformer  # Options: bow, tfidf, word2vec, fasttext, transformer, all
```

### Sauvegarder les vecteurs générés

```bash
python main.py --save_vectors
```

## Méthodes d'embedding implémentées

1. **Bag-of-Words (BOW)** : Représentation simple des fréquences des mots
2. **TF-IDF** : Importance des mots dans le document par rapport au corpus
3. **Word2Vec** : Embeddings sémantiques (Gensim)
4. **FastText** : Embeddings tenant compte des sous-mots (Gensim)
5. **Transformer** : Utilisation du modèle multilingue `paraphrase-multilingual-MiniLM-L12-v2` (SentenceTransformers)

## Pipeline de traitement

1. **Chargement des données** : Les paroles sont extraites des fichiers texte avec leurs métadonnées.
2. **Tokenisation** : Les textes sont tokenisés avec un tokenizer BPE (Byte Pair Encoding).
3. **Vectorisation** : Les textes tokenisés sont convertis en représentations vectorielles.
4. **Classification** : Un modèle de régression logistique est entraîné pour prédire l'étiquette cible.
5. **Évaluation** : Les performances du modèle sont évaluées et visualisées.

## Exemples de résultats

Le script génère des évaluations détaillées pour chaque méthode d'embedding :
- Accuracy et scores F1
- Matrices de confusion
- Comparaison des performances entre les différentes méthodes
