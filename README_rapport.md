# Instructions pour le rapport du projet NLP

Ce document contient les instructions pour générer les données et compiler le rapport LaTeX requis pour le projet NLP.

## Checklist d'évaluation du projet

Selon les exigences définies, votre rapport doit couvrir les points suivants :

- [x] **Présentation du jeu de données** (2.5 pts)
  - Description des sources, structure et caractéristiques
  - Statistiques descriptives
  - Particularités et défis

- [x] **Pré-traitement et analyse exploratoire** (2.5 pts)
  - Normalisation et tokenisation
  - Analyse de la distribution des tokens, classes, documents
  - Statistiques descriptives sur les documents

- [x] **Benchmark sur les modèles de classification** (5 pts)
  - Bayésien naïf
  - Régression logistique
  - TF-IDF
  - Word2Vec
  - Réseaux de neurones feedforward
  - Réseaux de neurones récurrents
  - Transformer

- [x] **Benchmark sur les modèles de génération de texte** (5 pts)
  - N-grams
  - TF-IDF et Word2Vec
  - Réseaux neuronaux feedforward
  - Réseaux neuronaux récurrents
  - Transformer

- [x] **Trois approches avancées explorées en profondeur** (7.5 pts)
  - Augmentation de données (au moins 3 méthodes)
  - Interprétation des modèles (au moins 2 méthodes)
  - Transfert de connaissances entre jeux de données

- [x] **Clarté, organisation et analyse critique** (5 pts)
  - Présentation claire des objectifs
  - Mise en page soignée
  - Analyse des limites et difficultés
  - Propositions d'amélioration

## Instructions pour générer les données

1. Assurez-vous que votre environnement Python est correctement configuré avec toutes les dépendances nécessaires :
   ```bash
   pip install -r requirements.txt
   ```

2. Exécutez le script bash pour générer toutes les données nécessaires au rapport :
   ```bash
   bash generate_results.sh
   ```
   Ce script va exécuter les différentes analyses et générer les résultats dans le dossier `results_rapport/`.

## Instructions pour compiler le rapport LaTeX

1. Modifiez le fichier `structure_rapport.tex` pour y inclure vos résultats et analyses spécifiques. La structure de base est déjà fournie et suit les exigences du projet.

2. Compilez le rapport LaTeX (avec les références) :
   ```bash
   pdflatex structure_rapport.tex
   pdflatex structure_rapport.tex  # Une seconde fois pour les références
   ```

3. Le rapport final sera généré sous forme de fichier PDF : `structure_rapport.pdf`

## Structure du rapport

Le rapport est structuré selon les exigences, avec les sections suivantes :

1. **Introduction** - Contexte et objectifs du projet
2. **Présentation du jeu de données** - Description et statistiques
3. **Prétraitement et analyse exploratoire** - Normalisation, tokenisation, et analyse
4. **Modèles de classification** - Description et comparaison des approches
5. **Modèles de génération de texte** - Description et évaluation
6. **Approches avancées explorées** - Augmentation de données, interprétation des modèles, transfert de connaissances
7. **Conclusion et perspectives** - Synthèse, limites et améliorations

## Conseils pour l'évaluation

- Assurez-vous que votre rapport ne dépasse pas 10 pages
- Suivez exactement la structure définie
- Incluez des visualisations claires et pertinentes
- Analysez de manière critique les résultats
- Identifiez clairement les limites et proposez des améliorations
- La date limite pour la soumission du rapport est le 12/05
- La présentation orale aura lieu le 19/05 
