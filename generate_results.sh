#!/bin/bash

# Script pour générer toutes les données nécessaires au rapport de projet NLP
# Execution: bash generate_results.sh

export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32

# Création des répertoires pour les résultats
RESULTS_DIR="results_rapport"
FIGURES_DIR="figures_rapport"

mkdir -p $RESULTS_DIR
mkdir -p $FIGURES_DIR

echo "=== Génération des données pour le rapport ==="
echo "Résultats sauvegardés dans: $RESULTS_DIR"
echo "Figures sauvegardées dans: $FIGURES_DIR"

# Activation de l'environnement virtuel (décommenter si nécessaire)
# source venv/bin/activate

# 1. Analyse du jeu de données
echo -e "\n=== 1. Analyse exploratoire des données ==="
python analyze_dataset.py --output_dir $RESULTS_DIR

# 2. Classification: benchmark des différents modèles
echo -e "\n=== 2. Benchmark des modèles de classification ==="
python run.py --mode classify --vectorizers bow tfidf word2vec fasttext transformer --output_dir $RESULTS_DIR

# 3. Génération: benchmark des différents modèles de génération
echo -e "\n=== 3. Benchmark des modèles de génération ==="
python run.py --mode generate --generator all --output_dir $RESULTS_DIR

# 4. Augmentation de données: évaluation de l'impact
echo -e "\n=== 4. Évaluation de l'augmentation de données ==="
python run.py --mode augment --augmentation random_deletion random_swap random_insertion synonym_replacement --output_dir $RESULTS_DIR

# 5. Interprétation des modèles
echo -e "\n=== 5. Interprétation des modèles ==="
python run.py --mode interpret --interpretation coefficients permutation --output_dir $RESULTS_DIR

# 6. Cross-validation entre datasets (si disponible)
if [ -d "lyrics_dataset2" ]; then
    echo -e "\n=== 6. Validation croisée entre datasets ==="
    python run.py --mode cross_validate --dataset_dirs lyrics_dataset lyrics_dataset2 --output_dir $RESULTS_DIR
fi

# 7. Génération de figures supplémentaires (optionnel)
echo -e "\n=== 7. Génération de figures supplémentaires ==="
if [ -f "$RESULTS_DIR/classification_results.npy" ]; then
    # Script pour générer des graphiques à partir des résultats (confusion matrices, etc.)
    # python visualize_results.py --input_dir $RESULTS_DIR --output_dir $FIGURES_DIR
    echo "Figures générées dans: $FIGURES_DIR"
fi

echo -e "\n=== Terminé! Toutes les données ont été générées ==="
echo "Les résultats sont prêts pour être inclus dans le rapport LaTeX." 
