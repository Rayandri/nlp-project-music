#!/bin/bash

# Script pour générer toutes les données nécessaires au rapport de projet NLP
# Execution: bash generate_results.sh

# Détection automatique du nombre de threads disponibles
TOTAL_THREADS=$(nproc)

# Configuration pour utiliser TOUTE la puissance disponible
export OPENBLAS_NUM_THREADS=$TOTAL_THREADS
export MKL_NUM_THREADS=$TOTAL_THREADS
export OMP_NUM_THREADS=$TOTAL_THREADS
export PYTHONPATH=.
export JOBLIB_TEMP_FOLDER=/tmp
export JOBLIB_THREADS=$TOTAL_THREADS

echo "Utilisation maximale: $TOTAL_THREADS threads pour tous les calculs"

# Création des répertoires pour les résultats
RESULTS_DIR="results_rapport"

mkdir -p $RESULTS_DIR

echo "=== Génération des données pour le rapport ==="
echo "Résultats sauvegardés dans: $RESULTS_DIR"

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

# 7. Génération de résumé global des résultats
echo -e "\n=== 7. Génération du résumé global ==="
{
    echo "=== RÉSUMÉ GLOBAL DES RÉSULTATS ==="
    echo "Date: $(date)"
    echo ""
    echo "=== Classification ==="
    [ -f "$RESULTS_DIR/classification_summary.txt" ] && cat "$RESULTS_DIR/classification_summary.txt" | grep "Meilleure méthode\|Précision\|F1-score" || echo "Résultats de classification non disponibles"
    echo ""
    echo "=== Génération ==="
    [ -f "$RESULTS_DIR/generation_summary.txt" ] && head -n 10 "$RESULTS_DIR/generation_summary.txt" || echo "Résultats de génération non disponibles"
    echo ""
    echo "=== Augmentation ==="
    [ -f "$RESULTS_DIR/augmentation_summary.txt" ] && grep "Amélioration" "$RESULTS_DIR/augmentation_summary.txt" || echo "Résultats d'augmentation non disponibles"
    echo ""
    echo "=== Interprétation ==="
    [ -f "$RESULTS_DIR/interpretation_summary.txt" ] && grep -A 5 "Importance des features" "$RESULTS_DIR/interpretation_summary.txt" | head -n 10 || echo "Résultats d'interprétation non disponibles"
    echo ""
    echo "=== VISUALISATIONS GÉNÉRÉES ==="
    find "$RESULTS_DIR" -name "*.png" | sort
} > "$RESULTS_DIR/resume_global.txt"

echo "Résumé global généré: $RESULTS_DIR/resume_global.txt"

echo -e "\n=== Terminé! Toutes les données ont été générées ==="
echo "Les résultats sont prêts pour être inclus dans le rapport LaTeX." 
