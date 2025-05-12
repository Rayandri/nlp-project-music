#!/bin/bash

# Script optimisé pour exécuter l'interprétation de modèle avec des performances améliorées

echo "=== Lancement de l'analyse d'interprétation optimisée ==="

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

# Stocker les résultats dans un répertoire spécifique
RESULTS_DIR="results_rapport"
mkdir -p $RESULTS_DIR

# Exécuter uniquement le mode d'interprétation avec des paramètres optimisés
python run.py --mode interpret \
              --interpretation coefficients permutation \
              --output_dir $RESULTS_DIR

echo "=== Terminé ! ==="
echo "Résultats sauvegardés dans: $RESULTS_DIR" 
