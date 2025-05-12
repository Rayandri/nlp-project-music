#!/bin/bash

# Script optimisé pour exécuter l'interprétation de modèle avec des performances améliorées

echo "=== Lancement de l'analyse d'interprétation optimisée ==="

# Détection automatique du nombre de threads disponibles
TOTAL_THREADS=$(nproc)
PHYSICAL_CORES=$((TOTAL_THREADS / 2)) # Estimation pour CPU hyperthreadé

# Pour les bibliothèques numériques (laisser quelques threads libres pour le système)
MATH_THREADS=$((PHYSICAL_CORES - 1) * 2 )
MATH_THREADS=$((MATH_THREADS < 1 ? 1 : MATH_THREADS))

# Pour joblib (utiliser tous les threads disponibles)
JOBLIB_THREADS=$TOTAL_THREADS

echo "Détection auto: $TOTAL_THREADS threads total (estimé $PHYSICAL_CORES cœurs physiques)"
echo "Allocation: $MATH_THREADS threads pour calculs mathématiques, $JOBLIB_THREADS pour parallélisation"

# Configuration des threads pour les différentes bibliothèques
export OPENBLAS_NUM_THREADS=$MATH_THREADS
export MKL_NUM_THREADS=$MATH_THREADS
export OMP_NUM_THREADS=$MATH_THREADS
export PYTHONPATH=.
export JOBLIB_TEMP_FOLDER=/tmp

# Stocker les résultats dans un répertoire spécifique
RESULTS_DIR="results_rapport"
mkdir -p $RESULTS_DIR

echo "Utilisation configurée pour VM avec 32 cœurs"

# Exécuter uniquement le mode d'interprétation avec des paramètres optimisés
python run.py --mode interpret \
              --interpretation coefficients permutation \
              --output_dir $RESULTS_DIR

echo "=== Terminé ! ==="
echo "Résultats sauvegardés dans: $RESULTS_DIR" 
