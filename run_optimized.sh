#!/bin/bash

# Script optimisé pour exécuter l'interprétation de modèle avec des performances améliorées

echo "=== Lancement de l'analyse d'interprétation optimisée ==="

# Définir le nombre de processus pour scikit-learn
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Stocker les résultats dans un répertoire spécifique
RESULTS_DIR="results_optimized"
mkdir -p $RESULTS_DIR

# Exécuter uniquement le mode d'interprétation avec des paramètres optimisés
python run.py --mode interpret \
              --interpretation coefficients permutation \
              --output_dir $RESULTS_DIR

echo "=== Terminé ! ==="
echo "Résultats sauvegardés dans: $RESULTS_DIR" 
