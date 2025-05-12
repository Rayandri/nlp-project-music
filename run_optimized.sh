#!/bin/bash

echo "=== Lancement de l'analyse d'interprétation optimisée ==="

TOTAL_THREADS=$(nproc)

export OPENBLAS_NUM_THREADS=$TOTAL_THREADS
export MKL_NUM_THREADS=$TOTAL_THREADS
export OMP_NUM_THREADS=$TOTAL_THREADS
export PYTHONPATH=.
export JOBLIB_TEMP_FOLDER=/tmp
export JOBLIB_THREADS=$TOTAL_THREADS

echo "Utilisation maximale: $TOTAL_THREADS threads pour tous les calculs"

RESULTS_DIR="results_rapport"
mkdir -p $RESULTS_DIR

python run.py --mode interpret \
              --interpretation coefficients permutation \
              --output_dir $RESULTS_DIR

echo "=== Terminé ! ==="
echo "Résultats sauvegardés dans: $RESULTS_DIR" 
