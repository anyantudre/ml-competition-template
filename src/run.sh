#!/bin/bash

# Extract N_FOLDS and MODEL from config.py
N_FOLDS=$(python -c "from config import N_FOLDS; print(N_FOLDS)")
MODEL=$(python -c "from config import MODEL; print(MODEL)")
TRAINING_SCRIPT=$(python -c "from config import TRAINING_SCRIPT; print(TRAINING_SCRIPT)")

# Loop through each fold
for ((i=0; i<N_FOLDS; i++)); do
    python -W ignore "$TRAINING_SCRIPT" --fold "$i" --model "$MODEL"
done
