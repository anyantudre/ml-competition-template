#!/bin/bash

MODEL=$(python -c "from config import MODEL; print(MODEL)")
TRAINING_SCRIPT=$(python -c "from config import TRAINING_SCRIPT; print(TRAINING_SCRIPT)")

python -W ignore "$TRAINING_SCRIPT" --model "$MODEL"