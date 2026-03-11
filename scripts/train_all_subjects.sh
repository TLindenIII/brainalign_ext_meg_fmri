#!/bin/bash

# Ensure we exit if any step fails
set -e

MODALITY="eeg"
EPOCHS=""
RESUME=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --modality) MODALITY="$2"; shift ;;
        --epochs) EPOCHS="--epochs $2"; shift ;;
        --resume) RESUME="--resume" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

MODALITY_UPPER=$(echo $MODALITY | tr '[:lower:]' '[:upper:]')
echo "Starting sequential multi-subject training for THINGS-${MODALITY_UPPER}..."

for i in {1..10}
do
    echo "============================================================"
    echo "Training Subject $i / 10 (${MODALITY_UPPER})"
    echo "============================================================"
    
    # Check if the checkpoint already exists to avoid redundant training
    SUB_ID=$(printf "%02d" $i)
    if [ -f "checkpoints/${MODALITY}/${MODALITY}_brainalign_sub${SUB_ID}_best.pt" ] && [ -z "$RESUME" ]; then
        echo "Checkpoint for Subject $i already exists. Skipping training..."
        echo ""
        continue
    fi
    
    # Activate virtual environment if running from root
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # Run the training script for the specific modality and subject
    PYTHONPATH=. python src/train.py --modality $MODALITY --subject $i $EPOCHS $RESUME
    
    echo "Finished training Subject $i."
    echo ""
done

echo "============================================================"
echo "All 10 subjects have been trained! Automatically generating the final averaged SOTA matrices..."
echo "============================================================"

PYTHONPATH=. python src/evaluate_table.py
