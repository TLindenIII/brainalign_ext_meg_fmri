#!/bin/bash

# Ensure we exit if any step fails
set -e

echo "Starting sequential multi-subject training for THINGS-EEG2..."

for i in {1..10}
do
    echo "============================================================"
    echo "Training Subject $i / 10"
    echo "============================================================"
    
    # Check if the checkpoint already exists to avoid redundant training
    SUB_ID=$(printf "%02d" $i)
    if [ -f "checkpoints/eeg/eeg_brainalign_sub${SUB_ID}_best.pt" ]; then
        echo "Checkpoint for Subject $i already exists (checkpoints/eeg/eeg_brainalign_sub${SUB_ID}_best.pt). Skipping training..."
        echo ""
        continue
    fi
    
    # Activate virtual environment if running from root
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # Run the training script for the specific modality and subject
    PYTHONPATH=. python src/train.py --modality eeg --subject $i
    
    echo "Finished training Subject $i."
    echo ""
done

echo "============================================================"
echo "All 10 subjects have been trained! Automatically generating the final averaged SOTA matrices..."
echo "============================================================"

PYTHONPATH=. python src/evaluate_table.py
