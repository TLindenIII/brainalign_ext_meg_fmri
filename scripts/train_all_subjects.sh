#!/bin/bash

# Ensure we exit if any step fails
set -e

MODALITY="eeg"
EPOCHS=""
RESUME=""
SHARED_ONLY=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --modality) MODALITY="$2"; shift ;;
        --epochs) EPOCHS="--epochs $2"; shift ;;
        --resume) RESUME="--resume" ;;
        --shared-only) SHARED_ONLY="--shared-only" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

discover_subjects() {
    case "$1" in
        eeg)
            find data/things-eeg2/preprocessed -maxdepth 1 -type d -name 'sub-*' \
                | sed 's#.*/sub-##' | sort -n
            ;;
        meg)
            find data/things-meg-ds004212/derivatives/preprocessed -maxdepth 1 -type f \
                \( -name 'preprocessed_P*-epo.fif' -o -name 'preprocessed_P*-epo-*.fif' \) \
                | sed -E 's#.*preprocessed_P([0-9]+)-epo(-[0-9]+)?\.fif#\1#' \
                | sort -n -u
            ;;
        fmri)
            find data/things-fmri-ds004192/derivatives/ICA-betas -maxdepth 1 -type d -name 'sub-*' \
                | sed 's#.*/sub-##' | sort -n
            ;;
        *)
            return 1
            ;;
    esac
}

MODALITY_UPPER=$(echo $MODALITY | tr '[:lower:]' '[:upper:]')
echo "Starting sequential multi-subject training for THINGS-${MODALITY_UPPER}..."
mapfile -t SUBJECT_IDS < <(discover_subjects "$MODALITY")

if [ ${#SUBJECT_IDS[@]} -eq 0 ]; then
    echo "No local subjects found for modality '${MODALITY}'."
    exit 1
fi

CHECKPOINT_SUFFIX=""
if [ -n "$SHARED_ONLY" ]; then
    CHECKPOINT_SUFFIX="_shared"
fi

for i in "${SUBJECT_IDS[@]}"
do
    echo "============================================================"
    echo "Training Subject $i / ${#SUBJECT_IDS[@]} (${MODALITY_UPPER})"
    echo "============================================================"
    
    # Check if the checkpoint already exists to avoid redundant training
    SUB_ID=$(printf "%02d" $i)
    if [ -f "checkpoints/${MODALITY}/${MODALITY}_brainalign_sub${SUB_ID}${CHECKPOINT_SUFFIX}_best.pt" ] && [ -z "$RESUME" ]; then
        echo "Checkpoint for Subject $i already exists. Skipping training..."
        echo ""
        continue
    fi
    
    # Activate virtual environment if running from root
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # Run the training script for the specific modality and subject
    PYTHONPATH=. python src/train.py --modality $MODALITY --subject $i $EPOCHS $RESUME $SHARED_ONLY
    
    echo "Finished training Subject $i."
    echo ""
done

echo "============================================================"
echo "All discovered ${MODALITY_UPPER} subjects have been trained."
echo "============================================================"

if [ "$MODALITY" = "eeg" ] && [ -z "$SHARED_ONLY" ]; then
    echo "Automatically generating the EEG summary table..."
    PYTHONPATH=. python scripts/evaluate_eeg_table.py
else
    echo "Skipping automatic summary for ${MODALITY_UPPER}."
    echo "Use scripts/evaluate_retrieval.py for modality-vs-image metrics and scripts/evaluate_conversion.py for shared-image conversion."
fi
