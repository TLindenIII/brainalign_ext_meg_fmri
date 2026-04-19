import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.experimental_training import (  # noqa: E402
    VALID_EXPERIMENTAL_CLIP_OBJECTIVES,
    train_joint_models,
)


VALID_MODALITIES = ("eeg", "meg", "fmri")
OBJECTIVE_NAME = "clip_anchor_xmodal"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train one paired conversion experiment with CLIP anchoring plus "
            "a cross-modal same-image contrastive term."
        )
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--source-modality", type=str, required=True, choices=VALID_MODALITIES)
    parser.add_argument("--target-modality", type=str, required=True, choices=VALID_MODALITIES)
    parser.add_argument("--source-subject", type=int, required=True)
    parser.add_argument("--target-subject", type=int, required=True)
    parser.add_argument(
        "--shared-manifest",
        type=str,
        required=True,
        help="Shared image manifest used to build aligned same-image batches",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Experiment namespace under checkpoints/experiments/<name>/",
    )
    parser.add_argument(
        "--alignment-objective",
        type=str,
        default=OBJECTIVE_NAME,
        choices=[OBJECTIVE_NAME],
        help="Named experimental objective for checkpoint metadata.",
    )
    parser.add_argument(
        "--clip-objective",
        type=str,
        default="brain_to_clip",
        choices=VALID_EXPERIMENTAL_CLIP_OBJECTIVES,
        help="CLIP-anchor loss used for each modality before the cross-modal term.",
    )
    parser.add_argument(
        "--lambda-cross",
        type=float,
        default=1.0,
        help="Weight applied to the source-target cross-modal contrastive loss.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override aligned-batch size")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoints")
    parser.add_argument("--resume-best", action="store_true", help="Resume from best checkpoints")
    args = parser.parse_args()

    if args.source_modality == args.target_modality:
        parser.error("source and target modalities must be different")

    modality_subjects = {
        args.source_modality: args.source_subject,
        args.target_modality: args.target_subject,
    }

    train_joint_models(
        args.config,
        modality_subjects,
        args.shared_manifest,
        args.experiment_name,
        epochs_override=args.epochs,
        resume=args.resume,
        resume_best=args.resume_best,
        clip_objective=args.clip_objective,
        lambda_cross=args.lambda_cross,
        batch_size=args.batch_size,
        objective_name=args.alignment_objective,
    )


if __name__ == "__main__":
    main()
