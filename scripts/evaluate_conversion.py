from pathlib import Path
import sys
import argparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate_conversion import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cross-modality conversion on shared images")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--source-modality", type=str, required=True, choices=["eeg", "meg", "fmri"])
    parser.add_argument("--source-ckpt", type=str, required=True, help="Checkpoint for the source modality")
    parser.add_argument("--source-subject", type=int, default=1, help="Source subject ID")
    parser.add_argument("--target-modality", type=str, required=True, choices=["eeg", "meg", "fmri"])
    parser.add_argument("--target-ckpt", type=str, required=True, help="Checkpoint for the target modality")
    parser.add_argument("--target-subject", type=int, default=1, help="Target subject ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--shared-manifest",
        type=str,
        default=None,
        help="Optional shared image manifest",
    )
    args = parser.parse_args()

    main(
        args.config,
        args.source_modality,
        args.source_ckpt,
        args.source_subject,
        args.target_modality,
        args.target_ckpt,
        args.target_subject,
        args.split,
        args.shared_manifest,
    )
