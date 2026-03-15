from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate_conversion_matrix import main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate all pairwise subject conversions for two modalities while loading each subject once"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--source-modality", type=str, required=True, choices=["eeg", "meg", "fmri"])
    parser.add_argument("--target-modality", type=str, required=True, choices=["eeg", "meg", "fmri"])
    parser.add_argument(
        "--source-subjects",
        type=str,
        required=True,
        help="Comma-separated subject list and/or ranges, e.g. 1-10 or 1,3,5",
    )
    parser.add_argument(
        "--target-subjects",
        type=str,
        required=True,
        help="Comma-separated subject list and/or ranges, e.g. 1-4 or 1,2",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--shared-manifest",
        type=str,
        default=None,
        help="Optional shared image manifest. Defaults to data/manifests/intersections/<modalities>.txt if present.",
    )
    parser.add_argument(
        "--source-ckpt-pattern",
        type=str,
        default=None,
        help="Optional source checkpoint pattern, e.g. checkpoints/eeg/eeg_brainalign_sub{subject02}_best.pt",
    )
    parser.add_argument(
        "--target-ckpt-pattern",
        type=str,
        default=None,
        help="Optional target checkpoint pattern, e.g. checkpoints/meg/meg_brainalign_sub{subject02}_attnpool_best.pt",
    )
    parser.add_argument(
        "--source-shared-checkpoints",
        action="store_true",
        help="Use the default shared-only checkpoint naming for the source modality",
    )
    parser.add_argument(
        "--target-shared-checkpoints",
        action="store_true",
        help="Use the default shared-only checkpoint naming for the target modality",
    )
    args = parser.parse_args()

    main(
        args.config,
        args.source_modality,
        args.target_modality,
        args.source_subjects,
        args.target_subjects,
        args.split,
        shared_manifest_path=args.shared_manifest,
        source_ckpt_pattern=args.source_ckpt_pattern,
        target_ckpt_pattern=args.target_ckpt_pattern,
        source_shared_checkpoints=args.source_shared_checkpoints,
        target_shared_checkpoints=args.target_shared_checkpoints,
    )
