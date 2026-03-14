from pathlib import Path
import sys
import argparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate modality/image retrieval against CLIP")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--modality", type=str, required=True, choices=["eeg", "meg", "fmri"])
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint to evaluate")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID to evaluate")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--shared-only",
        action="store_true",
        help="Restrict evaluation to the shared image intersection",
    )
    args = parser.parse_args()
    main(args.config, args.modality, args.ckpt, args.subject, args.split, args.shared_only)
