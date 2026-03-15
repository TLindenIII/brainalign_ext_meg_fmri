from pathlib import Path
import sys
import argparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build_clip_cache import build_clip_cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the CLIP feature cache for EEG/MEG/fMRI manifests")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional manifest TSV to build a universal cache from",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Root directory containing the source THINGS images",
    )
    args = parser.parse_args()
    build_clip_cache(args.config, manifest_path=args.manifest, image_root=args.image_root)
