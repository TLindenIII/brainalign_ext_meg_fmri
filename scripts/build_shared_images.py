from pathlib import Path
import sys
import argparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_image_manifests import main as build_manifests_main


def main(config_path, output_path):
    build_manifests_main(config_path)
    shared_path = ROOT / "data" / "shared_images.txt"
    if shared_path.exists():
        print(f"Legacy MEG/fMRI shared image list remains at {shared_path}")
    else:
        print(
            "No legacy MEG/fMRI shared image list was written. "
            "This usually means the full THINGS image map is still missing."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build modality manifests and keep the legacy data/shared_images.txt in sync"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--output",
        type=str,
        default="data/shared_images.txt",
        help="Legacy output path (kept for compatibility; intersections are written under data/manifests/)",
    )
    args = parser.parse_args()

    main(args.config, ROOT / args.output)
