from pathlib import Path
import sys
import argparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate_table import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the EEG evaluation summary tables")
    parser.add_argument(
        "--subject",
        type=int,
        default=None,
        help="Optional single EEG subject to evaluate. Omit to evaluate all available EEG checkpoints.",
    )
    args = parser.parse_args()
    main(args.subject)
