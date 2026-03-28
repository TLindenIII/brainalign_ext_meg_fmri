import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_results import parse_conversion_file, parse_retrieval_file
from src.checkpoints import conversion_results_path, retrieval_results_path


def move_file(source_path, target_path):
    if source_path.resolve() == target_path.resolve():
        return "skipped"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        if target_path.read_text() == source_path.read_text():
            source_path.unlink()
            return "deduped"
        raise FileExistsError(f"Refusing to overwrite non-identical file: {target_path}")

    source_path.rename(target_path)
    return "moved"


def migrate_retrieval_results(results_root):
    moved = 0
    deduped = 0
    for source_path in sorted(results_root.glob("*/evaluation_sub*.txt")):
        if "retrieval" in source_path.parts or "summary" in source_path.parts:
            continue
        record = parse_retrieval_file(source_path)
        target_path = retrieval_results_path(
            record["modality"],
            record["subject"],
            record["split"],
            record["evaluation_scope"],
            record["shared_group"],
        )
        outcome = move_file(source_path, target_path)
        if outcome == "moved":
            moved += 1
        elif outcome == "deduped":
            deduped += 1
    return moved, deduped


def migrate_conversion_results(results_root):
    moved = 0
    deduped = 0
    conversion_root = results_root / "conversion"
    if not conversion_root.exists():
        return moved, deduped

    for source_path in sorted(conversion_root.glob("*.txt")):
        record = parse_conversion_file(source_path)
        target_path = conversion_results_path(
            record["source_modality"],
            record["source_subject"],
            record["target_modality"],
            record["target_subject"],
            record["split"],
            record["evaluation_scope"],
            record["shared_group"],
        )
        outcome = move_file(source_path, target_path)
        if outcome == "moved":
            moved += 1
        elif outcome == "deduped":
            deduped += 1
    return moved, deduped


def main(results_root):
    results_root = Path(results_root)
    retrieval_moved, retrieval_deduped = migrate_retrieval_results(results_root)
    conversion_moved, conversion_deduped = migrate_conversion_results(results_root)

    print(
        "Migrated results layout: "
        f"retrieval moved={retrieval_moved}, retrieval deduped={retrieval_deduped}, "
        f"conversion moved={conversion_moved}, conversion deduped={conversion_deduped}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move flat legacy result files into the scoped results layout")
    parser.add_argument("--results-root", type=str, default="results", help="Root results directory")
    args = parser.parse_args()
    main(args.results_root)
