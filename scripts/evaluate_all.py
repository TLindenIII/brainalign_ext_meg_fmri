import argparse
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_results import main as summarize_results_main
from src.checkpoints import discover_best_checkpoints
from src.data.image_manifest import default_intersection_manifest_path
from src.eval_utils import load_config
from src.evaluate import main as evaluate_retrieval_main
from src.evaluate_conversion_matrix import main as evaluate_conversion_matrix_main


VALID_MODALITIES = ("eeg", "meg", "fmri")


def parse_modalities(value):
    modalities = []
    seen = set()
    for raw_token in value.split(","):
        token = raw_token.strip().lower()
        if not token:
            continue
        if token not in VALID_MODALITIES:
            raise ValueError(f"Unsupported modality '{token}'")
        if token in seen:
            continue
        seen.add(token)
        modalities.append(token)
    if not modalities:
        raise ValueError("No modalities provided")
    return modalities


def subject_spec(subjects):
    return ",".join(str(subject) for subject in subjects)


def clean_results(modalities, split, remove_summary):
    removed = []
    for modality in modalities:
        modality_dir = ROOT / "results" / modality
        if not modality_dir.exists():
            continue
        for path in modality_dir.glob(f"evaluation_sub*_{split}_*.txt"):
            path.unlink()
            removed.append(path)

    conversion_dir = ROOT / "results" / "conversion"
    if conversion_dir.exists():
        conversion_pattern = re.compile(
            r"^(?P<source>[a-z]+)_sub\d+_to_(?P<target>[a-z]+)_sub\d+_(?P<split>\w+)\.txt$"
        )
        selected_modalities = set(modalities)
        for path in conversion_dir.glob(f"*_{split}.txt"):
            match = conversion_pattern.match(path.name)
            if not match:
                continue
            if (
                match.group("source") in selected_modalities
                or match.group("target") in selected_modalities
            ):
                path.unlink()
                removed.append(path)

    if remove_summary:
        summary_dir = ROOT / "results" / "summary"
        if summary_dir.exists():
            for path in summary_dir.iterdir():
                if path.is_file():
                    path.unlink()
                    removed.append(path)

    print(f"Removed {len(removed)} existing result files.")


def run_full_retrieval(config_path, modalities, split):
    for modality in modalities:
        checkpoints = discover_best_checkpoints(modality, shared_only=False)
        if not checkpoints:
            print(f"Skipping full retrieval for {modality.upper()}: no full best checkpoints found.")
            continue

        print(
            f"Running full retrieval for {modality.upper()} "
            f"subjects {', '.join(f'{subject:02d}' for subject in checkpoints)}"
        )
        for subject, checkpoint_path in checkpoints.items():
            evaluate_retrieval_main(
                config_path,
                modality,
                str(checkpoint_path),
                subject,
                split,
                False,
                None,
            )


def resolve_shared_manifest(config_path, modalities, explicit_manifest):
    if explicit_manifest:
        manifest_path = Path(explicit_manifest)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"Shared manifest not found: {manifest_path}")
        return manifest_path

    config = load_config(config_path)
    manifest_path = default_intersection_manifest_path(config, list(modalities))
    if manifest_path.exists():
        return manifest_path
    return None


def run_shared_suite(config_path, modalities, split, shared_manifest):
    if len(modalities) != 2:
        print("Skipping shared retrieval/conversion: exactly two modalities are required.")
        return

    manifest_path = resolve_shared_manifest(config_path, modalities, shared_manifest)
    if manifest_path is None:
        pair_name = "_".join(sorted(modalities))
        print(
            f"Skipping shared retrieval/conversion: no shared manifest found for {pair_name}. "
            "Pass --shared-manifest explicitly if needed."
        )
        return

    left_modality, right_modality = modalities
    left_checkpoints = discover_best_checkpoints(
        left_modality,
        shared_only=True,
        shared_manifest_path=str(manifest_path),
    )
    right_checkpoints = discover_best_checkpoints(
        right_modality,
        shared_only=True,
        shared_manifest_path=str(manifest_path),
    )

    if not left_checkpoints or not right_checkpoints:
        print(
            "Skipping shared retrieval/conversion: shared-only checkpoints are missing for "
            f"{left_modality.upper()} or {right_modality.upper()}."
        )
        return

    print(f"Using shared manifest: {manifest_path}")

    for modality, checkpoints in ((left_modality, left_checkpoints), (right_modality, right_checkpoints)):
        print(
            f"Running shared-only retrieval for {modality.upper()} "
            f"subjects {', '.join(f'{subject:02d}' for subject in checkpoints)}"
        )
        for subject, checkpoint_path in checkpoints.items():
            evaluate_retrieval_main(
                config_path,
                modality,
                str(checkpoint_path),
                subject,
                split,
                True,
                str(manifest_path),
            )

    print(
        f"Running full conversion matrix for {left_modality.upper()}[{subject_spec(left_checkpoints)}] "
        f"and {right_modality.upper()}[{subject_spec(right_checkpoints)}]"
    )
    evaluate_conversion_matrix_main(
        config_path,
        left_modality,
        right_modality,
        subject_spec(left_checkpoints),
        subject_spec(right_checkpoints),
        split,
        shared_manifest_path=str(manifest_path),
        source_ckpt_pattern=None,
        target_ckpt_pattern=None,
        source_shared_checkpoints=True,
        target_shared_checkpoints=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run full retrieval, shared retrieval, shared conversion, and result summarization in one pass"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--modalities",
        type=str,
        default="meg,fmri",
        help="Comma-separated modalities to evaluate, e.g. meg,fmri or eeg,meg,fmri",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--shared-manifest",
        type=str,
        default=None,
        help="Optional shared manifest for shared-only retrieval and conversion. If omitted, infer the pairwise manifest.",
    )
    parser.add_argument(
        "--skip-full-retrieval",
        action="store_true",
        help="Skip full-data retrieval evaluation",
    )
    parser.add_argument(
        "--skip-shared-suite",
        action="store_true",
        help="Skip shared-only retrieval and conversion-matrix evaluation",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip results summarization",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing result files for the selected modalities and split before rerunning",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results",
        help="Root directory containing evaluation text files for summarization",
    )
    parser.add_argument(
        "--summary-output-dir",
        type=str,
        default="results/summary",
        help="Directory for combined summary outputs",
    )
    args = parser.parse_args()

    modalities = parse_modalities(args.modalities)
    print(f"Evaluation modalities: {', '.join(modality.upper() for modality in modalities)}")

    if args.clean:
        clean_results(modalities, args.split, remove_summary=not args.skip_summary)

    if not args.skip_full_retrieval:
        run_full_retrieval(args.config, modalities, args.split)

    if not args.skip_shared_suite:
        run_shared_suite(args.config, modalities, args.split, args.shared_manifest)

    if not args.skip_summary:
        summarize_results_main(args.results_root, args.summary_output_dir)


if __name__ == "__main__":
    main()
