from pathlib import Path
import sys
import argparse
import csv
import math
import re
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


RETRIEVAL_PATTERN = re.compile(
    r"--- Evaluation Results \((?P<modality>[A-Z]+) / subject (?P<subject>\d+)\) ---\s+"
    r"Checkpoint: (?P<checkpoint>.+)\s+"
    r"Split: (?P<split>\w+)\s+"
    r"(?:Evaluation scope: (?P<evaluation_scope>[a-z_]+)\s+)?"
    r"(?:Shared group: (?P<shared_group>[a-z0-9-]+)\s+)?"
    r"Shared-only images: (?P<shared_only>True|False)\s+"
    r"Candidate images: (?P<candidate_count>\d+)\s+"
    r"Modality -> Image\s+"
    r"Top-1 Retrieval: (?P<m2i_top1>[0-9.]+)%\s+"
    r"Top-5 Retrieval: (?P<m2i_top5>[0-9.]+)%\s+"
    r"CLIP 2-Way:\s+(?P<m2i_two_way>[0-9.]+)%\s+"
    r"Image -> Modality\s+"
    r"Top-1 Retrieval: (?P<i2m_top1>[0-9.]+)%\s+"
    r"Top-5 Retrieval: (?P<i2m_top5>[0-9.]+)%\s+"
    r"CLIP 2-Way:\s+(?P<i2m_two_way>[0-9.]+)%",
    re.DOTALL,
)


CONVERSION_PATTERN = re.compile(
    r"--- Conversion Results \((?P<source_modality>[A-Z]+) sub-(?P<source_subject>\d+) <-> "
    r"(?P<target_modality>[A-Z]+) sub-(?P<target_subject>\d+)\) ---\s+"
    r"Source checkpoint: (?P<source_checkpoint>.+)\s+"
    r"Target checkpoint: (?P<target_checkpoint>.+)\s+"
    r"Split: (?P<split>\w+)\s+"
    r"(?:Evaluation scope: (?P<evaluation_scope>[a-z_]+)\s+)?"
    r"(?:Shared group: (?P<shared_group>[a-z0-9-]+)\s+)?"
    r"Shared-only images: (?P<shared_only>True|False)\s+"
    r"Aligned shared test images: (?P<candidate_count>\d+)\s+"
    r"(?P<forward_label>[a-z_]+)\s+"
    r"Top-1 Retrieval: (?P<forward_top1>[0-9.]+)%\s+"
    r"Top-5 Retrieval: (?P<forward_top5>[0-9.]+)%\s+"
    r"CLIP 2-Way:\s+(?P<forward_two_way>[0-9.]+)%\s+"
    r"(?P<reverse_label>[a-z_]+)\s+"
    r"Top-1 Retrieval: (?P<reverse_top1>[0-9.]+)%\s+"
    r"Top-5 Retrieval: (?P<reverse_top5>[0-9.]+)%\s+"
    r"CLIP 2-Way:\s+(?P<reverse_two_way>[0-9.]+)%",
    re.DOTALL,
)


EEG_TABLE_FIELD_MAP = {
    1: "m2i_top1",
    2: "i2m_top1",
    3: "m2i_top5",
    4: "i2m_top5",
    5: "m2i_two_way",
    6: "i2m_two_way",
}


def parse_bool(value):
    return str(value).strip().lower() == "true"


def infer_shared_group(*paths):
    pattern = re.compile(r"checkpoints[/\\]conversion[/\\](?P<group>shared-[^/\\]+)")
    for path in paths:
        if not path:
            continue
        match = pattern.search(str(path))
        if match:
            return match.group("group")
    return "none"


def infer_evaluation_scope(shared_only, shared_group):
    if not shared_only:
        return "full"
    if not shared_group or shared_group == "none":
        return "shared"
    modality_count = len([token for token in shared_group.replace("shared-", "").split("-") if token])
    if modality_count >= 3:
        return "three_way"
    if modality_count == 2:
        return "pair"
    return "shared"


def parse_float_fields(record, keys):
    for key in keys:
        record[key] = float(record[key])
    return record


def parse_int_fields(record, keys):
    for key in keys:
        record[key] = int(record[key])
    return record


def parse_retrieval_file(path):
    match = RETRIEVAL_PATTERN.search(path.read_text())
    if not match:
        raise ValueError(f"Could not parse retrieval results file: {path}")

    record = match.groupdict()
    record["file_name"] = path.name
    record["modality"] = record["modality"].lower()
    record["subject"] = int(record["subject"])
    record["shared_only"] = parse_bool(record["shared_only"])
    record["shared_group"] = record.get("shared_group") or infer_shared_group(record["checkpoint"])
    record["evaluation_scope"] = record.get("evaluation_scope") or infer_evaluation_scope(
        record["shared_only"],
        record["shared_group"],
    )
    parse_int_fields(record, ["candidate_count"])
    parse_float_fields(
        record,
        ["m2i_top1", "m2i_top5", "m2i_two_way", "i2m_top1", "i2m_top5", "i2m_two_way"],
    )
    return record


def parse_conversion_file(path):
    match = CONVERSION_PATTERN.search(path.read_text())
    if not match:
        raise ValueError(f"Could not parse conversion results file: {path}")

    record = match.groupdict()
    record["file_name"] = path.name
    record["source_modality"] = record["source_modality"].lower()
    record["target_modality"] = record["target_modality"].lower()
    record["source_subject"] = int(record["source_subject"])
    record["target_subject"] = int(record["target_subject"])
    record["shared_only"] = parse_bool(record["shared_only"])
    record["shared_group"] = record.get("shared_group") or infer_shared_group(
        record["source_checkpoint"],
        record["target_checkpoint"],
    )
    record["evaluation_scope"] = record.get("evaluation_scope") or infer_evaluation_scope(
        record["shared_only"],
        record["shared_group"],
    )
    parse_int_fields(record, ["candidate_count"])
    parse_float_fields(
        record,
        ["forward_top1", "forward_top5", "forward_two_way", "reverse_top1", "reverse_top5", "reverse_two_way"],
    )
    return record


def parse_eeg_summary_file(path):
    text = path.read_text()
    candidate_match = re.search(r"EEG-to-Image\s+(?P<count>\d+)-way", text)
    candidate_count = int(candidate_match.group("count")) if candidate_match else 200

    subject_rows = {
        subject: {
            "file_name": path.name,
            "modality": "eeg",
            "subject": subject,
            "split": "test_200way",
            "evaluation_scope": "full",
            "shared_group": "none",
            "shared_only": False,
            "candidate_count": candidate_count,
            "checkpoint": "summary_only",
        }
        for subject in range(1, 11)
    }

    table_rows = {}
    lines = [line.rstrip() for line in text.splitlines()]
    for idx, line in enumerate(lines):
        match = re.match(r"Table (\d+):", line.strip())
        if not match:
            continue
        table_idx = int(match.group(1))
        if table_idx not in EEG_TABLE_FIELD_MAP:
            continue
        if idx + 2 >= len(lines):
            raise ValueError(f"Incomplete EEG summary table {table_idx} in {path}")
        table_rows[table_idx] = lines[idx + 2].strip()

    for table_idx, metric_key in EEG_TABLE_FIELD_MAP.items():
        row_text = table_rows.get(table_idx)
        if row_text is None:
            raise ValueError(f"Could not parse EEG summary table {table_idx} from {path}")
        numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", row_text)]
        if len(numbers) < 12:
            raise ValueError(f"Unexpected EEG summary row format in table {table_idx} from {path}")
        subject_values = numbers[:10]
        for subject, value in enumerate(subject_values, start=1):
            subject_rows[subject][metric_key] = value

    return [subject_rows[subject] for subject in sorted(subject_rows)]


def add_retrieval_baselines(rows):
    for row in rows:
        candidate_count = row["candidate_count"]
        top_k = min(5, candidate_count)
        row["baseline_top1_pct"] = 100.0 / candidate_count
        row["baseline_top5_pct"] = 100.0 * top_k / candidate_count
        # In the current evaluator one averaged embedding is retained per image_id,
        # so candidate_count is both the retrieval set size and the class count.
        row["retrieval_dataset_size"] = candidate_count
        row["number_of_classes"] = candidate_count
    return rows


def build_retrieval_lookup(rows):
    lookup = {}
    for row in rows:
        key = (row["modality"], row["subject"], row["split"], row["shared_only"])
        lookup[key] = row
    return lookup


def lookup_retrieval_reference(lookup, modality, subject, split, shared_only=False):
    exact = lookup.get((modality, subject, split, shared_only))
    if exact is not None:
        return exact

    if modality == "eeg":
        legacy = lookup.get((modality, subject, "test_200way", shared_only))
        if legacy is not None:
            return legacy

    return None


def add_conversion_normalization(rows, retrieval_lookup):
    for row in rows:
        forward_ref = lookup_retrieval_reference(
            retrieval_lookup,
            row["target_modality"],
            row["target_subject"],
            row["split"],
            False,
        )
        reverse_ref = lookup_retrieval_reference(
            retrieval_lookup,
            row["source_modality"],
            row["source_subject"],
            row["split"],
            False,
        )
        row["forward_reference_two_way"] = forward_ref["m2i_two_way"] if forward_ref else float("nan")
        row["reverse_reference_two_way"] = reverse_ref["m2i_two_way"] if reverse_ref else float("nan")
        row["forward_normalized_two_way"] = (
            row["forward_two_way"] / row["forward_reference_two_way"]
            if forward_ref and forward_ref["m2i_two_way"] > 0
            else float("nan")
        )
        row["reverse_normalized_two_way"] = (
            row["reverse_two_way"] / row["reverse_reference_two_way"]
            if reverse_ref and reverse_ref["m2i_two_way"] > 0
            else float("nan")
        )
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def metric_summary(values):
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize_group(rows, group_keys, metric_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key, group_rows in sorted(grouped.items()):
        summary = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        summary["count"] = len(group_rows)
        for metric_key in metric_keys:
            stats = metric_summary([row[metric_key] for row in group_rows])
            summary[f"{metric_key}_mean"] = round(stats["mean"], 4)
            summary[f"{metric_key}_std"] = round(stats["std"], 4)
            summary[f"{metric_key}_min"] = round(stats["min"], 4)
            summary[f"{metric_key}_max"] = round(stats["max"], 4)
        summary_rows.append(summary)
    return summary_rows


def format_value(value):
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.2f}"
    return str(value)


def markdown_table(rows, headers, columns):
    if not rows:
        return "_No rows found._"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def build_report(retrieval_rows, retrieval_summary_rows, conversion_rows, conversion_summary_rows):
    lines = ["# Results Summary", ""]
    has_legacy_eeg_rows = any(
        row["modality"] == "eeg" and row["split"] == "test_200way"
        for row in retrieval_rows
    )
    retrieval_note = (
        "Note: EEG rows come from the existing 200-way EEG summary table, while MEG/fMRI rows come from the full-image retrieval evaluator."
        if has_legacy_eeg_rows
        else "Note: Retrieval rows come from the per-subject retrieval evaluator for all available modalities."
    )

    lines.extend(
        [
            "## Retrieval By Subject",
            "",
            retrieval_note,
            "",
            markdown_table(
                retrieval_rows,
                ["Modality", "Subject", "Split", "Scope", "Group", "Shared", "Candidates", "M->I Top-1", "M->I Top-5", "M->I 2-way", "I->M Top-1", "I->M Top-5", "I->M 2-way"],
                ["modality", "subject", "split", "evaluation_scope", "shared_group", "shared_only", "candidate_count", "m2i_top1", "m2i_top5", "m2i_two_way", "i2m_top1", "i2m_top5", "i2m_two_way"],
            ),
            "",
            "## Retrieval Summary",
            "",
            markdown_table(
                retrieval_summary_rows,
                ["Modality", "Split", "Scope", "Group", "Shared", "N", "M->I Top-1", "M->I Top-5", "M->I 2-way", "Base Top-1", "Base Top-5", "Retrieval Size", "Classes"],
                ["modality", "split", "evaluation_scope", "shared_group", "shared_only", "count", "m2i_top1_mean", "m2i_top5_mean", "m2i_two_way_mean", "baseline_top1_pct_mean", "baseline_top5_pct_mean", "retrieval_dataset_size_mean", "number_of_classes_mean"],
            ),
            "",
            "## Conversion By Pair",
            "",
            markdown_table(
                conversion_rows,
                ["Source", "Src Sub", "Target", "Tgt Sub", "Split", "Scope", "Group", "Shared", "Candidates", "Forward Top-1", "Forward Top-5", "Forward 2-way", "Forward Norm", "Reverse Top-1", "Reverse Top-5", "Reverse 2-way", "Reverse Norm"],
                ["source_modality", "source_subject", "target_modality", "target_subject", "split", "evaluation_scope", "shared_group", "shared_only", "candidate_count", "forward_top1", "forward_top5", "forward_two_way", "forward_normalized_two_way", "reverse_top1", "reverse_top5", "reverse_two_way", "reverse_normalized_two_way"],
            ),
            "",
            "## Conversion Summary",
            "",
            markdown_table(
                conversion_summary_rows,
                ["Source", "Target", "Split", "Scope", "Group", "Shared", "N", "Forward 2-way", "Forward Norm", "Reverse 2-way", "Reverse Norm"],
                ["source_modality", "target_modality", "split", "evaluation_scope", "shared_group", "shared_only", "count", "forward_two_way_mean", "forward_normalized_two_way_mean", "reverse_two_way_mean", "reverse_normalized_two_way_mean"],
            ),
            "",
        ]
    )

    return "\n".join(lines)


def preferred_decoding_rows(summary_rows):
    preferred = {}
    split_priority = {"test": 0, "test_200way": 1, "val": 2, "train": 3}

    for row in summary_rows:
        if row["shared_only"]:
            continue

        modality = row["modality"]
        candidate = (
            split_priority.get(row["split"], 99),
            -row["count"],
        )
        current = preferred.get(modality)
        if current is None or candidate < current[0]:
            preferred[modality] = (candidate, row)

    return [preferred[key][1] for key in sorted(preferred)]


def collect_retrieval_files(results_root):
    files = []
    retrieval_root = results_root / "retrieval"
    if retrieval_root.exists():
        files.extend(retrieval_root.rglob("evaluation_sub*.txt"))

    for path in results_root.glob("*/evaluation_sub*.txt"):
        if "retrieval" in path.parts or "summary" in path.parts:
            continue
        files.append(path)

    return sorted(set(files))


def collect_conversion_files(results_root):
    conversion_root = results_root / "conversion"
    if not conversion_root.exists():
        return []
    return sorted(path for path in conversion_root.rglob("*.txt") if path.is_file())


def main(results_root, output_dir):
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_files = collect_retrieval_files(results_root)
    conversion_files = collect_conversion_files(results_root)
    eeg_summary_path = results_root / "eeg" / "evaluation_summary.txt"

    retrieval_rows = [parse_retrieval_file(path) for path in retrieval_files]
    has_subject_level_eeg = any(row["modality"] == "eeg" for row in retrieval_rows)
    if eeg_summary_path.exists() and not has_subject_level_eeg:
        retrieval_rows.extend(parse_eeg_summary_file(eeg_summary_path))
    retrieval_rows = add_retrieval_baselines(retrieval_rows)
    retrieval_lookup = build_retrieval_lookup(retrieval_rows)
    conversion_rows = add_conversion_normalization(
        [parse_conversion_file(path) for path in conversion_files],
        retrieval_lookup,
    )

    retrieval_summary_rows = summarize_group(
        retrieval_rows,
        ["modality", "split", "evaluation_scope", "shared_group", "shared_only"],
        [
            "m2i_top1",
            "m2i_top5",
            "m2i_two_way",
            "i2m_top1",
            "i2m_top5",
            "i2m_two_way",
            "baseline_top1_pct",
            "baseline_top5_pct",
            "retrieval_dataset_size",
            "number_of_classes",
        ],
    )
    conversion_summary_rows = summarize_group(
        conversion_rows,
        ["source_modality", "target_modality", "split", "evaluation_scope", "shared_group", "shared_only"],
        [
            "forward_top1",
            "forward_top5",
            "forward_two_way",
            "forward_normalized_two_way",
            "reverse_top1",
            "reverse_top5",
            "reverse_two_way",
            "reverse_normalized_two_way",
        ],
    )

    decoding_table_rows = [
        {
            "neural_module": row["modality"].upper(),
            "top1_accuracy": row["m2i_top1_mean"],
            "top5_accuracy": row["m2i_top5_mean"],
            "clip_2_way": row["m2i_two_way_mean"],
            "baseline_accuracy_pct": row["baseline_top1_pct_mean"],
            "baseline_top5_pct": row["baseline_top5_pct_mean"],
            "retrieval_dataset_size": round(row["retrieval_dataset_size_mean"], 4),
            "number_of_classes": round(row["number_of_classes_mean"], 4),
            "split": row["split"],
            "evaluation_scope": row["evaluation_scope"],
            "shared_group": row["shared_group"],
            "shared_only": row["shared_only"],
        }
        for row in preferred_decoding_rows(retrieval_summary_rows)
    ]

    conversion_table_rows = []
    for row in conversion_summary_rows:
        conversion_table_rows.append(
            {
                "conversion": f"{row['source_modality'].upper()} to {row['target_modality'].upper()}",
                "clip_2_way_decoding_accuracy": row["forward_two_way_mean"] / 100.0,
                "normalized_clip_2_way_decoding_accuracy": row["forward_normalized_two_way_mean"],
                "split": row["split"],
                "evaluation_scope": row["evaluation_scope"],
                "shared_group": row["shared_group"],
                "shared_only": row["shared_only"],
                "pair_count": row["count"],
            }
        )
        conversion_table_rows.append(
            {
                "conversion": f"{row['target_modality'].upper()} to {row['source_modality'].upper()}",
                "clip_2_way_decoding_accuracy": row["reverse_two_way_mean"] / 100.0,
                "normalized_clip_2_way_decoding_accuracy": row["reverse_normalized_two_way_mean"],
                "split": row["split"],
                "evaluation_scope": row["evaluation_scope"],
                "shared_group": row["shared_group"],
                "shared_only": row["shared_only"],
                "pair_count": row["count"],
            }
        )

    retrieval_fields = [
        "file_name",
        "modality",
        "subject",
        "split",
        "evaluation_scope",
        "shared_group",
        "shared_only",
        "candidate_count",
        "checkpoint",
        "baseline_top1_pct",
        "baseline_top5_pct",
        "retrieval_dataset_size",
        "number_of_classes",
        "m2i_top1",
        "m2i_top5",
        "m2i_two_way",
        "i2m_top1",
        "i2m_top5",
        "i2m_two_way",
    ]
    write_csv(output_dir / "retrieval_by_subject.csv", retrieval_rows, retrieval_fields)

    retrieval_summary_fields = [
        "modality",
        "split",
        "evaluation_scope",
        "shared_group",
        "shared_only",
        "count",
        "m2i_top1_mean",
        "m2i_top1_std",
        "m2i_top1_min",
        "m2i_top1_max",
        "m2i_top5_mean",
        "m2i_top5_std",
        "m2i_top5_min",
        "m2i_top5_max",
        "m2i_two_way_mean",
        "m2i_two_way_std",
        "m2i_two_way_min",
        "m2i_two_way_max",
        "baseline_top1_pct_mean",
        "baseline_top1_pct_std",
        "baseline_top1_pct_min",
        "baseline_top1_pct_max",
        "baseline_top5_pct_mean",
        "baseline_top5_pct_std",
        "baseline_top5_pct_min",
        "baseline_top5_pct_max",
        "retrieval_dataset_size_mean",
        "retrieval_dataset_size_std",
        "retrieval_dataset_size_min",
        "retrieval_dataset_size_max",
        "number_of_classes_mean",
        "number_of_classes_std",
        "number_of_classes_min",
        "number_of_classes_max",
        "i2m_top1_mean",
        "i2m_top1_std",
        "i2m_top1_min",
        "i2m_top1_max",
        "i2m_top5_mean",
        "i2m_top5_std",
        "i2m_top5_min",
        "i2m_top5_max",
        "i2m_two_way_mean",
        "i2m_two_way_std",
        "i2m_two_way_min",
        "i2m_two_way_max",
    ]
    write_csv(output_dir / "retrieval_summary.csv", retrieval_summary_rows, retrieval_summary_fields)

    conversion_fields = [
        "file_name",
        "source_modality",
        "source_subject",
        "target_modality",
        "target_subject",
        "split",
        "evaluation_scope",
        "shared_group",
        "shared_only",
        "candidate_count",
        "source_checkpoint",
        "target_checkpoint",
        "forward_label",
        "forward_reference_two_way",
        "forward_top1",
        "forward_top5",
        "forward_two_way",
        "forward_normalized_two_way",
        "reverse_label",
        "reverse_reference_two_way",
        "reverse_top1",
        "reverse_top5",
        "reverse_two_way",
        "reverse_normalized_two_way",
    ]
    write_csv(output_dir / "conversion_by_pair.csv", conversion_rows, conversion_fields)

    conversion_summary_fields = [
        "source_modality",
        "target_modality",
        "split",
        "evaluation_scope",
        "shared_group",
        "shared_only",
        "count",
        "forward_top1_mean",
        "forward_top1_std",
        "forward_top1_min",
        "forward_top1_max",
        "forward_top5_mean",
        "forward_top5_std",
        "forward_top5_min",
        "forward_top5_max",
        "forward_two_way_mean",
        "forward_two_way_std",
        "forward_two_way_min",
        "forward_two_way_max",
        "forward_normalized_two_way_mean",
        "forward_normalized_two_way_std",
        "forward_normalized_two_way_min",
        "forward_normalized_two_way_max",
        "reverse_top1_mean",
        "reverse_top1_std",
        "reverse_top1_min",
        "reverse_top1_max",
        "reverse_top5_mean",
        "reverse_top5_std",
        "reverse_top5_min",
        "reverse_top5_max",
        "reverse_two_way_mean",
        "reverse_two_way_std",
        "reverse_two_way_min",
        "reverse_two_way_max",
        "reverse_normalized_two_way_mean",
        "reverse_normalized_two_way_std",
        "reverse_normalized_two_way_min",
        "reverse_normalized_two_way_max",
    ]
    write_csv(output_dir / "conversion_summary.csv", conversion_summary_rows, conversion_summary_fields)

    decoding_table_fields = [
        "neural_module",
        "top1_accuracy",
        "top5_accuracy",
        "clip_2_way",
        "baseline_accuracy_pct",
        "baseline_top5_pct",
        "retrieval_dataset_size",
        "number_of_classes",
        "split",
        "evaluation_scope",
        "shared_group",
        "shared_only",
    ]
    write_csv(output_dir / "paper_decoding_table.csv", decoding_table_rows, decoding_table_fields)

    conversion_table_fields = [
        "conversion",
        "clip_2_way_decoding_accuracy",
        "normalized_clip_2_way_decoding_accuracy",
        "split",
        "evaluation_scope",
        "shared_group",
        "shared_only",
        "pair_count",
    ]
    write_csv(output_dir / "paper_conversion_table.csv", conversion_table_rows, conversion_table_fields)

    report = build_report(retrieval_rows, retrieval_summary_rows, conversion_rows, conversion_summary_rows)
    report_path = output_dir / "summary_report.md"
    report_path.write_text(report)

    print(f"Wrote {output_dir / 'retrieval_by_subject.csv'}")
    print(f"Wrote {output_dir / 'retrieval_summary.csv'}")
    print(f"Wrote {output_dir / 'conversion_by_pair.csv'}")
    print(f"Wrote {output_dir / 'conversion_summary.csv'}")
    print(f"Wrote {output_dir / 'paper_decoding_table.csv'}")
    print(f"Wrote {output_dir / 'paper_conversion_table.csv'}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate retrieval and conversion result text files")
    parser.add_argument("--results-root", type=str, default="results", help="Root results directory")
    parser.add_argument("--output-dir", type=str, default="results/summary", help="Directory for combined outputs")
    args = parser.parse_args()
    main(args.results_root, args.output_dir)
