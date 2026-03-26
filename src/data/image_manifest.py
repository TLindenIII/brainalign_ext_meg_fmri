import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.checkpoints import conversion_directory_name
from src.data.csv_utils import read_text_table


DEFAULT_THINGS_IMAGE_MAP_CANDIDATES = (
    "data/things_image_map.tsv",
    "data/things_image_map.csv",
    "data/things_image_map.json",
    "data/manifests/things_image_map.tsv",
)

DEFAULT_THINGS_IMAGE_LIST_CANDIDATES = (
    "osfstorage-archive/01_image-level/image-paths.csv",
    "data/things-images/01_image-level/image-paths.csv",
    "data/things/01_image-level/image-paths.csv",
)


def resolve_repo_path(path_like):
    if path_like is None:
        return None
    path = Path(path_like)
    return path if path.is_absolute() else Path.cwd() / path


def manifests_dir_from_config(config):
    return resolve_repo_path(config["data"].get("manifests_dir", "data/manifests"))


def split_manifests_dir_from_config(config, modality=None, split_mode=None):
    path = manifests_dir_from_config(config) / "splits"
    if modality:
        path = path / modality
    if split_mode:
        path = path / split_mode
    return path


def conversion_split_dir_from_config(config, shared_manifest_path=None, modalities=None):
    return (
        manifests_dir_from_config(config)
        / "splits"
        / "conversion"
        / conversion_directory_name(
            shared_manifest_path=shared_manifest_path,
            modalities=modalities,
        )
    )


def resolve_things_image_map_path(config=None, explicit_path=None):
    if explicit_path:
        resolved = resolve_repo_path(explicit_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Explicit THINGS image map path not found: {resolved}")
        return resolved

    candidates = []
    if config is not None:
        configured = config.get("data", {}).get("things_image_map_path")
        if configured:
            candidates.append(configured)
    candidates.extend(DEFAULT_THINGS_IMAGE_MAP_CANDIDATES)

    seen = set()
    for candidate in candidates:
        resolved = resolve_repo_path(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def resolve_things_image_list_path(explicit_path=None):
    candidates = []
    if explicit_path:
        resolved = resolve_repo_path(explicit_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Explicit THINGS image list path not found: {resolved}")
        return resolved

    candidates.extend(DEFAULT_THINGS_IMAGE_LIST_CANDIDATES)
    seen = set()
    for candidate in candidates:
        resolved = resolve_repo_path(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def default_relative_path_for_image_id(image_id):
    concept = image_id.rsplit("_", 1)[0]
    return f"{concept}/{image_id}.jpg"


def load_named_image_ids(path):
    with open(path, "r") as handle:
        return {line.strip() for line in handle if line.strip()}


def resolve_shared_manifest_path(shared_only, shared_manifest_path=None):
    if not shared_only:
        return None

    resolved = resolve_repo_path(shared_manifest_path) if shared_manifest_path else None
    if resolved:
        if not resolved.exists():
            raise FileNotFoundError(f"Explicit shared manifest path not found: {resolved}")
        return resolved

    legacy = resolve_repo_path("data/shared_images.txt")
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        "shared_only=True requires a shared image manifest. "
        "Provide --shared-manifest or generate manifests with scripts/build_shared_images.py."
    )


def default_intersection_manifest_path(config, modalities):
    name = "_".join(sorted(modalities))
    return manifests_dir_from_config(config) / "intersections" / f"{name}.txt"


def default_conversion_pool_manifest_path(config, modalities):
    name = "_".join(sorted(modalities))
    return manifests_dir_from_config(config) / "conversion_pools" / f"{name}.txt"


def load_things_image_map(path):
    path = resolve_repo_path(path)
    if path is None or not path.exists():
        raise FileNotFoundError(f"THINGS image map not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_things_image_map_json(path)
    if suffix in {".csv", ".tsv", ".txt"}:
        return _load_things_image_map_table(path)

    raise ValueError(f"Unsupported THINGS image map format: {path}")


def build_things_image_map_records(image_list_path):
    image_list_path = resolve_repo_path(image_list_path)
    if image_list_path is None or not image_list_path.exists():
        raise FileNotFoundError(f"THINGS image list not found: {image_list_path}")

    records = []
    with open(image_list_path, "r") as handle:
        for image_number, raw_line in enumerate(handle, start=1):
            relative_path = raw_line.strip().replace("\\", "/")
            if not relative_path:
                continue
            for prefix in ("images_THINGS/object_images/", "object_images/", "images/"):
                if relative_path.startswith(prefix):
                    relative_path = relative_path[len(prefix) :]
                    break
            image_id = Path(relative_path).stem
            records.append(
                {
                    "image_number": image_number,
                    "image_id": image_id,
                    "relative_path": relative_path,
                }
            )

    return records


def _load_things_image_map_json(path):
    payload = json.loads(path.read_text())
    mapping = {}

    if isinstance(payload, dict):
        items = payload.items()
    elif isinstance(payload, list):
        items = enumerate(payload)
    else:
        raise ValueError(f"Unexpected JSON structure in {path}")

    for key, value in items:
        if isinstance(value, dict):
            image_number = int(value.get("image_number", key))
            image_id = value.get("image_id")
            relative_path = value.get("relative_path")
        else:
            image_number = int(key)
            image_id = str(value)
            relative_path = None

        if not image_id:
            raise ValueError(f"Missing image_id for image number {image_number} in {path}")

        mapping[image_number] = {
            "image_id": image_id,
            "relative_path": relative_path or default_relative_path_for_image_id(image_id),
        }

    return mapping


def _load_things_image_map_table(path):
    df = read_text_table(path)

    numeric_col = _first_present(
        df.columns,
        ("image_number", "things_image_number", "things_image_nr", "value", "id"),
    )
    image_col = _first_present(
        df.columns,
        ("image_id", "stimulus", "file_path", "relative_path", "filename"),
    )

    if numeric_col is None or image_col is None:
        raise ValueError(
            f"{path} must contain an image number column and an image identifier/path column"
        )

    relative_col = _first_present(df.columns, ("relative_path", "file_path", "stimulus", "filename"))

    mapping = {}
    for _, row in df.iterrows():
        if pd.isna(row[numeric_col]) or pd.isna(row[image_col]):
            continue

        image_number = int(row[numeric_col])
        raw_value = str(row[image_col]).strip()
        image_id = Path(raw_value).stem
        relative_path = None
        if relative_col and not pd.isna(row[relative_col]):
            relative_path = str(row[relative_col]).strip().replace("\\", "/")

        mapping[image_number] = {
            "image_id": image_id,
            "relative_path": relative_path or default_relative_path_for_image_id(image_id),
        }

    return mapping


def load_eeg_image_records(eeg_dir, include_train=True, include_test=True):
    eeg_dir = resolve_repo_path(eeg_dir)
    metadata = np.load(eeg_dir / "stimuli" / "image_metadata.npy", allow_pickle=True).item()

    records = []
    file_names = []
    if include_train:
        file_names.extend(list(metadata["train_img_files"]))
    if include_test:
        file_names.extend(list(metadata["test_img_files"]))

    for file_name in file_names:
        image_id = Path(file_name).stem
        records.append(
            {
                "image_id": image_id,
                "relative_path": default_relative_path_for_image_id(image_id),
                "source": "eeg",
            }
        )
    return dedupe_named_records(records)


def load_fmri_image_records(fmri_dir, trial_types=None):
    fmri_dir = resolve_repo_path(fmri_dir)
    records = []
    normalized_trial_types = None
    if trial_types is not None:
        normalized_trial_types = {str(value).strip().lower() for value in trial_types}

    raw_event_files = sorted(fmri_dir.rglob("*task-things*_events.tsv"))
    for event_file in raw_event_files:
        df = read_text_table(event_file, expected_columns={"file_path"})
        if "file_path" not in df.columns:
            continue

        if "trial_type" in df.columns:
            if normalized_trial_types is None:
                df = df[df["trial_type"].isin(["exp", "test"])]
            else:
                trial_type_values = df["trial_type"].astype(str).str.strip().str.lower()
                df = df[trial_type_values.isin(normalized_trial_types)]

        for file_path in df["file_path"].dropna():
            rel_path = str(file_path).strip().replace("\\", "/")
            if not rel_path or rel_path.startswith("catch"):
                continue
            image_id = Path(rel_path).stem
            records.append(
                {
                    "image_id": image_id,
                    "relative_path": rel_path,
                    "source": "fmri",
                }
            )

    if records:
        return dedupe_named_records(records)

    beta_files = sorted((fmri_dir / "derivatives" / "ICA-betas").rglob("*stimulus-metadata.tsv"))
    for event_file in beta_files:
        df = read_text_table(event_file, expected_columns={"stimulus"})
        if "stimulus" not in df.columns:
            continue
        if "trial_type" in df.columns and normalized_trial_types is not None:
            trial_type_values = df["trial_type"].astype(str).str.strip().str.lower()
            df = df[trial_type_values.isin(normalized_trial_types)]
        for stim in df["stimulus"].dropna():
            image_id = Path(str(stim)).stem
            records.append(
                {
                    "image_id": image_id,
                    "relative_path": default_relative_path_for_image_id(image_id),
                    "source": "fmri",
                }
            )

    return dedupe_named_records(records)


def load_meg_numeric_records(meg_dir):
    meg_dir = resolve_repo_path(meg_dir)
    records = []
    event_files = sorted(meg_dir.rglob("*events.tsv"))

    for event_file in event_files:
        df = read_text_table(event_file, expected_columns={"value"})
        if "value" not in df.columns:
            continue
        if "trial_type" in df.columns:
            df = df[df["trial_type"].isin(["exp", "test"])]

        values = pd.to_numeric(df["value"], errors="coerce").dropna().astype(int)
        values = values[(values > 0) & (values != 999999)]
        for image_number in values.tolist():
            records.append({"image_number": image_number})

    unique_numbers = sorted({record["image_number"] for record in records})
    return [{"image_number": image_number} for image_number in unique_numbers]


def map_meg_numeric_records(meg_numeric_records, things_image_map):
    mapped_records = []
    unmapped_numbers = []

    for record in meg_numeric_records:
        image_number = record["image_number"]
        mapped = things_image_map.get(image_number)
        if mapped is None:
            unmapped_numbers.append(image_number)
            continue
        mapped_records.append(
            {
                "image_number": image_number,
                "image_id": mapped["image_id"],
                "relative_path": mapped["relative_path"],
                "source": "meg",
            }
        )

    return dedupe_named_records(mapped_records), unmapped_numbers


def dedupe_named_records(records):
    merged = {}
    for record in records:
        image_id = record["image_id"]
        existing = merged.get(image_id)
        if existing is None:
            merged[image_id] = dict(record)
            continue

        for key, value in record.items():
            if key == "source":
                existing_sources = set(existing.get("source", "").split(",")) if existing.get("source") else set()
                new_sources = set(str(value).split(",")) if value else set()
                existing["source"] = ",".join(sorted(s for s in existing_sources | new_sources if s))
            elif not existing.get(key) and value:
                existing[key] = value

    return [merged[key] for key in sorted(merged)]


def write_manifest_tsv(path, records):
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        with open(path, "w", newline="") as handle:
            handle.write("")
        return

    fieldnames = []
    seen = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def read_manifest_tsv(path):
    path = resolve_repo_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return read_text_table(path)


def write_image_id_list(path, image_ids):
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for image_id in sorted(set(image_ids)):
            handle.write(f"{image_id}\n")


def ensure_image_split_lists(split_dir, image_ids, seed=42, ratios=(0.8, 0.1, 0.1), overwrite=False):
    split_dir = resolve_repo_path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / "train.txt"
    val_path = split_dir / "val.txt"
    test_path = split_dir / "test.txt"
    if not overwrite and train_path.exists() and val_path.exists() and test_path.exists():
        return

    unique_images = sorted(set(image_ids))
    if not unique_images:
        raise ValueError(f"Cannot build split manifests under {split_dir}: no image IDs provided")

    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must be a 3-tuple summing to 1.0")

    rng = np.random.RandomState(seed)
    shuffled = unique_images.copy()
    rng.shuffle(shuffled)

    train_end = int(ratios[0] * len(shuffled))
    val_end = int((ratios[0] + ratios[1]) * len(shuffled))
    train_ids = shuffled[:train_end]
    val_ids = shuffled[train_end:val_end]
    test_ids = shuffled[val_end:]

    write_image_id_list(train_path, train_ids)
    write_image_id_list(val_path, val_ids)
    write_image_id_list(test_path, test_ids)


def ensure_eeg_style_meg_split_lists(
    split_dir,
    image_ids,
    seed=42,
    test_concept_count=200,
    val_ratio=0.1,
    overwrite=False,
):
    split_dir = resolve_repo_path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / "train.txt"
    val_path = split_dir / "val.txt"
    test_path = split_dir / "test.txt"
    excluded_path = split_dir / "excluded.txt"
    if (
        not overwrite
        and train_path.exists()
        and val_path.exists()
        and test_path.exists()
        and excluded_path.exists()
    ):
        return

    concept_to_images = defaultdict(list)
    for image_id in sorted(set(image_ids)):
        concept_to_images[image_id.rsplit("_", 1)[0]].append(image_id)

    concepts = sorted(concept_to_images)
    if len(concepts) < test_concept_count:
        raise ValueError(
            f"Cannot build EEG-style MEG split with {test_concept_count} test concepts; "
            f"only {len(concepts)} concepts are available."
        )

    rng = np.random.RandomState(seed)
    shuffled_concepts = concepts.copy()
    rng.shuffle(shuffled_concepts)
    test_concepts = set(shuffled_concepts[:test_concept_count])

    test_ids = []
    excluded_ids = []
    remaining_ids = []
    for concept in concepts:
        concept_images = sorted(concept_to_images[concept])
        if concept in test_concepts:
            picked_index = int(rng.randint(len(concept_images)))
            picked_image = concept_images[picked_index]
            test_ids.append(picked_image)
            excluded_ids.extend(image_id for image_id in concept_images if image_id != picked_image)
        else:
            remaining_ids.extend(concept_images)

    remaining_ids = sorted(remaining_ids)
    rng.shuffle(remaining_ids)
    val_count = int(round(val_ratio * len(remaining_ids)))
    val_ids = remaining_ids[:val_count]
    train_ids = remaining_ids[val_count:]

    write_image_id_list(train_path, train_ids)
    write_image_id_list(val_path, val_ids)
    write_image_id_list(test_path, test_ids)
    write_image_id_list(excluded_path, excluded_ids)


def ensure_shared_conversion_split_lists(
    split_dir,
    image_ids,
    seed=42,
    val_concept_count=100,
    test_concept_count=200,
    overwrite=False,
):
    split_dir = resolve_repo_path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / "train.txt"
    val_path = split_dir / "val.txt"
    test_path = split_dir / "test.txt"
    excluded_path = split_dir / "excluded.txt"
    if (
        not overwrite
        and train_path.exists()
        and val_path.exists()
        and test_path.exists()
        and excluded_path.exists()
    ):
        return

    concept_to_images = defaultdict(list)
    for image_id in sorted(set(image_ids)):
        concept_to_images[image_id.rsplit("_", 1)[0]].append(image_id)

    concepts = sorted(concept_to_images)
    required_concepts = val_concept_count + test_concept_count
    if len(concepts) <= required_concepts:
        raise ValueError(
            f"Cannot build shared conversion split with {val_concept_count} val concepts and "
            f"{test_concept_count} test concepts; only {len(concepts)} concepts are available."
        )

    rng = np.random.RandomState(seed)
    shuffled_concepts = concepts.copy()
    rng.shuffle(shuffled_concepts)
    test_concepts = set(shuffled_concepts[:test_concept_count])
    val_concepts = set(shuffled_concepts[test_concept_count : required_concepts])

    train_ids = []
    val_ids = []
    test_ids = []
    excluded_ids = []

    for concept in concepts:
        concept_images = sorted(concept_to_images[concept])
        if concept in test_concepts:
            picked_image = concept_images[int(rng.randint(len(concept_images)))]
            test_ids.append(picked_image)
            excluded_ids.extend(image_id for image_id in concept_images if image_id != picked_image)
            continue
        if concept in val_concepts:
            picked_image = concept_images[int(rng.randint(len(concept_images)))]
            val_ids.append(picked_image)
            excluded_ids.extend(image_id for image_id in concept_images if image_id != picked_image)
            continue
        train_ids.extend(concept_images)

    write_image_id_list(train_path, train_ids)
    write_image_id_list(val_path, val_ids)
    write_image_id_list(test_path, test_ids)
    write_image_id_list(excluded_path, excluded_ids)


def load_image_split_lists(split_dir):
    split_dir = resolve_repo_path(split_dir)
    splits = {}
    for split_name in ("train", "val", "test"):
        path = split_dir / f"{split_name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Split manifest not found: {path}")
        splits[split_name] = load_named_image_ids(path)
    return splits


def build_intersection_map(named_sets):
    intersections = {}
    modalities = sorted(named_sets)
    for size in range(2, len(modalities) + 1):
        for combo in _combinations(modalities, size):
            combo_name = "_".join(combo)
            shared = set.intersection(*(named_sets[name] for name in combo))
            intersections[combo_name] = sorted(shared)
    return intersections


def _combinations(values, size):
    if size == 0:
        yield ()
        return
    if len(values) < size:
        return
    if size == 1:
        for value in values:
            yield (value,)
        return

    for idx, value in enumerate(values):
        for tail in _combinations(values[idx + 1 :], size - 1):
            yield (value,) + tail


def _first_present(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None
