from pathlib import Path
import sys
import argparse

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.image_manifest import (
    ensure_eeg_style_meg_split_lists,
    build_intersection_map,
    build_things_image_map_records,
    dedupe_named_records,
    load_eeg_image_records,
    load_fmri_image_records,
    load_meg_numeric_records,
    load_things_image_map,
    manifests_dir_from_config,
    map_meg_numeric_records,
    resolve_repo_path,
    resolve_things_image_list_path,
    resolve_things_image_map_path,
    split_manifests_dir_from_config,
    write_image_id_list,
    write_manifest_tsv,
)


def load_config(config_path):
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def write_template_map(path, meg_numeric_records):
    path.parent.mkdir(parents=True, exist_ok=True)
    template_records = [
        {
            "image_number": record["image_number"],
            "image_id": "",
            "relative_path": "",
        }
        for record in meg_numeric_records
    ]
    write_manifest_tsv(path, template_records)


def main(config_path):
    config = load_config(config_path)
    manifests_dir = manifests_dir_from_config(config)
    intersections_dir = manifests_dir / "intersections"
    meg_split_dir = split_manifests_dir_from_config(config, "meg", "fixed_image_holdout")
    manifests_dir.mkdir(parents=True, exist_ok=True)
    intersections_dir.mkdir(parents=True, exist_ok=True)
    meg_split_dir.mkdir(parents=True, exist_ok=True)

    eeg_records = load_eeg_image_records(config["data"]["eeg_dir"])
    fmri_records = load_fmri_image_records(config["data"]["fmri_dir"])
    meg_numeric_records = load_meg_numeric_records(config["data"]["meg_dir"])

    write_manifest_tsv(manifests_dir / "eeg_all.tsv", eeg_records)
    write_manifest_tsv(manifests_dir / "fmri_all.tsv", fmri_records)
    write_manifest_tsv(manifests_dir / "meg_numeric.tsv", meg_numeric_records)

    print(f"EEG images:  {len(eeg_records)}")
    print(f"fMRI images: {len(fmri_records)}")
    print(f"MEG image numbers: {len(meg_numeric_records)}")

    map_path = resolve_things_image_map_path(config)
    if map_path is None:
        image_list_path = resolve_things_image_list_path()
        configured_map_path = resolve_repo_path(
            config.get("data", {}).get("things_image_map_path", "data/things_image_map.tsv")
        )
        if image_list_path is not None:
            map_records = build_things_image_map_records(image_list_path)
            write_manifest_tsv(configured_map_path, map_records)
            map_path = configured_map_path
            print(f"Auto-generated THINGS image map from {image_list_path}")
            print(f"Saved {len(map_records)} rows to {map_path}")

    if map_path is None:
        template_path = manifests_dir / "things_image_map.template.tsv"
        write_template_map(template_path, meg_numeric_records)
        print("")
        print("No full THINGS image map was found.")
        print(f"Wrote a template to {template_path}")
        print("Populate data/things_image_map.tsv (or set data.things_image_map_path in config.yaml) and rerun.")
        return

    print(f"Using THINGS image map: {map_path}")
    things_image_map = load_things_image_map(map_path)
    meg_records, unmapped_numbers = map_meg_numeric_records(meg_numeric_records, things_image_map)
    if unmapped_numbers:
        print(f"Warning: {len(unmapped_numbers)} MEG image numbers were not found in the THINGS image map.")

    write_manifest_tsv(manifests_dir / "meg_all.tsv", meg_records)
    ensure_eeg_style_meg_split_lists(
        meg_split_dir,
        [record["image_id"] for record in meg_records],
        seed=42,
        test_concept_count=200,
        val_ratio=0.1,
        overwrite=True,
    )
    print(f"MEG fixed-image split manifests: {meg_split_dir}")

    all_records = dedupe_named_records(eeg_records + fmri_records + meg_records)
    write_manifest_tsv(manifests_dir / "all_modalities_union.tsv", all_records)

    named_sets = {
        "eeg": {record["image_id"] for record in eeg_records},
        "fmri": {record["image_id"] for record in fmri_records},
        "meg": {record["image_id"] for record in meg_records},
    }
    intersections = build_intersection_map(named_sets)
    for name, image_ids in intersections.items():
        out_path = intersections_dir / f"{name}.txt"
        write_image_id_list(out_path, image_ids)
        print(f"{name}: {len(image_ids)}")

    legacy_shared = intersections_dir / "fmri_meg.txt"
    if legacy_shared.exists():
        write_image_id_list(ROOT / "data" / "shared_images.txt", load_image_ids(legacy_shared))
        print("Updated legacy data/shared_images.txt from fmri_meg intersection")

    print(f"Saved manifests under {manifests_dir}")


def load_image_ids(path):
    with open(path, "r") as handle:
        return [line.strip() for line in handle if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build modality image manifests and intersections")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
