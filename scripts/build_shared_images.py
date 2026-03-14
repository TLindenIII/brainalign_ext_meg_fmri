from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.csv_utils import read_text_table


def load_config(config_path):
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def extract_meg_numeric_ids(meg_dir):
    meg_event_files = sorted(meg_dir.rglob("*events.tsv"))
    numeric_ids = set()

    print(f"Found {len(meg_event_files)} MEG events files.")
    for event_file in meg_event_files:
        df = pd.read_csv(
            event_file,
            sep="\t",
            encoding="latin-1",
            on_bad_lines="skip",
            engine="python",
        )
        if "value" not in df.columns:
            continue

        values = pd.to_numeric(df["value"], errors="coerce").dropna().astype(int)
        values = values[values > 0]
        numeric_ids.update(values.astype(str).tolist())

    print(f"Extracted {len(numeric_ids)} unique positive MEG event IDs.")
    return numeric_ids


def extract_fmri_image_ids(fmri_dir):
    beta_dir = fmri_dir / "derivatives" / "ICA-betas"
    fmri_event_files = sorted(beta_dir.rglob("*stimulus-metadata.tsv"))
    image_ids = set()

    print(f"Found {len(fmri_event_files)} fMRI stimulus metadata files.")
    for event_file in fmri_event_files:
        df = read_text_table(event_file, expected_columns={"stimulus"})
        if "stimulus" not in df.columns:
            continue

        stimuli = df.loc[df["stimulus"].notna(), "stimulus"]
        image_ids.update(Path(str(path)).stem for path in stimuli)

    print(f"Extracted {len(image_ids)} unique fMRI image IDs.")
    return image_ids


def build_meg_id_map(metadata_path):
    metadata = np.load(metadata_path, allow_pickle=True).item()
    train_files = [Path(path).stem for path in metadata["train_img_files"]]
    test_files = [Path(path).stem for path in metadata["test_img_files"]]
    all_files = train_files + test_files
    return {str(index + 1): image_id for index, image_id in enumerate(all_files)}


def main(config_path, output_path):
    config = load_config(config_path)

    meg_dir = ROOT / config["data"]["meg_dir"]
    fmri_dir = ROOT / config["data"]["fmri_dir"]
    metadata_path = ROOT / "data" / "things-eeg2" / "stimuli" / "image_metadata.npy"

    meg_numeric_ids = extract_meg_numeric_ids(meg_dir)
    fmri_image_ids = extract_fmri_image_ids(fmri_dir)

    id_to_image = build_meg_id_map(metadata_path)
    mapped_meg_images = {id_to_image[event_id] for event_id in meg_numeric_ids if event_id in id_to_image}
    unmapped_ids = meg_numeric_ids - set(id_to_image)

    print(f"Mapped {len(mapped_meg_images)} MEG event IDs into THINGS image IDs.")
    print(f"Ignored {len(unmapped_ids)} unmapped MEG event IDs (for example button presses / artifacts).")

    shared_images = sorted(mapped_meg_images & fmri_image_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        for image_id in shared_images:
            handle.write(f"{image_id}\n")

    print(f"Shared MEG/fMRI image IDs: {len(shared_images)}")
    print(f"Saved shared image list to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the shared THINGS image list for MEG/fMRI conversion")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--output",
        type=str,
        default="data/shared_images.txt",
        help="Path to write the shared image list",
    )
    args = parser.parse_args()

    main(args.config, ROOT / args.output)
