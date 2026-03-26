import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(text),
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(code),
    }


def _lines(text):
    normalized = textwrap.dedent(text).strip("\n")
    if not normalized:
        return []
    return [line + "\n" for line in normalized.splitlines()]


def write_notebook(path, cells):
    payload = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")


def main():
    root_setup = """
    from pathlib import Path
    import sys

    ROOT = Path.cwd().resolve()
    if ROOT.name == "notebooks":
        ROOT = ROOT.parent
    sys.path.insert(0, str(ROOT))
    """

    write_notebook(
        NOTEBOOKS_DIR / "00_data_loader_scratchpad.ipynb",
        [
            md_cell(
                """
                # Data Loader Scratchpad

                This notebook is a quick sanity check for the current EEG, MEG, and fMRI loaders.
                It uses the repo APIs directly, so the displayed split sizes stay aligned with the codebase.
                """
            ),
            code_cell(
                root_setup
                + """

                import pandas as pd

                from src.eval_utils import create_dataset, load_config

                config = load_config(ROOT / "config.yaml")
                """
            ),
            code_cell(
                """
                def summarize_dataset(modality, subject, split, shared_only=False, shared_manifest=None):
                    config_local = load_config(ROOT / "config.yaml")
                    if shared_manifest:
                        config_local.setdefault("data", {})["shared_manifest_path"] = str(ROOT / shared_manifest)
                    dataset = create_dataset(
                        config_local,
                        modality,
                        split,
                        subject=subject,
                        shared_only=shared_only,
                        quiet=True,
                    )
                    if modality == "eeg":
                        unique_images = len({Path(path).stem for path in dataset.files})
                        unique_concepts = len({concept for concept in dataset.concepts})
                        trial_count = len(dataset)
                    else:
                        unique_images = len({trial["image_id"] for trial in dataset.trials})
                        unique_concepts = len({trial["image_id"].rsplit("_", 1)[0] for trial in dataset.trials})
                        trial_count = len(dataset.trials)
                    sample = dataset[0]["x"]
                    return {
                        "modality": modality,
                        "subject": subject,
                        "split": split,
                        "shared_only": shared_only,
                        "unique_images": unique_images,
                        "unique_concepts": unique_concepts,
                        "trial_count": trial_count,
                        "sample_shape": tuple(sample.shape),
                    }

                rows = []
                for modality, subject in [("eeg", 1), ("meg", 1), ("fmri", 1)]:
                    for split in ["train", "val", "test"]:
                        rows.append(summarize_dataset(modality, subject, split))

                pd.DataFrame(rows)
                """
            ),
        ],
    )

    write_notebook(
        NOTEBOOKS_DIR / "01_shared_image_intersection.ipynb",
        [
            md_cell(
                """
                # Shared Overlaps And Conversion Pools

                This notebook distinguishes three related objects:

                - raw cross-modal overlaps in `data/manifests/intersections/`
                - trainable shared conversion pools in `data/manifests/conversion_pools/`
                - fixed shared conversion split manifests in `data/manifests/splits/conversion/`
                """
            ),
            code_cell(
                root_setup
                + """

                from pathlib import Path
                import pandas as pd

                from src.data.image_manifest import conversion_split_dir_from_config
                from src.eval_utils import load_config

                config = load_config(ROOT / "config.yaml")
                """
            ),
            code_cell(
                """
                def load_ids(path):
                    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]

                def concept_count(image_ids):
                    return len({image_id.rsplit("_", 1)[0] for image_id in image_ids})

                raw_rows = []
                conversion_rows = []
                for name in ["eeg_meg", "eeg_fmri", "fmri_meg", "eeg_fmri_meg"]:
                    raw_path = ROOT / "data" / "manifests" / "intersections" / f"{name}.txt"
                    pool_path = ROOT / "data" / "manifests" / "conversion_pools" / f"{name}.txt"
                    split_dir = conversion_split_dir_from_config(config, shared_manifest_path=pool_path)

                    raw_ids = load_ids(raw_path)
                    pool_ids = load_ids(pool_path)
                    train_ids = load_ids(split_dir / "train.txt")
                    val_ids = load_ids(split_dir / "val.txt")
                    test_ids = load_ids(split_dir / "test.txt")
                    excluded_ids = load_ids(split_dir / "excluded.txt")

                    raw_rows.append(
                        {
                            "pool": name,
                            "raw_overlap_images": len(raw_ids),
                            "raw_overlap_concepts": concept_count(raw_ids),
                        }
                    )
                    conversion_rows.append(
                        {
                            "pool": name,
                            "conversion_pool_images": len(pool_ids),
                            "conversion_pool_concepts": concept_count(pool_ids),
                            "train_images": len(train_ids),
                            "train_concepts": concept_count(train_ids),
                            "val_images": len(val_ids),
                            "val_concepts": concept_count(val_ids),
                            "test_images": len(test_ids),
                            "test_concepts": concept_count(test_ids),
                            "excluded_images": len(excluded_ids),
                        }
                    )

                raw_df = pd.DataFrame(raw_rows)
                conversion_df = pd.DataFrame(conversion_rows)
                """
            ),
            code_cell(
                """
                raw_df
                """
            ),
            code_cell(
                """
                conversion_df
                """
            ),
        ],
    )

    write_notebook(
        NOTEBOOKS_DIR / "02_retrieval_evaluation.ipynb",
        [
            md_cell(
                """
                # Retrieval Evaluation

                This notebook evaluates one checkpoint through the same code path used by the CLI.
                It works for EEG, MEG, or fMRI, and it can also run shared-only retrieval when you provide a shared manifest.
                """
            ),
            code_cell(
                root_setup
                + """

                from pathlib import Path
                import pandas as pd
                import torch

                from src.checkpoints import resolve_existing_checkpoint_path
                from src.eval_utils import (
                    build_model,
                    create_dataloader,
                    load_checkpoint,
                    load_clip_cache,
                    load_config,
                )
                from src.evaluate import evaluate

                CONFIG_PATH = ROOT / "config.yaml"
                MODALITY = "meg"
                SUBJECT = 1
                SPLIT = "test"
                SHARED_ONLY = False
                SHARED_MANIFEST = None  # e.g. ROOT / "data/manifests/conversion_pools/eeg_fmri.txt"

                config = load_config(CONFIG_PATH)
                if SHARED_MANIFEST is not None:
                    config.setdefault("data", {})["shared_manifest_path"] = str(SHARED_MANIFEST)
                elif SHARED_ONLY:
                    raise ValueError("Set SHARED_MANIFEST when SHARED_ONLY=True")

                checkpoint_path = resolve_existing_checkpoint_path(
                    MODALITY,
                    SUBJECT,
                    kind="best",
                    shared_only=SHARED_ONLY,
                    shared_manifest_path=str(SHARED_MANIFEST) if SHARED_MANIFEST else None,
                )
                checkpoint_path
                """
            ),
            code_cell(
                """
                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu"
                )
                clip_dict = load_clip_cache(config)
                loader = create_dataloader(
                    config,
                    MODALITY,
                    SPLIT,
                    subject=SUBJECT,
                    shared_only=SHARED_ONLY,
                    quiet=False,
                    shuffle=False,
                )
                sample_x = loader.dataset[0]["x"]
                model = build_model(config, MODALITY, sample_x, device)
                load_checkpoint(model, checkpoint_path, device)
                metrics = evaluate(model, loader, clip_dict, device)

                pd.DataFrame(
                    [
                        {
                            "modality": MODALITY,
                            "subject": SUBJECT,
                            "split": SPLIT,
                            "shared_only": SHARED_ONLY,
                            "candidate_images": metrics["candidate_count"],
                            "modality_to_image_top1": metrics["modality_to_image"]["top1"],
                            "modality_to_image_top5": metrics["modality_to_image"]["top5"],
                            "modality_to_image_two_way": metrics["modality_to_image"]["two_way"],
                            "image_to_modality_top1": metrics["image_to_modality"]["top1"],
                            "image_to_modality_top5": metrics["image_to_modality"]["top5"],
                            "image_to_modality_two_way": metrics["image_to_modality"]["two_way"],
                        }
                    ]
                )
                """
            ),
        ],
    )

    write_notebook(
        NOTEBOOKS_DIR / "03_linear_baselines.ipynb",
        [
            md_cell(
                """
                # Linear Baselines

                This notebook runs ridge and CCA baselines against the current repo loaders.
                It respects the active modality split definitions from `config.yaml`.
                """
            ),
            code_cell(
                root_setup
                + """

                import numpy as np
                import pandas as pd
                from sklearn.cross_decomposition import CCA
                from sklearn.linear_model import Ridge

                from src.eval_utils import create_dataset, load_clip_cache, load_config

                CONFIG_PATH = ROOT / "config.yaml"
                MODALITY = "eeg"  # eeg, meg, fmri
                SUBJECT = 1

                config = load_config(CONFIG_PATH)
                clip_dict = load_clip_cache(config)
                """
            ),
            code_cell(
                """
                def flatten_dataset(dataset):
                    xs = []
                    ys = []
                    image_ids = []
                    for index in range(len(dataset)):
                        item = dataset[index]
                        xs.append(item["x"].numpy().reshape(-1))
                        ys.append(item["y_clip"].numpy())
                        image_ids.append(item["image_id"])
                    return np.stack(xs), np.stack(ys), image_ids

                def retrieval_scores(predictions, candidate_ids, candidate_embeddings, image_ids):
                    grouped = {}
                    for image_id, prediction in zip(image_ids, predictions):
                        grouped.setdefault(image_id, []).append(prediction)

                    candidate_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
                    candidate_matrix = candidate_embeddings / np.clip(candidate_norms, 1e-12, None)

                    top1 = 0
                    top5 = 0
                    total = 0
                    for image_id, vectors in grouped.items():
                        averaged = np.mean(np.stack(vectors, axis=0), axis=0, dtype=np.float32)
                        averaged = averaged / max(np.linalg.norm(averaged), 1e-12)
                        scores = averaged @ candidate_matrix.T
                        ranking = np.argsort(scores)[::-1]
                        top_ids = [candidate_ids[idx] for idx in ranking[:5]]
                        top1 += int(top_ids[0] == image_id)
                        top5 += int(image_id in top_ids)
                        total += 1

                    return {
                        "top1": 100.0 * top1 / total,
                        "top5": 100.0 * top5 / total,
                        "candidate_images": total,
                    }

                train_dataset = create_dataset(config, MODALITY, "train", subject=SUBJECT, quiet=True)
                test_dataset = create_dataset(config, MODALITY, "test", subject=SUBJECT, quiet=True)

                X_train, Y_train, _ = flatten_dataset(train_dataset)
                X_test, Y_test, test_ids = flatten_dataset(test_dataset)
                candidate_ids = sorted(set(test_ids))
                candidate_embeddings = np.stack([clip_dict[image_id] for image_id in candidate_ids], axis=0)
                """
            ),
            code_cell(
                """
                ridge = Ridge(alpha=10000.0)
                ridge.fit(X_train, Y_train)
                ridge_predictions = ridge.predict(X_test)
                ridge_scores = retrieval_scores(ridge_predictions, candidate_ids, candidate_embeddings, test_ids)

                cca = CCA(n_components=min(100, Y_train.shape[1], X_train.shape[1]))
                cca.fit(X_train, Y_train)
                X_test_c, _ = cca.transform(X_test, Y_test)
                _, candidate_c = cca.transform(np.zeros((len(candidate_ids), X_train.shape[1])), candidate_embeddings)
                cca_scores = retrieval_scores(X_test_c, candidate_ids, candidate_c, test_ids)

                pd.DataFrame(
                    [
                        {"baseline": "ridge", **ridge_scores},
                        {"baseline": "cca", **cca_scores},
                    ]
                )
                """
            ),
        ],
    )

    write_notebook(
        NOTEBOOKS_DIR / "04_conversion_heatmaps.ipynb",
        [
            md_cell(
                """
                # Conversion Heatmaps

                This notebook visualizes the summarized conversion results from `results/summary/conversion_by_pair.csv`.
                Regenerate the summaries first if needed.
                """
            ),
            code_cell(
                root_setup
                + """

                from pathlib import Path

                import matplotlib.pyplot as plt
                import pandas as pd

                RESULTS_PATH = ROOT / "results" / "summary" / "conversion_by_pair.csv"
                assert RESULTS_PATH.exists(), f"Missing summary CSV: {RESULTS_PATH}"
                conversion_df = pd.read_csv(RESULTS_PATH)
                conversion_df.head()
                """
            ),
            code_cell(
                """
                def expand_rows(frame):
                    rows = []
                    for _, record in frame.iterrows():
                        rows.append(
                            {
                                "direction": f"{record['source_modality']}→{record['target_modality']}",
                                "source_subject": record["source_subject"],
                                "target_subject": record["target_subject"],
                                "top1": record["forward_top1"],
                                "top5": record["forward_top5"],
                                "two_way": record["forward_two_way"],
                            }
                        )
                        rows.append(
                            {
                                "direction": f"{record['target_modality']}→{record['source_modality']}",
                                "source_subject": record["target_subject"],
                                "target_subject": record["source_subject"],
                                "top1": record["reverse_top1"],
                                "top5": record["reverse_top5"],
                                "two_way": record["reverse_two_way"],
                            }
                        )
                    return pd.DataFrame(rows)

                long_df = expand_rows(conversion_df)
                long_df.head()
                """
            ),
            code_cell(
                """
                DIRECTION = sorted(long_df["direction"].unique())[0]
                METRIC = "top1"

                subset = long_df[long_df["direction"] == DIRECTION]
                heatmap = subset.pivot(index="source_subject", columns="target_subject", values=METRIC)

                fig, ax = plt.subplots(figsize=(6, 5))
                image = ax.imshow(heatmap.values, cmap="YlGnBu")
                ax.set_title(f"{DIRECTION} | {METRIC}")
                ax.set_xlabel("Target subject")
                ax.set_ylabel("Source subject")
                ax.set_xticks(range(len(heatmap.columns)))
                ax.set_xticklabels(heatmap.columns)
                ax.set_yticks(range(len(heatmap.index)))
                ax.set_yticklabels(heatmap.index)
                fig.colorbar(image, ax=ax)
                plt.show()
                """
            ),
        ],
    )

    write_notebook(
        NOTEBOOKS_DIR / "05_retrieval_example_grids.ipynb",
        [
            md_cell(
                """
                # Retrieval Example Grids

                This notebook builds small qualitative retrieval grids using the current checkpoint resolver and loader APIs.
                """
            ),
            code_cell(
                root_setup
                + """

                from pathlib import Path

                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd
                import torch
                from PIL import Image

                from src.checkpoints import resolve_existing_checkpoint_path
                from src.eval_utils import (
                    build_model,
                    clip_embeddings_for_ids,
                    collect_modality_embeddings,
                    create_dataloader,
                    load_checkpoint,
                    load_clip_cache,
                    load_config,
                )

                CONFIG_PATH = ROOT / "config.yaml"
                MODALITY = "meg"
                SUBJECT = 1
                SPLIT = "test"
                NUM_QUERIES = 4
                TOPK = 5
                SHARED_ONLY = False
                SHARED_MANIFEST = None  # e.g. ROOT / "data/manifests/conversion_pools/eeg_fmri_meg.txt"

                config = load_config(CONFIG_PATH)
                if SHARED_MANIFEST is not None:
                    config.setdefault("data", {})["shared_manifest_path"] = str(SHARED_MANIFEST)
                elif SHARED_ONLY:
                    raise ValueError("Set SHARED_MANIFEST when SHARED_ONLY=True")

                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu"
                )
                checkpoint_path = resolve_existing_checkpoint_path(
                    MODALITY,
                    SUBJECT,
                    kind="best",
                    shared_only=SHARED_ONLY,
                    shared_manifest_path=str(SHARED_MANIFEST) if SHARED_MANIFEST else None,
                )
                clip_dict = load_clip_cache(config)
                loader = create_dataloader(
                    config,
                    MODALITY,
                    SPLIT,
                    subject=SUBJECT,
                    shared_only=SHARED_ONLY,
                    quiet=False,
                    shuffle=False,
                )
                sample_x = loader.dataset[0]["x"]
                model = build_model(config, MODALITY, sample_x, device)
                load_checkpoint(model, checkpoint_path, device)
                modality_embeddings = collect_modality_embeddings(model, loader, device)
                query_ids = sorted(modality_embeddings)[:NUM_QUERIES]

                union_manifest = pd.read_csv(ROOT / "data" / "manifests" / "all_modalities_union.tsv", sep="\\t")
                image_paths = dict(zip(union_manifest["image_id"], union_manifest["relative_path"]))
                things_root = ROOT / config["data"]["things_image_root"]
                """
            ),
            code_cell(
                """
                candidate_ids = sorted(modality_embeddings)
                brain_matrix = np.stack([modality_embeddings[image_id] for image_id in candidate_ids], axis=0).astype(np.float32)
                clip_matrix = clip_embeddings_for_ids(clip_dict, candidate_ids)
                similarity = brain_matrix @ clip_matrix.T
                id_to_index = {image_id: index for index, image_id in enumerate(candidate_ids)}

                def load_image(image_id):
                    relative_path = image_paths.get(image_id)
                    if relative_path is None:
                        raise KeyError(f"Missing relative_path for {image_id}")
                    return Image.open(things_root / relative_path).convert("RGB")

                fig, axes = plt.subplots(len(query_ids), TOPK + 1, figsize=(3 * (TOPK + 1), 3 * len(query_ids)))
                if len(query_ids) == 1:
                    axes = np.expand_dims(axes, axis=0)

                for row_index, image_id in enumerate(query_ids):
                    query_index = id_to_index[image_id]
                    ranked = np.argsort(similarity[query_index])[::-1][:TOPK]
                    ranked_ids = [candidate_ids[index] for index in ranked]

                    axes[row_index, 0].imshow(load_image(image_id))
                    axes[row_index, 0].set_title(f"Query\\n{image_id}")
                    axes[row_index, 0].axis("off")

                    for col_index, retrieved_id in enumerate(ranked_ids, start=1):
                        axes[row_index, col_index].imshow(load_image(retrieved_id))
                        axes[row_index, col_index].set_title(f"Top-{col_index}\\n{retrieved_id}")
                        axes[row_index, col_index].axis("off")

                plt.tight_layout()
                plt.show()
                """
            ),
        ],
    )

    print("Notebook refresh complete.")


if __name__ == "__main__":
    main()
