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
        NOTEBOOKS_DIR / "04_paper_tables.ipynb",
        [
            md_cell(
                """
                # Paper Tables

                This notebook builds the paper-facing tables from `results/summary/`.

                Recommended placement:

                - Main paper: native retrieval and pairwise conversion
                - Appendix: shared-retrieval context and 3-way conversion
                """
            ),
            code_cell(
                """
                from pathlib import Path
                import sys

                import pandas as pd
                from IPython.display import display

                ROOT = Path.cwd().resolve()
                if ROOT.name == "notebooks":
                    ROOT = ROOT.parent
                sys.path.insert(0, str(ROOT))

                SUMMARY_DIR = ROOT / "results" / "summary"
                TABLE_DIR = ROOT / "results" / "paper" / "tables"
                TABLE_DIR.mkdir(parents=True, exist_ok=True)

                retrieval_summary = pd.read_csv(SUMMARY_DIR / "retrieval_summary.csv")
                conversion_summary = pd.read_csv(SUMMARY_DIR / "conversion_summary.csv")
                """
            ),
            code_cell(
                """
                POOL_LABELS = {
                    "none": "Native",
                    "shared-eeg-meg": "EEG↔MEG",
                    "shared-eeg-fmri": "EEG↔fMRI",
                    "shared-fmri-meg": "MEG↔fMRI",
                    "shared-eeg-fmri-meg": "EEG↔MEG↔fMRI",
                }
                MODALITY_ORDER = {"eeg": 0, "meg": 1, "fmri": 2}

                def pct(mean, std):
                    return f"{mean:.2f} ± {std:.2f}"

                def ratio(mean, std):
                    return f"{mean:.3f} ± {std:.3f}"

                def ratio_only(mean):
                    return f"{mean:.3f}"

                def pool_label(shared_group):
                    return POOL_LABELS.get(shared_group, shared_group)

                def save_table(df, stem):
                    csv_path = TABLE_DIR / f"{stem}.csv"
                    tex_path = TABLE_DIR / f"{stem}.tex"
                    df.to_csv(csv_path, index=False)
                    tex_path.write_text(df.to_latex(index=False, escape=False))
                    print(f"Saved {csv_path}")
                    print(f"Saved {tex_path}")
                    return df

                def build_native_retrieval_table():
                    df = retrieval_summary.query("evaluation_scope == 'full' and shared_only == False").copy()
                    df = df.sort_values("modality", key=lambda col: col.map(MODALITY_ORDER))
                    return pd.DataFrame(
                        {
                            "Modality": df["modality"].str.upper(),
                            "Candidate Set": df["retrieval_dataset_size_mean"].round(0).astype(int).astype(str) + "-way",
                            "Top-1 (%)": [pct(m, s) for m, s in zip(df["m2i_top1_mean"], df["m2i_top1_std"])],
                            "Top-5 (%)": [pct(m, s) for m, s in zip(df["m2i_top5_mean"], df["m2i_top5_std"])],
                            "CLIP 2-way (%)": [pct(m, s) for m, s in zip(df["m2i_two_way_mean"], df["m2i_two_way_std"])],
                        }
                    )

                def build_shared_retrieval_context_table():
                    shared_df = retrieval_summary.query("shared_only == True").copy()
                    full_lookup = {
                        row["modality"]: row["m2i_two_way_mean"]
                        for _, row in retrieval_summary.query("evaluation_scope == 'full' and shared_only == False").iterrows()
                    }
                    shared_df["retained_two_way"] = shared_df.apply(
                        lambda row: row["m2i_two_way_mean"] / full_lookup[row["modality"]],
                        axis=1,
                    )
                    shared_df = shared_df.sort_values(
                        ["modality", "evaluation_scope", "shared_group"],
                        key=lambda col: col.map(MODALITY_ORDER) if col.name == "modality" else col,
                    )
                    return pd.DataFrame(
                        {
                            "Modality": shared_df["modality"].str.upper(),
                            "Scope": shared_df["evaluation_scope"].str.replace("_", " ").str.title(),
                            "Shared Pool": shared_df["shared_group"].map(pool_label),
                            "Top-1 (%)": [pct(m, s) for m, s in zip(shared_df["m2i_top1_mean"], shared_df["m2i_top1_std"])],
                            "CLIP 2-way (%)": [pct(m, s) for m, s in zip(shared_df["m2i_two_way_mean"], shared_df["m2i_two_way_std"])],
                            "2-way / Native": [ratio_only(m) for m in shared_df["retained_two_way"]],
                        }
                    )

                def build_conversion_table(scope):
                    df = conversion_summary.query("evaluation_scope == @scope").copy()
                    df = df.sort_values(["shared_group", "source_modality", "target_modality"])
                    rows = []
                    for _, row in df.iterrows():
                        rows.append(
                            {
                                "Shared Pool": pool_label(row["shared_group"]),
                                "Direction": f"{row['source_modality'].upper()}→{row['target_modality'].upper()}",
                                "Top-1 (%)": pct(row["forward_top1_mean"], row["forward_top1_std"]),
                                "Top-5 (%)": pct(row["forward_top5_mean"], row["forward_top5_std"]),
                                "CLIP 2-way (%)": pct(row["forward_two_way_mean"], row["forward_two_way_std"]),
                                "2-way / matched retrieval": ratio(
                                    row["forward_normalized_two_way_mean"],
                                    row["forward_normalized_two_way_std"],
                                ),
                                "2-way / native retrieval": ratio(
                                    row["forward_normalized_two_way_full_mean"],
                                    row["forward_normalized_two_way_full_std"],
                                ),
                            }
                        )
                        rows.append(
                            {
                                "Shared Pool": pool_label(row["shared_group"]),
                                "Direction": f"{row['target_modality'].upper()}→{row['source_modality'].upper()}",
                                "Top-1 (%)": pct(row["reverse_top1_mean"], row["reverse_top1_std"]),
                                "Top-5 (%)": pct(row["reverse_top5_mean"], row["reverse_top5_std"]),
                                "CLIP 2-way (%)": pct(row["reverse_two_way_mean"], row["reverse_two_way_std"]),
                                "2-way / matched retrieval": ratio(
                                    row["reverse_normalized_two_way_mean"],
                                    row["reverse_normalized_two_way_std"],
                                ),
                                "2-way / native retrieval": ratio(
                                    row["reverse_normalized_two_way_full_mean"],
                                    row["reverse_normalized_two_way_full_std"],
                                ),
                            }
                        )
                    return pd.DataFrame(rows)
                """
            ),
            md_cell(
                """
                ## Main-Paper Tables

                These are the most compact tables for the nine-page NeurIPS main paper:

                - native retrieval performance on each modality's native evaluation protocol
                - pairwise shared conversion using matched-scope normalization
                """
            ),
            code_cell(
                """
                native_retrieval_table = save_table(build_native_retrieval_table(), "table1_native_retrieval")
                display(native_retrieval_table)
                """
            ),
            code_cell(
                """
                pairwise_conversion_table = save_table(build_conversion_table("pair"), "table2_pairwise_conversion")
                display(pairwise_conversion_table)
                """
            ),
            md_cell(
                """
                ## Appendix Tables

                These tables help interpret the main results but are better suited for the appendix:

                - how much each shared pool degrades retrieval relative to the native setting
                - three-way shared conversion, which is a stronger stress test than the pairwise main benchmark
                """
            ),
            code_cell(
                """
                shared_retrieval_context_table = save_table(
                    build_shared_retrieval_context_table(),
                    "tableA1_shared_retrieval_context",
                )
                display(shared_retrieval_context_table)
                """
            ),
            code_cell(
                """
                three_way_conversion_table = save_table(
                    build_conversion_table("three_way"),
                    "tableA2_three_way_conversion",
                )
                display(three_way_conversion_table)
                """
            ),
        ],
    )

    write_notebook(
        NOTEBOOKS_DIR / "05_paper_figures.ipynb",
        [
            md_cell(
                """
                # Paper Figures

                This notebook creates paper-facing figures from `results/summary/`.

                Recommended placement:

                - Main paper: shared-retrieval context and pairwise-vs-3-way conversion overview
                - Appendix: subject-pair heatmaps for the pairwise conversion benchmark
                """
            ),
            code_cell(
                """
                from pathlib import Path
                import sys

                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd

                ROOT = Path.cwd().resolve()
                if ROOT.name == "notebooks":
                    ROOT = ROOT.parent
                sys.path.insert(0, str(ROOT))

                SUMMARY_DIR = ROOT / "results" / "summary"
                FIGURE_DIR = ROOT / "results" / "paper" / "figures"
                FIGURE_DIR.mkdir(parents=True, exist_ok=True)

                retrieval_summary = pd.read_csv(SUMMARY_DIR / "retrieval_summary.csv")
                conversion_summary = pd.read_csv(SUMMARY_DIR / "conversion_summary.csv")
                conversion_by_pair = pd.read_csv(SUMMARY_DIR / "conversion_by_pair.csv")

                plt.rcParams.update(
                    {
                        "figure.dpi": 140,
                        "savefig.dpi": 300,
                        "axes.spines.top": False,
                        "axes.spines.right": False,
                    }
                )
                """
            ),
            code_cell(
                """
                POOL_LABELS = {
                    "shared-eeg-meg": "EEG↔MEG",
                    "shared-eeg-fmri": "EEG↔fMRI",
                    "shared-fmri-meg": "MEG↔fMRI",
                    "shared-eeg-fmri-meg": "EEG↔MEG↔fMRI",
                }
                MODALITY_ORDER = ["eeg", "meg", "fmri"]
                COLOR_BY_POOL = {
                    "shared-eeg-meg": "#355070",
                    "shared-eeg-fmri": "#6D597A",
                    "shared-fmri-meg": "#B56576",
                    "shared-eeg-fmri-meg": "#E56B6F",
                }

                def pool_label(shared_group):
                    return POOL_LABELS.get(shared_group, shared_group)

                def save_figure(fig, stem):
                    png_path = FIGURE_DIR / f"{stem}.png"
                    pdf_path = FIGURE_DIR / f"{stem}.pdf"
                    fig.savefig(png_path, bbox_inches="tight")
                    fig.savefig(pdf_path, bbox_inches="tight")
                    print(f"Saved {png_path}")
                    print(f"Saved {pdf_path}")

                def retrieval_retention_frame():
                    full_lookup = {
                        row["modality"]: row["m2i_two_way_mean"]
                        for _, row in retrieval_summary.query("evaluation_scope == 'full' and shared_only == False").iterrows()
                    }
                    shared_df = retrieval_summary.query("shared_only == True").copy()
                    shared_df["retained_two_way"] = shared_df.apply(
                        lambda row: row["m2i_two_way_mean"] / full_lookup[row["modality"]],
                        axis=1,
                    )
                    return shared_df

                def expanded_conversion_summary(scope):
                    rows = []
                    df = conversion_summary.query("evaluation_scope == @scope").copy()
                    for _, row in df.iterrows():
                        rows.append(
                            {
                                "Direction": f"{row['source_modality'].upper()}→{row['target_modality'].upper()}",
                                "Shared Pool": pool_label(row["shared_group"]),
                                "shared_group": row["shared_group"],
                                "normalized_two_way": row["forward_normalized_two_way_mean"],
                                "normalized_two_way_std": row["forward_normalized_two_way_std"],
                                "two_way": row["forward_two_way_mean"],
                            }
                        )
                        rows.append(
                            {
                                "Direction": f"{row['target_modality'].upper()}→{row['source_modality'].upper()}",
                                "Shared Pool": pool_label(row["shared_group"]),
                                "shared_group": row["shared_group"],
                                "normalized_two_way": row["reverse_normalized_two_way_mean"],
                                "normalized_two_way_std": row["reverse_normalized_two_way_std"],
                                "two_way": row["reverse_two_way_mean"],
                            }
                        )
                    return pd.DataFrame(rows)
                """
            ),
            code_cell(
                """
                retention_df = retrieval_retention_frame()
                fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)

                for ax, modality in zip(axes, MODALITY_ORDER):
                    subset = retention_df[retention_df["modality"] == modality].copy()
                    subset = subset.sort_values(["evaluation_scope", "shared_group"])
                    x = np.arange(len(subset))
                    colors = [COLOR_BY_POOL[group] for group in subset["shared_group"]]
                    ax.bar(x, subset["retained_two_way"], color=colors)
                    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
                    ax.set_ylim(0.45, 1.05)
                    ax.set_title(modality.upper())
                    ax.set_xticks(x)
                    ax.set_xticklabels(
                        [pool_label(group) for group in subset["shared_group"]],
                        rotation=35,
                        ha="right",
                    )
                    ax.set_ylabel("Retained CLIP 2-way vs native")

                fig.suptitle("Shared-pool retrieval context")
                fig.tight_layout(rect=(0, 0, 1, 0.96))
                save_figure(fig, "fig1_shared_retrieval_context")
                plt.show()
                """
            ),
            code_cell(
                """
                pair_df = expanded_conversion_summary("pair")
                three_way_df = expanded_conversion_summary("three_way")
                direction_order = [
                    "EEG→MEG",
                    "MEG→EEG",
                    "EEG→FMRI",
                    "FMRI→EEG",
                    "MEG→FMRI",
                    "FMRI→MEG",
                ]

                fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), sharey=True)
                for ax, data, title in [
                    (axes[0], pair_df, "Pairwise shared conversion"),
                    (axes[1], three_way_df, "Three-way shared conversion"),
                ]:
                    data = data.set_index("Direction").loc[direction_order].reset_index()
                    x = np.arange(len(data))
                    colors = [COLOR_BY_POOL[group] for group in data["shared_group"]]
                    ax.bar(
                        x,
                        data["normalized_two_way"],
                        yerr=data["normalized_two_way_std"],
                        capsize=4,
                        color=colors,
                    )
                    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
                    ax.set_xticks(x)
                    ax.set_xticklabels(data["Direction"], rotation=35, ha="right")
                    ax.set_ylim(0.5, 1.05)
                    ax.set_title(title)
                    ax.set_ylabel("Conversion 2-way / matched retrieval")

                fig.tight_layout()
                save_figure(fig, "fig2_conversion_overview")
                plt.show()
                """
            ),
            code_cell(
                """
                pair_heatmap_specs = [
                    ("shared-eeg-meg", "eeg", "meg"),
                    ("shared-eeg-fmri", "eeg", "fmri"),
                    ("shared-fmri-meg", "meg", "fmri"),
                ]

                fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
                for ax, (group, source_modality, target_modality) in zip(axes, pair_heatmap_specs):
                    subset = conversion_by_pair[
                        (conversion_by_pair["evaluation_scope"] == "pair")
                        & (conversion_by_pair["shared_group"] == group)
                        & (conversion_by_pair["source_modality"] == source_modality)
                        & (conversion_by_pair["target_modality"] == target_modality)
                    ]
                    heatmap = subset.pivot(
                        index="source_subject",
                        columns="target_subject",
                        values="forward_normalized_two_way",
                    )
                    image = ax.imshow(heatmap.values, vmin=0.65, vmax=1.0, cmap="YlGnBu")
                    ax.set_title(f"{source_modality.upper()}→{target_modality.upper()}")
                    ax.set_xlabel("Target subject")
                    ax.set_ylabel("Source subject")
                    ax.set_xticks(range(len(heatmap.columns)))
                    ax.set_xticklabels(heatmap.columns)
                    ax.set_yticks(range(len(heatmap.index)))
                    ax.set_yticklabels(heatmap.index)

                fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="2-way / matched retrieval")
                save_figure(fig, "figA1_pairwise_conversion_heatmaps")
                plt.show()
                """
            ),
        ],
    )

    print("Notebook refresh complete.")


if __name__ == "__main__":
    main()
