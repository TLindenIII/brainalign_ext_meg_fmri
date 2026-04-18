import re
from pathlib import Path


EXPERIMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

MEG_ARCH_PRIORITY = {
    "_temporalcnn": 0,
    "_attnpool": 1,
    "": 2,
}


def normalize_experiment_name(experiment_name=None):
    if experiment_name is None:
        return None
    experiment_name = str(experiment_name).strip()
    if not experiment_name:
        return None
    if not EXPERIMENT_NAME_PATTERN.match(experiment_name):
        raise ValueError(
            "Experiment names must start with a letter or number and only contain "
            "letters, numbers, '.', '_', or '-'."
        )
    return experiment_name


def checkpoint_root(experiment_name=None):
    experiment_name = normalize_experiment_name(experiment_name)
    if experiment_name is None:
        return Path("checkpoints")
    return Path("checkpoints") / "experiments" / experiment_name


def results_root(experiment_name=None):
    experiment_name = normalize_experiment_name(experiment_name)
    if experiment_name is None:
        return Path("results")
    return Path("results") / "experiments" / experiment_name


def conversion_manifest_slug(shared_manifest_path=None, modalities=None):
    if shared_manifest_path:
        slug = Path(shared_manifest_path).stem.strip().lower().replace("-", "_")
    elif modalities:
        slug = "_".join(sorted(modality.strip().lower() for modality in modalities))
    else:
        raise ValueError("Shared checkpoint resolution requires a shared manifest path or modalities")

    if not slug:
        raise ValueError("Could not derive a shared conversion slug")
    return slug


def conversion_directory_name(shared_manifest_path=None, modalities=None):
    slug = conversion_manifest_slug(
        shared_manifest_path=shared_manifest_path,
        modalities=modalities,
    )
    return f"shared-{slug.replace('_', '-')}"


def evaluation_scope_for(shared_manifest_path=None, modalities=None):
    slug = conversion_manifest_slug(
        shared_manifest_path=shared_manifest_path,
        modalities=modalities,
    )
    modality_count = len([token for token in slug.split("_") if token])
    if modality_count >= 3:
        return "three_way"
    if modality_count == 2:
        return "pair"
    return "shared"


def retrieval_results_dir(modality, evaluation_scope, shared_group="none", experiment_name=None):
    results_dir = results_root(experiment_name) / "retrieval" / evaluation_scope
    if shared_group and shared_group != "none" and evaluation_scope != "full":
        results_dir = results_dir / shared_group
    return results_dir / modality


def retrieval_results_path(modality, subject, split, evaluation_scope, shared_group="none", experiment_name=None):
    results_dir = retrieval_results_dir(modality, evaluation_scope, shared_group, experiment_name=experiment_name)
    return results_dir / f"evaluation_sub{subject:02d}_{split}.txt"


def conversion_results_dir(evaluation_scope, shared_group="none", experiment_name=None):
    results_dir = results_root(experiment_name) / "conversion" / evaluation_scope
    if shared_group and shared_group != "none":
        results_dir = results_dir / shared_group
    return results_dir


def conversion_results_path(
    source_modality,
    source_subject,
    target_modality,
    target_subject,
    split,
    evaluation_scope,
    shared_group="none",
    experiment_name=None,
):
    results_dir = conversion_results_dir(evaluation_scope, shared_group, experiment_name=experiment_name)
    return (
        results_dir
        / (
            f"{source_modality}_sub{source_subject:02d}_to_"
            f"{target_modality}_sub{target_subject:02d}_{split}.txt"
        )
    )


def checkpoint_stem_for(modality, subject, arch_variant="current"):
    stem = f"{modality}_brainalign_sub{subject:02d}"
    if modality == "meg":
        if arch_variant == "current":
            stem += "_temporalcnn"
        elif arch_variant == "attnpool":
            stem += "_attnpool"
        elif arch_variant != "none":
            raise ValueError(f"Unsupported MEG checkpoint arch variant '{arch_variant}'")
    return stem


def checkpoint_dir(modality, shared_only=False, shared_manifest_path=None, modalities=None, experiment_name=None):
    if not shared_only:
        return checkpoint_root(experiment_name) / modality

    return (
        checkpoint_root(experiment_name)
        / "conversion"
        / conversion_directory_name(
            shared_manifest_path=shared_manifest_path,
            modalities=modalities,
        )
        / modality
    )


def checkpoint_paths_for(
    modality,
    subject,
    shared_only=False,
    shared_manifest_path=None,
    modalities=None,
    experiment_name=None,
):
    save_dir = checkpoint_dir(
        modality,
        shared_only=shared_only,
        shared_manifest_path=shared_manifest_path,
        modalities=modalities,
        experiment_name=experiment_name,
    )
    stem = checkpoint_stem_for(modality, subject)
    return {
        "save_dir": save_dir,
        "stem": stem,
        "best": save_dir / f"{stem}_best.pt",
        "latest": save_dir / f"{stem}_latest.pt",
    }


def candidate_checkpoint_paths(
    modality,
    subject,
    kind="best",
    shared_only=False,
    shared_manifest_path=None,
    modalities=None,
    experiment_name=None,
):
    suffix = f"_{kind}.pt"
    candidates = []

    if shared_only:
        primary_dir = checkpoint_dir(
            modality,
            shared_only=True,
            shared_manifest_path=shared_manifest_path,
            modalities=modalities,
            experiment_name=experiment_name,
        )
        candidates.append(primary_dir / f"{checkpoint_stem_for(modality, subject)}{suffix}")
        if modality == "meg":
            candidates.append(primary_dir / f"{checkpoint_stem_for(modality, subject, arch_variant='attnpool')}{suffix}")

        # Legacy shared checkpoint fallback is only enabled for the base run.
        # Experimental branches must resolve inside checkpoints/experiments/<name>/.
        if normalize_experiment_name(experiment_name) is None:
            legacy_dir = Path("checkpoints") / modality
            candidates.append(legacy_dir / f"{checkpoint_stem_for(modality, subject)}_shared{suffix}")
            if modality == "meg":
                candidates.append(
                    legacy_dir / f"{checkpoint_stem_for(modality, subject, arch_variant='attnpool')}_shared{suffix}"
                )
    else:
        primary_dir = checkpoint_dir(modality, shared_only=False, experiment_name=experiment_name)
        candidates.append(primary_dir / f"{checkpoint_stem_for(modality, subject)}{suffix}")
        if modality == "meg":
            candidates.append(primary_dir / f"{checkpoint_stem_for(modality, subject, arch_variant='attnpool')}{suffix}")

    deduped = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def resolve_existing_checkpoint_path(
    modality,
    subject,
    kind="best",
    shared_only=False,
    shared_manifest_path=None,
    modalities=None,
    experiment_name=None,
):
    candidates = candidate_checkpoint_paths(
        modality,
        subject,
        kind=kind,
        shared_only=shared_only,
        shared_manifest_path=shared_manifest_path,
        modalities=modalities,
        experiment_name=experiment_name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def checkpoint_filename_pattern(modality):
    return re.compile(
        rf"^{modality}_brainalign_sub(?P<subject>\d+)"
        rf"(?P<arch>_temporalcnn|_attnpool)?"
        rf"(?P<shared>_shared)?"
        rf"_(?P<kind>best|latest)\.pt$"
    )


def discover_best_checkpoints(
    modality,
    shared_only=False,
    shared_manifest_path=None,
    modalities=None,
    experiment_name=None,
):
    locations = []
    if shared_only:
        locations.append(
            (
                checkpoint_dir(
                    modality,
                    shared_only=True,
                    shared_manifest_path=shared_manifest_path,
                    modalities=modalities,
                    experiment_name=experiment_name,
                ),
                False,
                0,
            )
        )
        if normalize_experiment_name(experiment_name) is None:
            locations.append((Path("checkpoints") / modality, True, 10))
    else:
        locations.append((checkpoint_dir(modality, shared_only=False, experiment_name=experiment_name), False, 0))

    pattern = checkpoint_filename_pattern(modality)
    selected = {}

    for directory, expect_legacy_shared_suffix, location_priority in locations:
        if not directory.exists():
            continue

        for path in directory.glob(f"{modality}_brainalign_sub*_best.pt"):
            match = pattern.match(path.name)
            if not match or match.group("kind") != "best":
                continue

            has_shared_suffix = match.group("shared") is not None
            if expect_legacy_shared_suffix and not has_shared_suffix:
                continue
            if not expect_legacy_shared_suffix and has_shared_suffix:
                continue

            subject = int(match.group("subject"))
            arch = match.group("arch") or ""
            arch_priority = MEG_ARCH_PRIORITY.get(arch, 99) if modality == "meg" else 0
            priority = (location_priority, arch_priority)
            current = selected.get(subject)
            if current is None or priority < current[0]:
                selected[subject] = (priority, path)

    return {subject: record[1] for subject, record in sorted(selected.items())}
