import re
from pathlib import Path


MEG_ARCH_PRIORITY = {
    "_temporalcnn": 0,
    "_attnpool": 1,
    "": 2,
}


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


def checkpoint_dir(modality, shared_only=False, shared_manifest_path=None, modalities=None):
    if not shared_only:
        return Path("checkpoints") / modality

    return (
        Path("checkpoints")
        / "conversion"
        / conversion_directory_name(
            shared_manifest_path=shared_manifest_path,
            modalities=modalities,
        )
        / modality
    )


def checkpoint_paths_for(modality, subject, shared_only=False, shared_manifest_path=None, modalities=None):
    save_dir = checkpoint_dir(
        modality,
        shared_only=shared_only,
        shared_manifest_path=shared_manifest_path,
        modalities=modalities,
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
):
    suffix = f"_{kind}.pt"
    candidates = []

    if shared_only:
        primary_dir = checkpoint_dir(
            modality,
            shared_only=True,
            shared_manifest_path=shared_manifest_path,
            modalities=modalities,
        )
        candidates.append(primary_dir / f"{checkpoint_stem_for(modality, subject)}{suffix}")
        if modality == "meg":
            candidates.append(primary_dir / f"{checkpoint_stem_for(modality, subject, arch_variant='attnpool')}{suffix}")

        legacy_dir = Path("checkpoints") / modality
        candidates.append(legacy_dir / f"{checkpoint_stem_for(modality, subject)}_shared{suffix}")
        if modality == "meg":
            candidates.append(
                legacy_dir / f"{checkpoint_stem_for(modality, subject, arch_variant='attnpool')}_shared{suffix}"
            )
    else:
        primary_dir = checkpoint_dir(modality, shared_only=False)
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
):
    candidates = candidate_checkpoint_paths(
        modality,
        subject,
        kind=kind,
        shared_only=shared_only,
        shared_manifest_path=shared_manifest_path,
        modalities=modalities,
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


def discover_best_checkpoints(modality, shared_only=False, shared_manifest_path=None, modalities=None):
    locations = []
    if shared_only:
        locations.append(
            (
                checkpoint_dir(
                    modality,
                    shared_only=True,
                    shared_manifest_path=shared_manifest_path,
                    modalities=modalities,
                ),
                False,
                0,
            )
        )
        locations.append((Path("checkpoints") / modality, True, 10))
    else:
        locations.append((checkpoint_dir(modality, shared_only=False), False, 0))

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
