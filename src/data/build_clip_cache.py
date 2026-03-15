import os
import argparse
import yaml
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.data.image_manifest import read_manifest_tsv, resolve_repo_path

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def resolve_image_paths_from_manifest(image_root, manifest_path):
    image_root = resolve_repo_path(image_root)
    manifest_df = read_manifest_tsv(manifest_path)
    image_records = []

    for _, row in manifest_df.iterrows():
        image_id = str(row["image_id"]).strip()
        relative_path = str(row.get("relative_path", "")).strip() if "relative_path" in manifest_df.columns else ""
        candidate = resolve_image_path(image_root, image_id, relative_path)
        image_records.append((image_id, candidate))

    return image_records


def resolve_image_path(image_root, image_id, relative_path):
    candidates = []
    normalized = relative_path.replace("\\", "/").strip()
    trimmed = normalized
    for prefix in ("images_THINGS/object_images/", "object_images/", "images/"):
        if trimmed.startswith(prefix):
            trimmed = trimmed[len(prefix) :]
            break

    if normalized:
        candidates.append(image_root / normalized)
    if trimmed and trimmed != normalized:
        candidates.append(image_root / trimmed)
    if trimmed:
        candidates.append(image_root / "object_images" / trimmed)
        candidates.append(image_root / "images_THINGS" / "object_images" / trimmed)
    candidates.append(image_root / f"{image_id}.jpg")

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    matches = list(image_root.rglob(f"{image_id}.jpg"))
    if not matches:
        raise FileNotFoundError(
            f"Could not resolve image '{image_id}' under {image_root}. "
            "Check the manifest relative_path values and image root."
        )
    return matches[0]


def build_clip_cache(config_path="config.yaml", manifest_path=None, image_root=None):
    config = load_config(config_path)
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    clip_model_name = config["model"]["clip_name"]
    print(f"Loading CLIP model: {clip_model_name}")
    model, preprocess = clip.load(clip_model_name, device=device)
    model.eval()

    if manifest_path:
        image_root = image_root or config["data"].get("things_image_root")
        if not image_root:
            raise ValueError("Building a manifest-based CLIP cache requires --image-root or data.things_image_root")
        image_records = resolve_image_paths_from_manifest(image_root, manifest_path)
        print(f"Found {len(image_records)} manifest images. Building cache...")
    else:
        eeg_stimuli_dir = Path(config["data"]["eeg_stimuli_dir"])
        img_paths = list(eeg_stimuli_dir.rglob("*.jpg"))
        if len(img_paths) == 0:
            print(f"No images found in {eeg_stimuli_dir}!")
            return
        image_records = [(path.stem, path) for path in img_paths]
        print(f"Found {len(image_records)} images. Building cache...")
    
    cache_dir = Path(config["data"]["clip_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file_name = clip_model_name.replace("/", "-") + ".npz"
    cache_path = cache_dir / cache_file_name
    
    embeddings = {}
    
    batch_size = 128
    for i in tqdm(range(0, len(image_records), batch_size), desc="Processing Batches"):
        batch_records = image_records[i:i+batch_size]
        batch_ids = [image_id for image_id, _ in batch_records]
        batch_paths = [path for _, path in batch_records]
        
        try:
            images = [preprocess(Image.open(p)).unsqueeze(0) for p in batch_paths]
            images_tensor = torch.cat(images).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(images_tensor)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                
            features_np = image_features.cpu().numpy().astype(np.float32)
            
            for img_id, feat in zip(batch_ids, features_np):
                embeddings[img_id] = feat
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            continue
            
    print(f"Saving cache to {cache_path}...")
    np.savez_compressed(cache_path, **embeddings)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CLIP feature cache for stimuli")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--manifest", type=str, default=None, help="Optional manifest TSV with image_id/relative_path")
    parser.add_argument("--image-root", type=str, default=None, help="Root directory containing the source images")
    args = parser.parse_args()
    build_clip_cache(args.config, manifest_path=args.manifest, image_root=args.image_root)
