import os
import argparse
import yaml
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_clip_cache(config_path="config.yaml"):
    config = load_config(config_path)
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    clip_model_name = config["model"]["clip_name"]
    print(f"Loading CLIP model: {clip_model_name}")
    model, preprocess = clip.load(clip_model_name, device=device)
    model.eval()

    eeg_stimuli_dir = Path(config["data"]["eeg_stimuli_dir"])
    
    # THINGS-EEG2 has training_images/ and test_images/
    img_paths = list(eeg_stimuli_dir.rglob("*.jpg"))
    
    if len(img_paths) == 0:
        print(f"No images found in {eeg_stimuli_dir}!")
        return

    print(f"Found {len(img_paths)} images. Building cache...")
    
    cache_dir = Path(config["data"]["clip_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file_name = clip_model_name.replace("/", "-") + ".npz"
    cache_path = cache_dir / cache_file_name
    
    embeddings = {}
    
    batch_size = 128
    for i in tqdm(range(0, len(img_paths), batch_size), desc="Processing Batches"):
        batch_paths = img_paths[i:i+batch_size]
        
        # We use the filename without extension as the image_id
        # Example: 'aardvark_01b.jpg' -> 'aardvark_01b'
        batch_ids = [p.stem for p in batch_paths]
        
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
    args = parser.parse_args()
    build_clip_cache(args.config)
