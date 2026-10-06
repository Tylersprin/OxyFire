import os
import argparse
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import open_clip

# --- Dataset Handler ---
class WildfireImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            tensor = self.transform(image)
            return tensor, str(path), True
        except Exception as e:
            # Return dummy tensor if image is corrupt
            return torch.zeros((3, 224, 224)), str(path), False

def main(data_dir, output_csv, batch_size, num_workers, device_name):
    device = device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    print(f"Using device: {device}")

    # 1. Load OpenCLIP Model & Tokenizer
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    print(f"Loading OpenCLIP model ({model_name} / {pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    # 2. Define Category Labels & Prompts
    biomes = [
        "boreal forest or woodland", 
        "industrial site or facility", 
        "tundra or arctic landscape", 
        "grassland or savanna", 
        "residential or urban zone",
        "agricultural field",
        "indiscernable"
    ]
    
    factors = [
        "lens glare or bright glare",
        "cloudy sky",
        "heavy fog or mist",
        "smoke or haze",
        "snow covered ground",
        "nighttime"
    ]

    biome_prompts = [f"a photo of a {b}" for b in biomes]
    factor_prompts = [f"a photo with {f}" for f in factors]
    factor_absent_prompts = [f"a photo without {f}" for f in factors]

    # 3. Pre-encode Text Embeddings
    with torch.no_grad():
        b_tokens = tokenizer(biome_prompts).to(device)
        f_tokens = tokenizer(factor_prompts).to(device)
        f_absent_tokens = tokenizer(factor_absent_prompts).to(device)
        
        biome_text_feats = model.encode_text(b_tokens)
        factor_text_feats = model.encode_text(f_tokens)
        factor_absent_text_feats = model.encode_text(f_absent_tokens)
        
        biome_text_feats /= biome_text_feats.norm(dim=-1, keepdim=True)
        factor_text_feats /= factor_text_feats.norm(dim=-1, keepdim=True)
        factor_absent_text_feats /= factor_absent_text_feats.norm(dim=-1, keepdim=True)

    # 4. Gather Images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = [p for p in Path(data_dir).rglob("*") if p.suffix.lower() in valid_exts]
    print(f"Found {len(image_paths)} images in '{data_dir}'.")

    if not image_paths:
        print("No valid images found. Exiting.")
        return

    dataset = WildfireImageDataset(image_paths, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    # 5. Batch Inference Loop
    records = []
    print("Running inference...")
    with torch.no_grad():
        for images, paths, valid_flags in loader:
            images = images.to(device)
            img_feats = model.encode_image(images)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)

            # Cosine Similarities (scaled by 100 as per CLIP standard)
            biome_logits = 100.0 * (img_feats @ biome_text_feats.T)
            factor_logits = 100.0 * (img_feats @ factor_text_feats.T)
            factor_absent_logits = 100.0 * (img_feats @ factor_absent_text_feats.T)

            biome_probs = biome_logits.softmax(dim=-1)
            factor_pair_logits = torch.stack(
                (factor_logits, factor_absent_logits), dim=-1
            )
            factor_probs = factor_pair_logits.softmax(dim=-1)[..., 0]

            for i in range(len(paths)):
                if not valid_flags[i]:
                    continue
                    
                path = paths[i]
                
                # Top biome class
                top_biome_idx = biome_probs[i].argmax().item()
                assigned_biome = biomes[top_biome_idx]
                biome_confidence = round(biome_probs[i][top_biome_idx].item(), 3)

                # Each factor competes against its explicit absence prompt.
                detected_factors = {
                    f: bool(factor_probs[i][j].item() > 0.5)
                    for j, f in enumerate(factors)
                }

                records.append({
                    "file_path": path,
                    "filename": Path(path).name,
                    "primary_biome": assigned_biome,
                    "biome_confidence": biome_confidence,
                    **detected_factors
                })

    # 6. Export Results
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to: {output_csv}")

    print("\n=== Biome Distribution ===")
    print(df['primary_biome'].value_counts())

    print("\n=== Environmental Factor Counts ===")
    print(df[factors].sum())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to images folder")
    parser.add_argument("--output", type=str, default="dataset_environment_breakdown.csv", help="Output CSV name")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--device", type=str, default=None, help="Device, for example cuda or cpu")
    args = parser.parse_args()

    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch_size must be positive and --num_workers cannot be negative")
    main(args.data_dir, args.output, args.batch_size, args.num_workers, args.device)