import os
import shutil
import argparse
import pandas as pd
from pathlib import Path

def sanitize_folder_name(name):
    """Replace special characters and spaces for clean folder names."""
    return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)

def sample_category(group, target_folder, num_samples, use_symlinks):
    """Sample up to num_samples from a DataFrame group and copy or symlink files."""
    target_folder.mkdir(parents=True, exist_ok=True)
    sample_size = min(len(group), num_samples)
    
    if sample_size == 0:
        return 0

    sampled_df = group.sample(n=sample_size, random_state=42)

    for _, row in sampled_df.iterrows():
        src_file = Path(row['file_path'])
        
        if not src_file.exists():
            print(f"  [Warning] Missing file: {src_file}")
            continue

        dest_file = target_folder / src_file.name

        if use_symlinks:
            if dest_file.exists() or dest_file.is_symlink():
                dest_file.unlink()
            os.symlink(src_file.resolve(), dest_file)
        else:
            shutil.copy2(src_file, dest_file)

    return sample_size

def main():
    parser = argparse.ArgumentParser(description="Sample images based on CSV tags for verification.")
    parser.add_argument("--csv", type=str, default="dataset_environment_breakdown.csv", help="Input CSV path")
    parser.add_argument("--out_dir", type=str, default="verification_samples", help="Output directory for samples")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of images to sample per category")
    parser.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying files")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' not found.")
        return

    df = pd.read_csv(args.csv)
    out_path = Path(args.out_dir)

    print(f"Reading {len(df)} entries from '{args.csv}'...\n")

    # 1. Sample by Primary Biome
    if 'primary_biome' in df.columns:
        print("=== Sampling Biomes ===")
        biome_base_dir = out_path / "biomes"
        
        for biome, group in df.groupby('primary_biome'):
            folder_name = sanitize_folder_name(biome)
            target_dir = biome_base_dir / folder_name
            count = sample_category(group, target_dir, args.num_samples, args.symlink)
            print(f"  - {biome}: Sampled {count}/{len(group)} images -> {target_dir}")

    # 2. Sample by Environmental Factors (Multi-Label Flags)
    excluded_cols = {'file_path', 'filename', 'primary_biome', 'biome_confidence'}
    factor_cols = [c for c in df.columns if c not in excluded_cols]

    if factor_cols:
        print("\n=== Sampling Environmental Factors ===")
        factor_base_dir = out_path / "factors"

        for factor in factor_cols:
            # Handle boolean or string representation of booleans
            positive_mask = df[factor].astype(str).str.lower().isin(['true', '1'])
            group = df[positive_mask]
            
            folder_name = sanitize_folder_name(factor)
            target_dir = factor_base_dir / folder_name
            count = sample_category(group, target_dir, args.num_samples, args.symlink)
            print(f"  - {factor}: Sampled {count}/{len(group)} images -> {target_dir}")

    print(f"\nDone! Sampled images are organized in: {out_path.resolve()}")

if __name__ == "__main__":
    main()