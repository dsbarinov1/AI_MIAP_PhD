import os
import torch
import numpy as np
from tqdm import tqdm
import sys

def analyze_dataset(root_dir, num_samples=500):
    files = sorted([f for f in os.listdir(root_dir) if f.endswith('.pt')])
    if not files:
        print(f"No .pt files found in {root_dir}")
        return

    # Limit samples for speed
    files = files[:num_samples]
    
    print(f"Analyzing {len(files)} samples from {root_dir}...")

    # Accumulators
    all_col_feats = []
    all_row_feats = []
    
    # Store correlation data?
    # Maybe later. First just distribution.

    for f in tqdm(files):
        path = os.path.join(root_dir, f)
        try:
            d = torch.load(path)
            # data_collector saved: row_features, col_features
            # dataset.py renames them to constraint.x and variable.x
            
            # Check keys
            if 'col_features' in d:
                all_col_feats.append(d['col_features'].numpy())
            elif 'variable_features' in d: # Handle potential naming variations if any
                all_col_feats.append(d['variable_features'].numpy())
                
            if 'row_features' in d:
                all_row_feats.append(d['row_features'].numpy())
                
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_col_feats:
        print("No feature data found.")
        return

    # Concatenate all
    X_col = np.concatenate(all_col_feats, axis=0)
    X_row = np.concatenate(all_row_feats, axis=0)

    print(f"\n--- Variable (Column) Features Analysis ---")
    print(f"Total Variable Observations: {X_col.shape[0]}")
    print(f"Feature Dimension: {X_col.shape[1]}")
    print(f"{'Idx':<4} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10} | {'Sparsity %':<10} | {'Const?':<6}")
    print("-" * 80)
    
    for i in range(X_col.shape[1]):
        col = X_col[:, i]
        mean = np.mean(col)
        std = np.std(col)
        mn = np.min(col)
        mx = np.max(col)
        sparsity = 100 * np.sum(col == 0) / len(col)
        is_const = (std == 0)
        
        print(f"{i:<4} | {mean:<10.4f} | {std:<10.4f} | {mn:<10.4f} | {mx:<10.4f} | {sparsity:<10.1f} | {str(is_const):<6}")

    print(f"\n--- Constraint (Row) Features Analysis ---")
    print(f"Total Constraint Observations: {X_row.shape[0]}")
    print(f"Feature Dimension: {X_row.shape[1]}")
    print(f"{'Idx':<4} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10} | {'Sparsity %':<10} | {'Const?':<6}")
    print("-" * 80)

    for i in range(X_row.shape[1]):
        col = X_row[:, i]
        mean = np.mean(col)
        std = np.std(col)
        mn = np.min(col)
        mx = np.max(col)
        sparsity = 100 * np.sum(col == 0) / len(col)
        is_const = (std == 0)
        
        print(f"{i:<4} | {mean:<10.4f} | {std:<10.4f} | {mn:<10.4f} | {mx:<10.4f} | {sparsity:<10.1f} | {str(is_const):<6}")

if __name__ == "__main__":
    dir_to_analyze = sys.argv[1] if len(sys.argv) > 1 else "dataset_train"
    analyze_dataset(dir_to_analyze)
