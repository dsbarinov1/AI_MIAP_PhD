import torch
import os
import numpy as np

def check_edges():
    # Берем первый файл
    files = sorted([f for f in os.listdir("dataset_train") if f.endswith('.pt')])
    path = os.path.join("dataset_train", files[0])
    data = torch.load(path)
    
    attr = data['edge_attr'] # [E, 1]
    
    print(f"File: {path}")
    print(f"Edge Attr Shape: {attr.shape}")
    print(f"Min: {attr.min().item()}")
    print(f"Max: {attr.max().item()}")
    print(f"Mean: {attr.mean().item()}")
    
    # Уникальные значения
    unique, counts = torch.unique(attr, return_counts=True)
    print("\nUnique values in edge weights:")
    for val, count in zip(unique.tolist(), counts.tolist()):
        print(f"  Value: {val:.6f} | Count: {count}")
        
    if len(unique) == 1 and abs(unique[0]) < 1e-6:
        print("\n>>> VERDICT: EDGES ARE DEAD (ZERO). GCN IS BROKEN. <<<")
    elif len(unique) == 1 and abs(unique[0] - 1.0) < 1e-6:
        print("\n>>> VERDICT: All weights are 1.0. This is CORRECT for MIAP.")
    else:
        print("\n>>> VERDICT: Weights vary. Check if this makes sense.")
        
def check_avg_candidates():
    files = sorted([f for f in os.listdir("dataset_train") if f.endswith('.pt')])
    total_cands = 0
    for f in files:
        d = torch.load(os.path.join("dataset_train", f))
        total_cands += len(d['candidates'])
    
    print(f"Average candidates per instance: {total_cands / len(files):.2f}")

if __name__ == "__main__":
    check_edges()
    check_avg_candidates()