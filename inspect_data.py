import torch
import numpy as np
import os

def inspect():
    # Берем первый попавшийся файл
    files = sorted([f for f in os.listdir("dataset_train") if f.endswith('.pt')])
    if not files:
        print("Нет данных! Сгенерируй сначала.")
        return
        
    data_path = os.path.join("dataset_train", files[0])
    data = torch.load(data_path)
    
    print(f"--- Inspection of {data_path} ---")
    print(f"Type: {data['type']}")
    
    # 1. Анализ признаков переменных
    col_feats = data['col_features'] # [N_vars, 19]
    print(f"\nVariable Features shape: {col_feats.shape}")
    
    # Считаем дисперсию
    stds = torch.std(col_feats, dim=0)
    print("Standard Deviation per feature (Variables):")
    print(stds)
    
    dead_features = (stds < 1e-6).sum()
    print(f"Dead features (const for all vars): {dead_features}/{len(stds)}")
    
    # 2. Проверка симметрии (FIXED)
    candidates = data['candidates']
    cand_feats = col_feats[candidates]
    
    # Используем torch.cdist (он внутри torch)
    # cdist считает расстояния между всеми парами строк
    dists = torch.cdist(cand_feats, cand_feats)
    
    # Среднее расстояние (исключая диагональ, где 0)
    num_pairs = len(candidates)**2 - len(candidates)
    if num_pairs > 0:
        avg_dist = dists.sum() / num_pairs
    else:
        avg_dist = 0.0
        
    print(f"\nAverage distance between candidates features: {avg_dist:.6f}")
    
    # Если расстояние слишком маленькое, значит кандидаты почти одинаковые
    if avg_dist < 1e-2: 
        print(">>> CRITICAL: Candidates are indistinguishable! (Symmetry Problem)")
        print(">>> SOLUTION: Need to add Random Noise / IDs.")
    else:
        print("OK: Candidates look different.")

    # 3. Target
    label = data['label_var_idx']
    print(f"\nTarget Variable Index: {label}")

if __name__ == "__main__":
    inspect()