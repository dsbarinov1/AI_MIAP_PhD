import torch
import os
import numpy as np
import pandas as pd

def analyze_sample():
    # Берем первый попавшийся файл
    files = sorted([f for f in os.listdir("dataset_train") if f.endswith('.pt')])
    path = os.path.join("dataset_train", files[2])
    data = torch.load(path)
    
    print(f"=== ANALYZING {path} ===")
    
    # 1. Анализ ПРИЗНАКОВ (Features)
    # col_features: [N_vars, D_vars]
    # Последние 2 (или 4) колонки - это наш шум. Нас интересуют те, что ДО шума.
    cols = data['col_features']
    rows = data['row_features']
    
    # Предполагаем, что шум - это последние 4 колонки (как мы договорились)
    # Если ты добавил 4 колонки шума, то реальных фичей SCIP: Total - 4
    real_feat_dim = cols.shape[1] - 4 
    real_cols = cols[:, :real_feat_dim]
    
    print(f"\n1. Variable Features (Real SCIP features: {real_feat_dim})")
    # Считаем, сколько в каждой колонке НЕнулевых значений и уникальных значений
    for i in range(real_feat_dim):
        vals = real_cols[:, i]
        non_zeros = (vals != 0).sum().item()
        unique = len(torch.unique(vals))
        std = torch.std(vals).item()
        print(f"  Feat {i:02d}: Non-Zeros={non_zeros:<4} Unique={unique:<4} Std={std:.4f}  <-- {'DEAD?' if std < 1e-6 else 'OK'}")
        
    print(f"\n2. Row Features (Constraints)")
    for i in range(rows.shape[1]):
        vals = rows[:, i]
        std = torch.std(vals).item()
        print(f"  Feat {i:02d}: Std={std:.4f}")

    # 2. Анализ ТАРГЕТА (Labels)
    # Нам нужно понять, насколько очевиден выбор. 
    # К сожалению, в .pt файле мы сохранили только argmax (индекс).
    # Мы не сохранили сами scores (сырые значения SB). 
    # В следующий раз в data_collector надо сохранять и scores тоже.
    
    print(f"\n3. Target Info")
    target = data['label_var_idx'].item()
    print(f"  Target Index: {target}")
    print(f"  Target features: {real_cols[target]}")
    
    # 3. Анализ ГРАФА (Edges)
    edges = data['edge_indices']
    edge_attr = data['edge_attr']
    print(f"\n4. Graph Structure")
    print(f"  Num Edges: {edges.shape[1]}")
    print(f"  Edge Weights (mean): {edge_attr.mean().item():.4f}")
    
    # Проверка связности: есть ли изолированные переменные?
    # У переменных индексы от 0 до N_vars-1 (в исходном виде)
    unique_vars_in_edges = torch.unique(edges[1]) # В edge_indices[1] лежат переменные
    print(f"  Variables connected: {len(unique_vars_in_edges)} / {cols.shape[0]}")
    
    if len(unique_vars_in_edges) < cols.shape[0]:
        print("  WARNING: Some variables are disconnected from constraints!")

if __name__ == "__main__":
    analyze_sample()