import os
import torch
from torch_geometric.data import Dataset, Data

class MIAPDataset(Dataset):
    def __init__(self, root):
        super().__init__(root, None, None)
        self.root = root
        self.files = sorted([f for f in os.listdir(root) if f.endswith('.pt')])

    def len(self):
        return len(self.files)

    def get(self, idx):
        d = torch.load(os.path.join(self.root, self.files[idx]))
        
        row_feats = d['row_features'] 
        col_feats = d['col_features'] 
        
        # --- FIX 1: Positional Encodings (Координаты) ---
        # Мы знаем, что n_size^3 = col_feats.shape[0] (если k=3)
        # Вычислим N
        num_vars = col_feats.shape[0]
        N = round(num_vars ** (1/3)) 
        
        if N**3 != num_vars:
            # На всякий случай, если вдруг размерность другая
            # Просто добавим заглушки, чтобы не упало
            pos_feats = torch.zeros(num_vars, 3)
        else:
            # Генерируем координаты i, j, k
            indices = torch.arange(num_vars)
            k_idx = indices % N
            j_idx = (indices // N) % N
            i_idx = indices // (N * N)
            
            # Нормализуем к [0, 1] и стакаем
            pos_feats = torch.stack([i_idx, j_idx, k_idx], dim=1).float() / (N - 1)

        # Добавляем координаты к фичам переменных
        col_feats = torch.cat([col_feats, pos_feats], dim=1)
        # ------------------------------------------------
        
        # --- FIX 2: Random Noise (для надежности) ---
        # Оставим немного шума, чтобы совсем одинаковые ситуации различались
        noise = torch.rand(col_feats.shape[0], 2) 
        col_feats = torch.cat([col_feats, noise], dim=1)
        # -------------------------------------------
        
        # Дальше стандартная сборка графа
        dim_c = row_feats.shape[1]
        dim_v = col_feats.shape[1]
        max_dim = max(dim_c, dim_v)
        
        row_padded = torch.cat([row_feats, torch.zeros(row_feats.shape[0], max_dim - dim_c)], dim=1)
        col_padded = torch.cat([col_feats, torch.zeros(col_feats.shape[0], max_dim - dim_v)], dim=1)
        
        row_type = torch.zeros(row_feats.shape[0], 1)
        col_type = torch.ones(col_feats.shape[0], 1)
        
        x = torch.cat([
            torch.cat([row_padded, row_type], dim=1),
            torch.cat([col_padded, col_type], dim=1)
        ], dim=0)
        
        num_cons = row_feats.shape[0]
        u = d['edge_indices'][0]
        v = d['edge_indices'][1] + num_cons
        
        edge_index = torch.cat([
            torch.stack([u, v], dim=0),
            torch.stack([v, u], dim=0)
        ], dim=1)
        
        # Дублируем атрибуты ребер
        edge_attr = torch.cat([d['edge_attr'], d['edge_attr']], dim=0)
        
        y = d['label_var_idx'] + num_cons
        
        cand_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        cand_indices = d['candidates'] + num_cons
        cand_mask[cand_indices] = True
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y.unsqueeze(0), cand_mask=cand_mask)