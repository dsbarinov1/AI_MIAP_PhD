import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, HeteroData

class MIAPDataset(Dataset):
    def __init__(self, root):
        super().__init__(root)
        self.files = sorted([f for f in os.listdir(root) if f.endswith('.pt')])

    def len(self): return len(self.files)

    def get(self, idx):
        d = torch.load(os.path.join(self.root, self.files[idx]))
        num_vars = d['col_features'].shape[0]
        num_cons = d['row_features'].shape[0]
        N = int(round(num_vars**(1/3)))

        # 1. Разрушение симметрии через случайные признаки (Symmetry Breaking)
        # Каждому экземпляру задачи даем уникальный шум, чтобы различать "близнецов"
        # 8 признаков шума - достаточно для 1000 переменных
        var_noise = torch.randn(num_vars, 8) 

        # 2. Типизация ограничений (Constraint Types)
        # В MIAP k=3 измерения. Поймем, какое ограничение к чему относится.
        # Ограничения в генераторе идут по порядку: N для dim0, N для dim1, N для dim2.
        con_types = torch.zeros(num_cons, dtype=torch.long)
        con_types[N:2*N] = 1
        con_types[2*N:] = 2
        con_type_oh = F.one_hot(con_types, num_classes=3).float()

        # 3. Нормализация признаков SCIP
        v_feats = d['col_features']
        r_feats = d['row_features']
        v_feats = (v_feats - v_feats.mean(dim=0)) / (v_feats.std(dim=0) + 1e-6)
        r_feats = (r_feats - r_feats.mean(dim=0)) / (r_feats.std(dim=0) + 1e-6)

        data = HeteroData()
        
        # Переменные: Нормализованные фичи + Шум
        data['variable'].x = torch.cat([v_feats, var_noise], dim=1)
        # Ограничения: Нормализованные фичи + Тип (Работник/Задача/Время)
        data['constraint'].x = torch.cat([r_feats, con_type_oh], dim=1)

        # Ребра
        v2c = torch.stack([d['edge_indices'][1], d['edge_indices'][0]], dim=0)
        c2v = torch.stack([d['edge_indices'][0], d['edge_indices'][1]], dim=0)

        data['variable', 'to', 'constraint'].edge_index = v2c
        data['constraint', 'to', 'variable'].edge_index = c2v
        
        # Цель и маска
        data['variable'].y = d['label_var_idx'].view(1)
        cand_mask = torch.zeros(num_vars, dtype=torch.bool)
        cand_mask[d['candidates']] = True
        data['variable'].cand_mask = cand_mask

        return data