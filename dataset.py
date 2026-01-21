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
        N = int(round(num_vars**(1/3)))

        # 1. Позиционное кодирование (One-Hot)
        indices = torch.arange(num_vars)
        k_idx = indices % N
        j_idx = (indices // N) % N
        i_idx = indices // (N * N)
        
        # Превращаем в One-Hot: [1000, 10] для каждого измерения
        i_oh = F.one_hot(i_idx, num_classes=N).float()
        j_oh = F.one_hot(j_idx, num_classes=N).float()
        k_oh = F.one_hot(k_idx, num_classes=N).float()
        pos_encoding = torch.cat([i_oh, j_oh, k_oh], dim=1) # [1000, 30]

        # 2. Создаем HeteroData
        data = HeteroData()
        
        # Узлы переменных: SCIP фичи + One-Hot адрес
        data['variable'].x = torch.cat([d['col_features'], pos_encoding], dim=1)
        # Узлы ограничений: SCIP фичи
        data['constraint'].x = d['row_features']

        # Ребра (Bipartite)
        # Направление: Переменная -> Ограничение
        v2c_edge_index = torch.stack([d['edge_indices'][1], d['edge_indices'][0]], dim=0)
        # Направление: Ограничение -> Переменная
        c2v_edge_index = torch.stack([d['edge_indices'][0], d['edge_indices'][1]], dim=0)

        data['variable', 'to', 'constraint'].edge_index = v2c_edge_index
        data['constraint', 'to', 'variable'].edge_index = c2v_edge_index
        
        # Атрибуты ребер (коэффициенты матрицы A)
        data['variable', 'to', 'constraint'].edge_attr = d['edge_attr']
        data['constraint', 'to', 'variable'].edge_attr = d['edge_attr']

        # Target и Маска (только для переменных)
        data['variable'].y = d['label_var_idx'].view(1)
        cand_mask = torch.zeros(num_vars, dtype=torch.bool)
        cand_mask[d['candidates']] = True
        data['variable'].cand_mask = cand_mask

        return data