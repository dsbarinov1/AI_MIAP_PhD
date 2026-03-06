import os
import torch
from torch_geometric.data import Dataset, Data


def observation_to_data(obs, action_set):
    """
    Convert Ecole NodeBipartite observation + action_set to a single PyG Data
    in the same format as MIAPDataset.get(), so the GNN model receives identical input.
    """
    row_feats = torch.tensor(obs.row_features, dtype=torch.float32)
    col_feats = torch.tensor(obs.variable_features, dtype=torch.float32)
    dim_c, dim_v = row_feats.shape[1], col_feats.shape[1]
    max_dim = max(dim_c, dim_v)

    row_padded = torch.cat([row_feats, torch.zeros(row_feats.shape[0], max_dim - dim_c)], dim=1)
    col_padded = torch.cat([col_feats, torch.zeros(col_feats.shape[0], max_dim - dim_v)], dim=1)
    row_type = torch.zeros(row_feats.shape[0], 1)
    col_type = torch.ones(col_feats.shape[0], 1)

    x = torch.cat([
        torch.cat([row_padded, row_type], dim=1),
        torch.cat([col_padded, col_type], dim=1),
    ], dim=0)

    num_cons = row_feats.shape[0]
    edge_indices = torch.tensor(obs.edge_features.indices, dtype=torch.long)
    u, v = edge_indices[0], edge_indices[1] + num_cons
    edge_index_fwd = torch.stack([u, v], dim=0)
    edge_index_bwd = torch.stack([v, u], dim=0)
    edge_index = torch.cat([edge_index_fwd, edge_index_bwd], dim=1)
    edge_attr_raw = torch.tensor(obs.edge_features.values, dtype=torch.float32).unsqueeze(1)
    edge_attr = torch.cat([edge_attr_raw, edge_attr_raw], dim=0)

    action_set_t = torch.tensor(action_set, dtype=torch.long)
    cand_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    cand_mask[action_set_t + num_cons] = True

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, cand_mask=cand_mask, num_cons=num_cons)


class MIAPDataset(Dataset):
    def __init__(self, root):
        super().__init__(root, None, None)
        self.root = root
        self.files = sorted([f for f in os.listdir(root) if f.endswith('.pt')])

    def len(self):
        return len(self.files)

    def get(self, idx):
        # Загружаем словарь
        d = torch.load(os.path.join(self.root, self.files[idx]))
        
        # 1. Создаем узлы (Constraints + Variables)
        # У нас двудольный граф. В PyG проще всего сделать его однородным,
        # объединив признаки и добавив тип узла.
        
        row_feats = d['row_features'] # [Nc, Dc]
        col_feats = d['col_features'] # [Nv, Dv]
        
        # Паддинг признаков (чтобы объединить их в одну матрицу)
        dim_c = row_feats.shape[1]
        dim_v = col_feats.shape[1]
        max_dim = max(dim_c, dim_v)
        
        # Дополняем нулями
        row_padded = torch.cat([row_feats, torch.zeros(row_feats.shape[0], max_dim - dim_c)], dim=1)
        col_padded = torch.cat([col_feats, torch.zeros(col_feats.shape[0], max_dim - dim_v)], dim=1)
        
        # Добавляем признак типа узла (1 = Variable, 0 = Constraint)
        # Это важно, чтобы сеть различала их
        row_type = torch.zeros(row_feats.shape[0], 1)
        col_type = torch.ones(col_feats.shape[0], 1)
        
        x = torch.cat([
            torch.cat([row_padded, row_type], dim=1), # Сначала ограничения
            torch.cat([col_padded, col_type], dim=1)  # Потом переменные
        ], dim=0)
        
        # 2. Ребра
        # В исходных данных: [0] -> constraint, [1] -> variable
        # Нам нужно сдвинуть индексы переменных на num_constraints
        num_cons = row_feats.shape[0]
        
        u = d['edge_indices'][0]            # Констрейнты (0..M-1)
        v = d['edge_indices'][1] + num_cons # Переменные (M..M+N-1)
        
        # Делаем граф неориентированным (GCN требует этого для прохода в обе стороны)
        # C -> V
        edge_index_fwd = torch.stack([u, v], dim=0)
        # V -> C
        edge_index_bwd = torch.stack([v, u], dim=0)
        
        edge_index = torch.cat([edge_index_fwd, edge_index_bwd], dim=1)
        edge_attr = torch.cat([d['edge_attr'], d['edge_attr']], dim=0)
        
        # 3. Target и Маска
        # Target тоже сдвигаем, так как переменные теперь начинаются с num_cons
        y = d['label_var_idx'] + num_cons
        
        # Маска кандидатов (тоже сдвигаем)
        cand_mask = torch.zeros(x.shape[0], dtype=torch.bool)
        cand_indices = d['candidates'] + num_cons
        cand_mask[cand_indices] = True
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y.unsqueeze(0), cand_mask=cand_mask)