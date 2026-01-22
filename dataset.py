import os
import torch
from torch_geometric.data import Dataset, HeteroData

class MIAPDataset(Dataset):
    def __init__(self, root, force_edge_one=True, normalize=True, simplify_features=False):
        super().__init__(root, None, None)
        self.root = root
        self.files = sorted([f for f in os.listdir(root) if f.endswith('.pt')])
        self.force_edge_one = force_edge_one
        self.normalize = normalize
        self.simplify_features = simplify_features

    def len(self):
        return len(self.files)

    def get(self, idx):
        d = torch.load(os.path.join(self.root, self.files[idx]))
        
        data = HeteroData()
        
        row_x = d['row_features']
        col_x = d['col_features']
        
        if self.simplify_features:
            indices = [0, 8, 16]
            col_x = col_x[:, indices]
        
        if self.normalize:
            # Normalize Cost (Column 0) - Scale up
            col_x[:, 0] = col_x[:, 0] * 10.0

            # Normalize Structural features?
            # Usually Log(Degree) is good.
            # But let's stick to minimal changes that worked.

        data['constraint'].x = row_x
        data['variable'].x = col_x

        data['constraint', 'adj', 'variable'].edge_index = d['edge_indices']
        data['variable', 'adj', 'constraint'].edge_index = d['edge_indices'].flip(0)
        
        if self.force_edge_one:
            num_edges = d['edge_indices'].shape[1]
            ones = torch.ones(num_edges, 1)
            data['constraint', 'adj', 'variable'].edge_attr = ones
            data['variable', 'adj', 'constraint'].edge_attr = ones
        else:
            data['constraint', 'adj', 'variable'].edge_attr = d['edge_attr']
            data['variable', 'adj', 'constraint'].edge_attr = d['edge_attr']

        data['variable'].y = d['label_var_idx'].unsqueeze(0)
        num_vars = d['col_features'].shape[0]
        cand_mask = torch.zeros(num_vars, dtype=torch.bool)
        cand_mask[d['candidates']] = True
        data['variable'].cand_mask = cand_mask
        
        return data
