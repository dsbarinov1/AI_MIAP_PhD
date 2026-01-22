import os
import torch
from torch_geometric.data import Dataset, HeteroData

class MIAPDataset(Dataset):
    def __init__(self, root, force_edge_one=True):
        super().__init__(root, None, None)
        self.root = root
        self.files = sorted([f for f in os.listdir(root) if f.endswith('.pt')])
        self.force_edge_one = force_edge_one

    def len(self):
        return len(self.files)

    def get(self, idx):
        d = torch.load(os.path.join(self.root, self.files[idx]))
        
        data = HeteroData()
        
        # Nodes
        # Gasse et al: Features are typically (Objective Coeff, Bounds, etc.)
        # Here we take what Ecole gives us.
        data['constraint'].x = d['row_features']
        data['variable'].x = d['col_features']
        
        # Edges
        # C -> V
        data['constraint', 'adj', 'variable'].edge_index = d['edge_indices']
        
        # V -> C
        data['variable', 'adj', 'constraint'].edge_index = d['edge_indices'].flip(0)
        
        if self.force_edge_one:
            # Overwrite with 1.0 (Assuming unweighted constraints for Assignment Problem)
            # This is robust against Ecole scaling artifacts
            num_edges = d['edge_indices'].shape[1]
            ones = torch.ones(num_edges, 1)
            data['constraint', 'adj', 'variable'].edge_attr = ones
            data['variable', 'adj', 'constraint'].edge_attr = ones
        else:
            data['constraint', 'adj', 'variable'].edge_attr = d['edge_attr']
            data['variable', 'adj', 'constraint'].edge_attr = d['edge_attr']

        # Meta
        data['variable'].y = d['label_var_idx'].unsqueeze(0)
        num_vars = d['col_features'].shape[0]
        cand_mask = torch.zeros(num_vars, dtype=torch.bool)
        cand_mask[d['candidates']] = True
        data['variable'].cand_mask = cand_mask
        
        return data
