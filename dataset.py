import os
import torch
from torch_geometric.data import Dataset, HeteroData
from tqdm import tqdm

class MIAPDataset(Dataset):
    def __init__(self, root, force_edge_one=True, normalize=True, simplify_features=False):
        super().__init__(root, None, None)
        self.root = root
        self.files = sorted([f for f in os.listdir(root) if f.endswith('.pt')])
        self.force_edge_one = force_edge_one
        self.normalize = normalize
        self.simplify_features = simplify_features

        # Load all into memory
        self.data_list = []
        if len(self.files) > 0:
            print(f"Loading {root} into memory...")
            for f in tqdm(self.files):
                self.data_list.append(self._process_file(f))
        else:
            print(f"Warning: {root} is empty.")

    def _process_file(self, filename):
        d = torch.load(os.path.join(self.root, filename))
        
        data = HeteroData()
        
        row_x = d['row_features']
        col_x = d['col_features']
        
        if self.simplify_features:
            indices = [0, 8, 16]
            col_x = col_x[:, indices]
        
        if self.normalize:
            # Cost
            col_x[:, 0] = col_x[:, 0] * 10.0

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

        # Soft Scores
        if 'scores' in d:
            # Normalize scores? No, we do it in train loop (Softmax)
            # But we might want to replace -inf with something valid if needed,
            # though masking handles it.
            data['variable'].scores = d['scores']

        num_vars = d['col_features'].shape[0]
        cand_mask = torch.zeros(num_vars, dtype=torch.bool)
        cand_mask[d['candidates']] = True
        data['variable'].cand_mask = cand_mask
        
        return data

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
