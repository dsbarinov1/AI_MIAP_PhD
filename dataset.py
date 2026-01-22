import os
import torch
from torch_geometric.data import Dataset, HeteroData

class MIAPDataset(Dataset):
    def __init__(self, root):
        super().__init__(root, None, None)
        self.root = root
        self.files = sorted([f for f in os.listdir(root) if f.endswith('.pt')])

    def len(self):
        return len(self.files)

    def get(self, idx):
        # Load dictionary
        d = torch.load(os.path.join(self.root, self.files[idx]))
        
        data = HeteroData()
        
        # 1. Nodes and Features
        # Constraints
        data['constraint'].x = d['row_features']  # [Nc, Dc]
        data['constraint'].num_nodes = d['row_features'].shape[0]
        
        # Variables
        data['variable'].x = d['col_features']    # [Nv, Dv]
        data['variable'].num_nodes = d['col_features'].shape[0]
        
        # 2. Edges
        # Input edge_indices is [2, E], Row 0 -> Cons, Row 1 -> Var
        # We define edges as bidirectional for message passing
        
        # Constraint -> Variable (if needed, though usually we strictly define flow in model)
        # But GCN usually implies undirected.
        # Gasse et al: C update uses V neighbors, V update uses C neighbors.
        # We need both directions.
        
        # Direction 1: Constraint -> Variable (or Variable -> Constraint)
        # Usually edge_index describes "Source -> Target".
        # d['edge_indices'][0] is C, [1] is V.
        # So this is C connected to V.
        
        # Let's define:
        # (constraint, adj, variable): edges from C to V
        data['constraint', 'adj', 'variable'].edge_index = d['edge_indices']
        data['constraint', 'adj', 'variable'].edge_attr = d['edge_attr']
        
        # (variable, adj, constraint): edges from V to C
        data['variable', 'adj', 'constraint'].edge_index = d['edge_indices'].flip(0)
        data['variable', 'adj', 'constraint'].edge_attr = d['edge_attr']
        
        # 3. Target and Mask
        # Target index is relative to Variables
        # We store it as a property of the variable set (but it's one scalar per graph)
        # To make it batch-friendly, we keep it as a 1-element tensor
        data['variable'].y = d['label_var_idx'].unsqueeze(0) # [1]
        
        # Candidate Mask on Variables
        num_vars = d['col_features'].shape[0]
        cand_mask = torch.zeros(num_vars, dtype=torch.bool)
        cand_mask[d['candidates']] = True
        data['variable'].cand_mask = cand_mask
        
        return data
