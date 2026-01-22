import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class WeightedSumConv(MessagePassing):
    """
    Computes sum_{j in N(i)} (e_{ij} * x_j)
    """
    def __init__(self):
        super().__init__(aggr='add') # Sum aggregation

    def forward(self, x_source, edge_index, edge_weight, size=None):
        # x_source: [N_src, D]
        # edge_index: [2, E]
        # edge_weight: [E, 1]
        # size: (N_src, N_dst) tuple for bipartite
        return self.propagate(edge_index, x=x_source, edge_weight=edge_weight, size=size)

    def message(self, x_j, edge_weight):
        # x_j: [E, D] - source node features for each edge
        # edge_weight: [E, 1]
        return x_j * edge_weight

class BipartiteGCNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        # --- Constraint Update Params ---
        # W_C * h_c
        self.lin_c_self = nn.Linear(hidden_dim, hidden_dim)
        # W_CV * sum(e * h_v)
        self.lin_c_msg = nn.Linear(hidden_dim, hidden_dim)

        # --- Variable Update Params ---
        # W_V * h_v
        self.lin_v_self = nn.Linear(hidden_dim, hidden_dim)
        # W_VC * sum(e * h_c)
        self.lin_v_msg = nn.Linear(hidden_dim, hidden_dim)

        self.conv = WeightedSumConv()
        self.norm_c = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)

    def forward(self, x_c, x_v, edge_index_cv, edge_weight_cv, edge_index_vc, edge_weight_vc):
        """
        x_c: [Nc, H]
        x_v: [Nv, H]
        edge_index_cv: Edges from C to V (Source=C, Target=V). Used for updating V.
        edge_index_vc: Edges from V to C (Source=V, Target=C). Used for updating C.
        """

        # 1. Aggregation (Messages)
        # Msg to C comes from V (via edge_index_vc)
        # Size: (Source_V, Target_C)
        msg_to_c = self.conv(x_v, edge_index_vc, edge_weight_vc, size=(x_v.size(0), x_c.size(0)))

        # Msg to V comes from C (via edge_index_cv)
        # Size: (Source_C, Target_V)
        msg_to_v = self.conv(x_c, edge_index_cv, edge_weight_cv, size=(x_c.size(0), x_v.size(0)))

        # 2. Update Steps
        # h_c' = ReLU( Lin_self(h_c) + Lin_msg(msg_to_c) )
        x_c_new = self.lin_c_self(x_c) + self.lin_c_msg(msg_to_c)
        x_c_new = self.norm_c(torch.relu(x_c_new)) # LayerNorm usually helps

        # h_v' = ReLU( Lin_self(h_v) + Lin_msg(msg_to_v) )
        x_v_new = self.lin_v_self(x_v) + self.lin_v_msg(msg_to_v)
        x_v_new = self.norm_v(torch.relu(x_v_new))

        return x_c_new, x_v_new

class GasseGCN(nn.Module):
    def __init__(self, dim_cons, dim_vars, hidden_dim=64, num_layers=2):
        super().__init__()
        
        # 1. Embeddings
        self.cons_embedding = nn.Sequential(
            nn.Linear(dim_cons, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.vars_embedding = nn.Sequential(
            nn.Linear(dim_vars, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. GCN Layers
        self.layers = nn.ModuleList([
            BipartiteGCNLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        # 3. Policy Head
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, data):
        # Extract data from HeteroData batch
        x_c = data['constraint'].x
        x_v = data['variable'].x
        
        # Edges
        # C -> V
        edge_index_cv = data['constraint', 'adj', 'variable'].edge_index
        edge_weight_cv = data['constraint', 'adj', 'variable'].edge_attr
        
        # V -> C
        edge_index_vc = data['variable', 'adj', 'constraint'].edge_index
        edge_weight_vc = data['variable', 'adj', 'constraint'].edge_attr
        
        # Initial Embedding
        h_c = self.cons_embedding(x_c)
        h_v = self.vars_embedding(x_v)
        
        # Message Passing
        for layer in self.layers:
            h_c, h_v = layer(h_c, h_v, edge_index_cv, edge_weight_cv, edge_index_vc, edge_weight_vc)
            
        # Prediction (on Variables only)
        logits = self.policy(h_v).squeeze(-1)
        
        return logits
