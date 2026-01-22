import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class WeightedConv(MessagePassing):
    def __init__(self, aggr='add'):
        super().__init__(aggr=aggr)

    def forward(self, x_source, edge_index, edge_weight, size=None):
        return self.propagate(edge_index, x=x_source, edge_weight=edge_weight, size=size)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight

class BipartiteGCNLayer(nn.Module):
    def __init__(self, hidden_dim, aggr='add'):
        super().__init__()

        # If we use concatenation of aggregators (e.g. mean|max), hidden_dim would change.
        # But for 'add', 'mean', 'max', it stays the same.
        self.conv = WeightedConv(aggr=aggr)

        self.mlp_c = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mlp_v = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm_c = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)

    def forward(self, x_c, x_v, edge_index_cv, edge_weight_cv, edge_index_vc, edge_weight_vc):
        msg_to_c = self.conv(x_v, edge_index_vc, edge_weight_vc, size=(x_v.size(0), x_c.size(0)))
        msg_to_v = self.conv(x_c, edge_index_cv, edge_weight_cv, size=(x_c.size(0), x_v.size(0)))

        # Update C
        out_c = torch.cat([x_c, msg_to_c], dim=1)
        x_c_new = self.mlp_c(out_c)
        x_c_new = self.norm_c(x_c_new) + x_c

        # Update V
        out_v = torch.cat([x_v, msg_to_v], dim=1)
        x_v_new = self.mlp_v(out_v)
        x_v_new = self.norm_v(x_v_new) + x_v

        return x_c_new, x_v_new

class GasseGCN(nn.Module):
    def __init__(self, dim_cons, dim_vars, hidden_dim=64, num_layers=2, aggr='add'):
        super().__init__()
        
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
        
        self.layers = nn.ModuleList([
            BipartiteGCNLayer(hidden_dim, aggr=aggr) for _ in range(num_layers)
        ])
        
        # Policy
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim + dim_vars, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, data):
        x_c = data['constraint'].x
        x_v = data['variable'].x
        
        edge_index_cv = data['constraint', 'adj', 'variable'].edge_index
        edge_weight_cv = data['constraint', 'adj', 'variable'].edge_attr
        
        edge_index_vc = data['variable', 'adj', 'constraint'].edge_index
        edge_weight_vc = data['variable', 'adj', 'constraint'].edge_attr
        
        h_c = self.cons_embedding(x_c)
        h_v = self.vars_embedding(x_v)
        
        for layer in self.layers:
            h_c, h_v = layer(h_c, h_v, edge_index_cv, edge_weight_cv, edge_index_vc, edge_weight_vc)
            
        # Skip Connection
        h_v_final = torch.cat([h_v, x_v], dim=1)

        logits = self.policy(h_v_final).squeeze(-1)
        
        return logits
