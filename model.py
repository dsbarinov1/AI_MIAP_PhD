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
    def __init__(self, hidden_dim, aggr='add', dropout=0.1):
        super().__init__()

        self.aggr = aggr

        # Determine message dimension
        # If 'cat', we simulate mean || max, so message dim is 2 * hidden
        # Otherwise it's 1 * hidden
        if aggr == 'cat':
            self.conv_mean = WeightedConv(aggr='mean')
            self.conv_max = WeightedConv(aggr='max')
            msg_dim = 2 * hidden_dim
        else:
            self.conv = WeightedConv(aggr=aggr)
            msg_dim = hidden_dim

        # MLP Input: Self (H) + Msg (msg_dim)
        input_dim = hidden_dim + msg_dim

        self.mlp_c = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mlp_v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm_c = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)

    def forward(self, x_c, x_v, edge_index_cv, edge_weight_cv, edge_index_vc, edge_weight_vc):
        if self.aggr == 'cat':
            # Run mean
            msg_to_c_mean = self.conv_mean(x_v, edge_index_vc, edge_weight_vc, size=(x_v.size(0), x_c.size(0)))
            msg_to_v_mean = self.conv_mean(x_c, edge_index_cv, edge_weight_cv, size=(x_c.size(0), x_v.size(0)))

            # Run max
            msg_to_c_max = self.conv_max(x_v, edge_index_vc, edge_weight_vc, size=(x_v.size(0), x_c.size(0)))
            msg_to_v_max = self.conv_max(x_c, edge_index_cv, edge_weight_cv, size=(x_c.size(0), x_v.size(0)))

            # Concat
            msg_to_c = torch.cat([msg_to_c_mean, msg_to_c_max], dim=1)
            msg_to_v = torch.cat([msg_to_v_mean, msg_to_v_max], dim=1)
        else:
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
    def __init__(self, dim_cons, dim_vars, hidden_dim=64, num_layers=2, aggr='add', dropout=0.1):
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
            BipartiteGCNLayer(hidden_dim, aggr=aggr, dropout=dropout) for _ in range(num_layers)
        ])
        
        # Policy
        # Input: Init (H) + Layers * (H) + Raw (D)
        input_dim = hidden_dim * (num_layers + 1) + dim_vars
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
        
        # Jumping Knowledge: Collect embeddings from all layers
        h_v_all = [h_v]

        for layer in self.layers:
            h_c, h_v = layer(h_c, h_v, edge_index_cv, edge_weight_cv, edge_index_vc, edge_weight_vc)
            h_v_all.append(h_v)
            
        # Concatenate all layer outputs + Raw Features
        # h_v_all contains [Init, L1, L2, ...]
        h_v_final = torch.cat(h_v_all + [x_v], dim=1)

        logits = self.policy(h_v_final).squeeze(-1)
        
        return logits
