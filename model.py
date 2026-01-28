import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

class GasseHeteroGCN(nn.Module):
    def __init__(self, dim_cons, dim_vars, hidden_dim=128):
        super().__init__()
        
        # Входные проекции с Dropout
        self.var_embed = nn.Sequential(
            nn.Linear(dim_vars, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.con_embed = nn.Sequential(
            nn.Linear(dim_cons, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 3 слоя итеративного обмена сообщениями
        self.convs = nn.ModuleList()
        for _ in range(3):
            conv = HeteroConv({
                ('variable', 'to', 'constraint'): SAGEConv(hidden_dim, hidden_dim),
                ('constraint', 'to', 'variable'): SAGEConv(hidden_dim, hidden_dim),
            }, aggr='sum')
            self.convs.append(conv)
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])

        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3), # Усиливаем регуляризацию перед выходом
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x_dict = {
            'variable': self.var_embed(data['variable'].x),
            'constraint': self.con_embed(data['constraint'].x)
        }
        
        for i, conv in enumerate(self.convs):
            h_dict = conv(x_dict, data.edge_index_dict)
            # Residual + Norm + ReLU
            x_dict = {
                key: torch.relu(self.layer_norms[i](x_dict[key] + h_dict[key]))
                for key in x_dict
            }

        return self.policy(x_dict['variable']).squeeze(-1)