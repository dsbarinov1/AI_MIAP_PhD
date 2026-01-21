import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

class GasseHeteroGCN(nn.Module):
    def __init__(self, dim_cons, dim_vars, hidden_dim=128):
        super().__init__()
        
        # Входные проекции
        self.var_embed = nn.Sequential(nn.Linear(dim_vars, hidden_dim), nn.ReLU())
        self.con_embed = nn.Sequential(nn.Linear(dim_cons, hidden_dim), nn.ReLU())

        # Итеративный Message Passing (2 слоя)
        # Слой 1: Обновляем констрейнты на основе переменных, затем переменные на основе констрейнтов
        self.convs = nn.ModuleList()
        for _ in range(2):
            conv = HeteroConv({
                ('variable', 'to', 'constraint'): SAGEConv(hidden_dim, hidden_dim),
                ('constraint', 'to', 'variable'): SAGEConv(hidden_dim, hidden_dim),
            }, aggr='sum')
            self.convs.append(conv)

        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data_dict):
        x_dict = {
            'variable': self.var_embed(data_dict['variable'].x),
            'constraint': self.con_embed(data_dict['constraint'].x)
        }
        edge_index_dict = data_dict.edge_index_dict

        for conv in self.convs:
            # Обновляем признаки всех типов узлов
            h_dict = conv(x_dict, edge_index_dict)
            # Добавляем Residual и ReLU
            x_dict = {key: torch.relu(x_dict[key] + h_dict[key]) for key in x_dict}

        # Предсказываем только для переменных
        return self.policy(x_dict['variable']).squeeze(-1)