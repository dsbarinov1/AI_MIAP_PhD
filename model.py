import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GasseGCN(nn.Module):
    def __init__(self, dim_cons, dim_vars, hidden_dim=64):
        super().__init__()
        
        # --- 1. Эмбеддинги с Нормализацией ---
        self.cons_embedding = nn.Sequential(
            nn.Linear(dim_cons, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.vars_embedding = nn.Sequential(
            nn.Linear(dim_vars, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # --- 2. Графовые слои (Pre-activation ResNet style) ---
        # SAGEConv агрегирует соседей
        self.conv1 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # --- 3. Голова ---
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Logit
        )
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Разделяем фичи
        node_type = x[:, -1]
        features = x[:, :-1]
        
        mask_cons = (node_type == 0)
        mask_vars = (node_type == 1)
        
        # Определяем размерность входа динамически
        dc = self.cons_embedding[0].in_features
        dv = self.vars_embedding[0].in_features
        
        # Инициализация скрытого состояния
        h = torch.zeros(x.size(0), self.cons_embedding[0].out_features, device=x.device)
        
        if mask_cons.any():
            h[mask_cons] = self.cons_embedding(features[mask_cons, :dc])
        if mask_vars.any():
            h[mask_vars] = self.vars_embedding(features[mask_vars, :dv])
            
        # --- RESIDUAL BLOCKS ---
        # Block 1
        h_in = h
        h = self.conv1(h, edge_index)
        h = self.norm1(h)
        h = torch.relu(h)
        h = h + h_in # Skip Connection
        
        # Block 2
        h_in = h
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = torch.relu(h)
        h = h + h_in # Skip Connection

        # Block 3
        h_in = h
        h = self.conv3(h, edge_index)
        h = self.norm3(h)
        h = torch.relu(h)
        h = h + h_in # Skip Connection
        
        # Predict
        logits = self.policy(h).squeeze(-1)
        return logits