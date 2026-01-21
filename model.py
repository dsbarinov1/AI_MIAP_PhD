import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import scatter

class GasseGCN(nn.Module):
    def __init__(self, dim_cons, dim_vars, hidden_dim=128):
        super().__init__()
        
        # 1. Раздельные энкодеры (как у Gasse)
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
        
        # 2. Графовые слои (SAGEConv лучше работает с Bipartite, чем GCN)
        # SAGE умеет агрегировать (mean/max) и конкатенировать с собой
        self.conv1 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        
        # 3. Голова
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, data):
        # data.x содержит [Features | Type_Flag]
        # Type_Flag=0 -> Constraint, Type_Flag=1 -> Variable
        
        x = data.x
        
        # Разделяем по типу
        # Последняя колонка - это тип
        node_type = x[:, -1]
        features = x[:, :-1]
        
        # Маски
        mask_cons = (node_type == 0)
        mask_vars = (node_type == 1)
        
        # Нам нужно знать исходную размерность фичей, чтобы отрезать паддинг
        # (Мы передали их в конструктор, но паддинг уже сделан в dataset)
        # Проще: просто подаем как есть, Linear слой сам разберется с нулями,
        # так как мы используем разные Linear для разных типов.
        
        # Но стоп: features имеет размерность max(dim_c, dim_v). 
        # cons_embedding ждет dim_cons.
        # Поэтому нам нужно обрезать лишние нули.
        
        # Определяем размерности из весов первого слоя
        dc = self.cons_embedding[0].in_features
        dv = self.vars_embedding[0].in_features
        
        # Создаем тензор для скрытых состояний
        h = torch.zeros(x.size(0), 128, device=x.device) # 128 = hidden_dim
        
        # Прогоняем ограничения
        if mask_cons.any():
            # Берем только первые dc колонок
            cons_input = features[mask_cons, :dc]
            h[mask_cons] = self.cons_embedding(cons_input)
            
        # Прогоняем переменные
        if mask_vars.any():
            # Берем только первые dv колонок
            vars_input = features[mask_vars, :dv]
            h[mask_vars] = self.vars_embedding(vars_input)
            
        # Теперь h - это качественные эмбеддинги. Пускаем их в граф.
        edge_index = data.edge_index
        
        h = self.conv1(h, edge_index)
        h = torch.relu(h)
        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        h = self.conv3(h, edge_index) # SAGEConv сам добавит нелинейность если надо, но лучше явно
        
        # Предсказание (только для переменных)
        logits = self.policy(h).squeeze(-1)
        
        # Для ограничений (mask_cons) логиты не имеют смысла, но мы их отфильтруем маской кандидатов
        return logits