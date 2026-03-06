import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GasseGCN(nn.Module):
    def __init__(self, dim_cons, dim_vars, hidden_dim=128, use_sum_aggr=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        aggr = "add" if use_sum_aggr else "mean"

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

        # 2. Графовые слои: sum aggregation (Gasse) + prenorm (LayerNorm after aggregation)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
        self.prenorm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
        self.prenorm2 = nn.LayerNorm(hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr=aggr)
        self.prenorm3 = nn.LayerNorm(hidden_dim)

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
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device, dtype=features.dtype)
        
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
            
        # Теперь h - это качественные эмбеддинги. Пускаем их в граф (sum agg + prenorm как у Gasse).
        edge_index = data.edge_index

        h = self.conv1(h, edge_index)
        h = self.prenorm1(h)
        h = torch.relu(h)
        h = self.conv2(h, edge_index)
        h = self.prenorm2(h)
        h = torch.relu(h)
        h = self.conv3(h, edge_index)
        h = self.prenorm3(h)
        h = torch.relu(h)

        # Предсказание (только для переменных)
        logits = self.policy(h).squeeze(-1)
        
        # Для ограничений (mask_cons) логиты не имеют смысла, но мы их отфильтруем маской кандидатов
        return logits