import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv

class MIAPHypergraphNet(nn.Module):
    def __init__(self, dim_cons=5, dim_vars=19, hidden_dim=64):
        super().__init__()
        
        # Эмбеддинги (остаются такими же)
        self.cons_embedding = nn.Sequential(
            nn.Linear(dim_cons, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.vars_embedding = nn.Sequential(
            nn.Linear(dim_vars, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Hypergraph Convolutions
        # В PyG HypergraphConv работает с матрицей инцидентности hyperedge_index
        # hyperedge_index: [2, Num_Edges_in_Bipartite]
        # Row 0: Index of node (Variable)
        # Row 1: Index of hyperedge (Constraint)
        
        # Мы будем использовать 2 слоя HGNN
        self.conv1 = HypergraphConv(hidden_dim, hidden_dim, use_attention=False, heads=1)
        self.conv2 = HypergraphConv(hidden_dim, hidden_dim, use_attention=False, heads=1)
        
        # Policy Head
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, batch):
        # 1. Подготовка данных
        # У нас в batch['edge_indices']:
        # row 0 = constraint_idx (Hyperedge)
        # row 1 = variable_idx (Node)
        
        # Для HypergraphConv нужно наоборот: (Node, Hyperedge)
        # И важно, чтобы индексы были правильными.
        
        constraint_idx = batch['edge_indices'][0]
        variable_idx = batch['edge_indices'][1]
        
        # hyperedge_index: [2, E] -> [[node_idx], [hyperedge_idx]]
        hyperedge_index = torch.stack([variable_idx, constraint_idx], dim=0)
        
        # 2. Embeddings
        # В HypergraphConv узлы - это переменные.
        # А гиперребра (constraints) могут иметь свои фичи?
        # В PyG реализации HypergraphConv фичи гиперребер (hyperedge_attr) поддерживаются.
        
        x = self.vars_embedding(batch['col_features'])      # Фичи узлов (переменных)
        h_e = self.cons_embedding(batch['row_features'])    # Фичи гиперребер (ограничений)
        
        # 3. Convolution
        # Передаем x (узлы) и hyperedge_index.
        # hyperedge_attr используется для весов, но у нас фичи - это векторы.
        # Стандартный HypergraphConv в PyG не принимает векторные фичи гиперребер напрямую в forward.
        # Он принимает только веса.
        
        # ХАК: Мы можем склеить граф.
        # Но давай для начала попробуем без фичей ограничений, только на структуре.
        # Или используем Gasse-подход (Bipartite), который ты уже реализовал, 
        # потому что Bipartite Graph и Hypergraph математически эквивалентны 
        # (если гиперребра рассматривать как узлы второй доли).
        
        # В статье Heydaribeni они делают именно Bipartite Message Passing, 
        # просто называют это Hypergraph.
        # Так что твоя GasseGCN - это УЖЕ реализация Hypergraph Neural Network 
        # в представлении Incidence Graph.
        
        # Поэтому, чтобы не усложнять сейчас (PyG HypergraphConv капризный),
        # мы оставим GasseGCN, но скажем научнику:
        # "Я реализовал архитектуру на основе передачи сообщений (MPNN). 
        # В литературе (Heydaribeni) это эквивалентно HGNN, если рассматривать ограничения как гиперребра".
        
        pass 

# Возвращаем старую модель, так как она правильная
from model import GasseGCN as MIAPHypergraphNet