import os
import argparse
import torch
import ecole
import numpy as np
import pyscipopt
import pandas as pd
import time
from torch_geometric.data import HeteroData
from tqdm import tqdm

# Импорты проекта
from generators import MIAPGenerator
from model import GasseGCN

# ==========================================
# 1. Класс Политики (GNN Inference)
# ==========================================
class GNNBranchingPolicy:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        state_dict = torch.load(model_path, map_location=device)
        
        try:
            dim_v = state_dict['vars_embedding.0.weight'].shape[1]
            dim_c = state_dict['cons_embedding.0.weight'].shape[1]
            hidden_dim = state_dict['vars_embedding.0.weight'].shape[0]
        except KeyError:
            # Фолбэк на старые ключи, если вдруг модель старая
            print("Warning: Keys 'vars_embedding' not found, trying 'var_embed'...")
            dim_v = state_dict['var_embed.0.weight'].shape[1]
            dim_c = state_dict['con_embed.0.weight'].shape[1]
            hidden_dim = state_dict['var_embed.0.weight'].shape[0]
            
        num_layers = 0
        while True:
            if f'layers.{num_layers}.mlp_c.0.weight' in state_dict or \
               f'layers.{num_layers}.conv.lin_l.weight' in state_dict:
                num_layers += 1
            else:
                break
        if num_layers == 0: num_layers = 3 # Наш стандарт

        self.model = GasseGCN(
            dim_cons=dim_c, 
            dim_vars=dim_v, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            aggr='max', # Предполагаем max, так как он давал лучшие результаты
            activation='relu',
            dropout=0.0 # Для инференса дропаут не нужен
        ).to(device)
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.input_dim_v = dim_v

    def __call__(self, candidate_set, observation):
        row_feats = torch.tensor(observation.row_features, dtype=torch.float32)
        col_feats = torch.tensor(observation.variable_features, dtype=torch.float32)
        edge_indices = torch.tensor(observation.edge_features.indices, dtype=torch.long)
        edge_attr = torch.tensor(observation.edge_features.values, dtype=torch.float32).unsqueeze(1)
        
        # Обработка признаков (как в dataset.py)
        # Если модель ожидает меньше фичей (simplify_features=True), режем вход
        if col_feats.shape[1] > self.input_dim_v:
             # Логика из dataset.py: indices = [0, 8, 16]
             # (Cost, SolVal, IsBasis?) - проверяем, совпадает ли размер
             indices = [0, 8, 16]
             if self.input_dim_v == len(indices):
                 col_feats = col_feats[:, indices]
             else:
                 # Если размер не совпадает ни с полным, ни с упрощенным - это проблема
                 # Попробуем просто отрезать лишнее (иногда работает для Random Features)
                 col_feats = col_feats[:, :self.input_dim_v]
        
        # Нормализация (как в dataset.py: col_x[:, 0] = col_x[:, 0] * 10.0)
        # Применяем только к 0-й колонке (Objective)
        col_feats[:, 0] = col_feats[:, 0] * 10.0

        data = HeteroData()
        data['constraint'].x = row_feats
        data['variable'].x = col_feats
        data['constraint', 'adj', 'variable'].edge_index = edge_indices
        data['constraint', 'adj', 'variable'].edge_attr = edge_attr
        data['variable', 'adj', 'constraint'].edge_index = edge_indices.flip(0)
        data['variable', 'adj', 'constraint'].edge_attr = edge_attr
        
        data = data.to(self.device)
        with torch.no_grad():
            logits = self.model(data)
            
        candidates = torch.tensor(candidate_set.astype(np.int64), device=self.device, dtype=torch.long)
        cand_logits = logits[candidates]
        return candidates[torch.argmax(cand_logits)].item()

# ==========================================
# 2. Основная функция эксперимента
# ==========================================
def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--n", type=int, default=10, help="Problem size (N)")
    parser.add_argument("--k", type=int, default=3, help="Index size (K)")
    parser.add_argument("--samples", type=int, default=10, help="Number of instances")
    parser.add_argument("--type", type=str, default="random", choices=["random", "euclidean"])
    parser.add_argument("--time_limit", type=int, default=100)
    args = parser.parse_args()

    gen = MIAPGenerator(n=args.n, k=args.k)
    policy = GNNBranchingPolicy(args.model)
    
    # Параметры солвера (общие для обоих режимов)
    scip_params = {
        "presolving/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "separating/maxrounds": 0,
        "separating/maxroundsroot": 0,
        "propagating/maxrounds": 0,
        "propagating/maxroundsroot": 0,
        "limits/time": args.time_limit,
    }

    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        information_function={
            "nb_nodes": ecole.reward.NNodes(),
            "time": ecole.reward.SolvingTime(),
            "scores": ecole.observation.StrongBranchingScores(pseudo_candidates=False)
        },
        scip_params=scip_params
    )

    results = []
    print(f"\n>>> Starting Benchmark: N={args.n}, K={args.k}, Type={args.type}, Num Instances={args.samples} <<<")

    for i in tqdm(range(args.samples)):
        # Генерируем задачу
        cost = gen.generate_random_uniform() if args.type == "random" else gen.generate_euclidean()
        model_scip, _ = gen.build_scip_model(cost)
        temp_file = f"temp_eval_{os.getpid()}.mps"
        model_scip.writeProblem(temp_file)

        # 1. SCIP Baseline
        m_def = pyscipopt.Model()
        m_def.readProblem(temp_file)
        for param, val in scip_params.items():
            m_def.setParam(param, val)
        m_def.hideOutput(True)
        
        t0 = time.time()
        m_def.optimize()
        scip_time = time.time() - t0
        scip_nodes = m_def.getNNodes()
        scip_obj = m_def.getObjVal()

        # 2. GNN Controlled Branching
        obs, action_set, _, done, info = env.reset(temp_file)
        
        gnn_time_start = time.time()
        matches = 0
        decisions = 0
        
        if done:
            gnn_nodes = 1
        else:
            while not done:
                # Находим экспертное решение для диагностики (Alignment)
                sb_scores = np.nan_to_num(info["scores"], nan=-1e9)
                expert_action = np.argmax(sb_scores)
                
                # Находим решение нейросети
                gnn_action = policy(action_set, obs)
                
                if gnn_action == expert_action:
                    matches += 1
                decisions += 1
                
                obs, action_set, _, done, info = env.step(gnn_action)
            
            gnn_nodes = info["nb_nodes"]
        
        gnn_time = time.time() - gnn_time_start
        # Вытаскиваем итоговый результат GNN-сессии
        gnn_obj = env.model.as_pyscipopt().getObjVal()
        
        # Расчет Gap (отклонение от базового SCIP)
        # Если Gap > 0, GNN нашла решение хуже. Если 0 - одинаковое.
        primal_gap = (gnn_obj - scip_obj) / abs(scip_obj) * 100 if abs(scip_obj) > 1e-7 else 0
        alignment = (matches / decisions * 100) if decisions > 0 else 100.0
        
        print(f"Instance {i+1}:")
        print(f"  SCIP: {scip_time:.2f}s, {scip_nodes} nodes")
        print(f"  GNN:  {gnn_time:.2f}s, {gnn_nodes} nodes")
        print(f"  Scores:  {scores_total}")
        
        scores_total = info["scores"]

        results.append({
            "id": i,
            "scip_nodes": scip_nodes,
            "gnn_nodes": gnn_nodes,
            "gap_pct": primal_gap,
            "alignment": alignment,
            "scip_time": scip_time,
            "gnn_time": gnn_time,
            "scores": scores_total
        })

        if os.path.exists(temp_file): os.remove(temp_file)

    # Статистика
    df = pd.DataFrame(results)
    print("\n" + "="*40)
    print(f"AVERAGE RESULTS (N={args.n}, {args.type})")
    print("="*40)
    print(f"Node Reduction: {df['scip_nodes'].mean():.1f} -> {df['gnn_nodes'].mean():.1f} " 
          f"({(1 - df['gnn_nodes'].mean()/df['scip_nodes'].mean())*100:.1f}% reduction)")
    print(f"Average Primal Gap: {df['gap_pct'].mean():.4f}%")
    print(f"Decision Alignment: {df['alignment'].mean():.1f}%")
    print(f"Time: SCIP {df['scip_time'].mean():.2f}s | GNN {df['gnn_time'].mean():.2f}s")
    
    df.to_csv(f"benchmark_N{args.n}_{args.type}.csv", index=False)

if __name__ == "__main__":
    run_benchmark()