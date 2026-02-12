import os
import torch
import ecole
import numpy as np
import pyscipopt
import pandas as pd
import time
from torch_geometric.data import Data, Batch

# Импорты проекта
from generators import MIAPGenerator
from model import GasseGCN

# ==========================================
# 1. Класс Политики (GNN Inference)
# ==========================================
class GNNBranchingPolicy:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # Загружаем веса
        state_dict = torch.load(model_path, map_location=device)
        
        # 1. Определяем размерности входа из весов
        # model.py: self.vars_embedding = nn.Sequential(...)
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
            
        # 2. Пытаемся угадать количество слоев (для GasseGCN)
        # Ищем ключи вида layers.0.conv..., layers.1.conv...
        num_layers = 0
        while True:
            if f'layers.{num_layers}.mlp_c.0.weight' in state_dict or \
               f'layers.{num_layers}.conv.lin_l.weight' in state_dict: # В зависимости от типа свертки
                num_layers += 1
            else:
                break
        if num_layers == 0: num_layers = 2 # Дефолт, если не нашли
        
        print(f"Loaded model from {model_path}")
        print(f"Detected: Vars={dim_v}, Cons={dim_c}, Hidden={hidden_dim}, Layers={num_layers}")
        
        # 3. Инициализация модели
        # Важно: aggr='max' (или тот, на котором лучшее качество), activation='relu'
        # Если модель сохранялась без метаданных аргументов, предполагаем дефолт или 'max'
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
        """
        Метод вызывается внутри цикла Ecole.
        """
        # --- 1. Препроцессинг (должен совпадать с dataset.py) ---
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

        # --- 2. Сборка Data (Hetero-style simulation via Bipartite keys) ---
        # Модель GasseGCN ожидает data['constraint'].x и т.д.
        # Используем класс Data, но наполняем его атрибутами как HeteroData
        # (PyG позволяет динамические атрибуты)
        
        # Создаем псевдо-HeteroData объект (или просто объект с нужными полями)
        # Так как GasseGCN использует data['constraint'].x, нам нужен объект, поддерживающий доступ через .
        
        class BatchData:
            def __init__(self):
                self.constraint = type('obj', (object,), {'x': None})
                self.variable = type('obj', (object,), {'x': None})
                self.edge_index_dict = {} # Для совместимости если нужно
                self.edge_attr_dict = {}

            def __getitem__(self, key):
                # Эмуляция data['constraint', 'adj', 'variable']
                if isinstance(key, tuple):
                    return self.edge_index_dict.get(key)
                if key == 'constraint': return self.constraint
                if key == 'variable': return self.variable
                return None

        # Но проще использовать реальный HeteroData, если установлен
        from torch_geometric.data import HeteroData
        data = HeteroData()
        
        data['constraint'].x = row_feats
        data['variable'].x = col_feats
        
        # dataset.py: data['constraint', 'adj', 'variable'].edge_index = d['edge_indices']
        data['constraint', 'adj', 'variable'].edge_index = edge_indices
        data['constraint', 'adj', 'variable'].edge_attr = edge_attr
        
        # dataset.py: data['variable', 'adj', 'constraint'].edge_index = d['edge_indices'].flip(0)
        data['variable', 'adj', 'constraint'].edge_index = edge_indices.flip(0)
        data['variable', 'adj', 'constraint'].edge_attr = edge_attr
        
        data = data.to(self.device)
        
        # --- 3. Инференс ---
        with torch.no_grad():
            logits = self.model(data) # [Num_Vars]
            
        # --- 4. Выбор действия ---
        candidates = torch.tensor(candidate_set.astype(np.int64), device=self.device, dtype=torch.long)
        
        # Выбираем оценки только для кандидатов
        cand_logits = logits[candidates]
        best_local_idx = torch.argmax(cand_logits).item()
        
        return candidates[best_local_idx].item()


# ==========================================
# 2. Функция Оценки (Evaluation Loop)
# ==========================================
def run_evaluation(model_path, num_instances=10, n_size=10, time_limit=60):
    
    # observation_function = ecole.observation.NodeBipartite()
    # information_function = {
    #     "scores": ecole.observation.StrongBranchingScores(pseudo_candidates=False)
    # }
    
    # Параметры SCIP
    # Убрали heuristics/emphasis, так как он вызывал ошибку
    
    # Генератор
    gen = MIAPGenerator(n=n_size, k=3)
    
    # Политика GNN
    try:
        gnn_policy = GNNBranchingPolicy(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Среда Ecole
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        information_function={
            "nb_nodes": ecole.reward.NNodes(),
            "time": ecole.reward.SolvingTime(),
            "scores": ecole.observation.StrongBranchingScores(pseudo_candidates=False)
        },
        scip_params = {
            "presolving/maxrounds": 0,           # Главное: отключить пресолвинг
            "presolving/maxrestarts": 0,         # Отключить рестарты
            "separating/maxrounds": 0,           # Отключить cuts
            "separating/maxroundsroot": 0,
            "propagating/maxrounds": 0,
            "propagating/maxroundsroot": 0,
            "limits/time": time_limit,
        }
    )
    
    results = []
    print(f"\n>>> Starting Evaluation on {num_instances} instances (N={n_size}) <<<")
    
    for i in range(num_instances):
        # 1. Генерируем задачу (Euclidean сложнее)
        # if i % 2 == 0:
        #     cost_tensor = gen.generate_random_uniform()
        #     ptype = "random"
        # else:
        #     cost_tensor = gen.generate_euclidean()
        #     ptype = "euclidean"
        # cost_tensor = gen.generate_euclidean()
        cost_tensor = gen.generate_random_uniform()
        model_scip, _ = gen.build_scip_model(cost_tensor)
        
        temp_file = f"eval_temp_{i}.mps"
        model_scip.writeProblem(temp_file)
        
        # === A. SCIP Default (Relpscost) ===
        # Создаем новую модель для чистоты
        m_def = pyscipopt.Model()
        m_def.readProblem(temp_file)
        
        # scip_params = {
        #     "presolving/maxrounds": 0,           # Главное: отключить пресолвинг
        #     "presolving/maxrestarts": 0,         # Отключить рестарты
        #     "separating/maxrounds": 0,           # Отключить cuts
        #     "separating/maxroundsroot": 0,
        #     "propagating/maxrounds": 0,
        #     "propagating/maxroundsroot": 0,
        #     "limits/time": time_limit,
        # }
        
        m_def.setParam("presolving/maxrounds", 0)
        m_def.setParam("presolving/maxrestarts", 0)
        m_def.setParam("separating/maxrounds", 0)
        m_def.setParam("separating/maxroundsroot", 0)
        m_def.setParam("propagating/maxrounds", 0)
        m_def.setParam("propagating/maxroundsroot", 0)
        m_def.setParam("limits/time", time_limit)
        m_def.hideOutput(True)
        
        t0 = time.time()
        m_def.optimize()
        t_scip = time.time() - t0
        nodes_scip = m_def.getNNodes()
        obj_scip = m_def.getObjVal()
        
        # === B. GNN Branching ===
        obs, action_set, _, done, info = env.reset(temp_file)
        
        gnn_time_start = time.time()
        gnn_nodes_total = 0
        
        if done:
            # Если SCIP решил задачу сразу в корневом узле
            gnn_nodes_total = 1
        else:
            # Цикл ветвления под управлением нейросети
            while not done:
                # Нейросеть выбирает переменную
                action = gnn_policy(action_set, obs)
                
                # Делаем шаг в SCIP. 
                # info будет содержать результат information_function из настроек env
                obs, action_set, _, done, info = env.step(action)
            
            # Извлекаем итоговое количество узлов из последней порции info
            # "nb_nodes" — это ключ, который мы задали в information_function выше
            gnn_nodes_total = info["nb_nodes"]
                
        gnn_time = time.time() - gnn_time_start
        # Ecole model object wrapper, access underlying scip via getters if needed
        # But we assume runs to completion
        
        # Для Ecole сложно получить ObjVal напрямую, если не настроено вознаграждение
        # Но мы можем считать, что если done=True, то оптимум найден (или лимит)
        
        print(f"Instance {i+1}:")
        print(f"  SCIP: {t_scip:.2f}s, {nodes_scip} nodes")
        print(f"  GNN:  {gnn_time:.2f}s, {gnn_nodes_total} nodes")
        
        scores_total = info["scores"]
        
        results.append({
            "id": i,
            "scip_time": t_scip,
            "gnn_time": gnn_time,
            "scip_nodes": nodes_scip,
            "gnn_nodes": gnn_nodes_total,
            "scores": scores_total
        })
        
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # Итоги
    df = pd.DataFrame(results)
    print("\n=== SUMMARY ===")
    print(df.mean(numeric_only=True))
    
    df.to_csv("evaluation_results.csv", index=False)
    print("Results saved to evaluation_results.csv")

if __name__ == "__main__":
    # Укажи путь к своей лучшей модели (в папке runs или корне)
    # Попробуем найти автоматически
    possible_models = [f for f in os.listdir(".") if f.endswith(".pt") and "best" and "miap" and "max" in f]
    if possible_models:
        MODEL_PATH = possible_models[0]
        print(f"Auto-selected model: {MODEL_PATH}")
        run_evaluation(MODEL_PATH, num_instances=5, n_size=20) # N=5 для скорости отладки
    else:
        print("No .pt model found! Please train a model first.")