import os
import torch
import ecole
import numpy as np
import pyscipopt
import random
from tqdm import tqdm
from generators import MIAPGenerator 

def collect_data(
    num_instances: int, 
    save_dir: str, 
    n_size: int = 10, 
    k_dim: int = 3,
    time_limit: float = 60.0,
    samples_per_instance: int = 10 # <-- НОВОЕ: Сколько сэмплов брать с одной задачи
):
    os.makedirs(save_dir, exist_ok=True)
    
    # Настраиваем среду
    observation_function = ecole.observation.NodeBipartite()
    information_function = {
        "scores": ecole.observation.StrongBranchingScores(pseudo_candidates=False)
    }
    
    # Параметры SCIP
    scip_params = {
        "presolving/maxrounds": 0,       # Без упрощения
        "separating/maxrounds": 0,       # Без cuts
        "separating/maxroundsroot": 0,
        "limits/time": time_limit,
        # ВАЖНО: Включаем решение LP
        "lp/solvefreq": 1,               # Решать LP в каждом узле
        "lp/presolving": True,           # Разрешить пресолвинг ТОЛЬКО для LP (это дешево)
    }
    
    env = ecole.environment.Branching(
        observation_function=observation_function,
        information_function=information_function,
        scip_params=scip_params
    )
    
    gen = MIAPGenerator(n=n_size, k=k_dim)
    data_counter = 0
    temp_file = f"temp_miap_{os.getpid()}.mps" 

    print(f"Starting collection in '{save_dir}'...")
    
    for i in tqdm(range(num_instances)):
        try:
            # 1. Генерация
            if i % 2 == 0:
                c_tensor = gen.generate_random_uniform()
                ptype = "random"
            else:
                c_tensor = gen.generate_euclidean()
                ptype = "euclidean"
            
            model, _ = gen.build_scip_model(c_tensor)
            model.writeProblem(temp_file)
            
            # 2. Запуск среды
            obs, action_set, _, done, info = env.reset(temp_file)
            
            # Внутренний цикл по узлам дерева
            samples_collected_here = 0
            
            while not done and samples_collected_here < samples_per_instance:
                # Если наблюдения нет (бывает в SCIP), просто делаем шаг дальше
                if obs is None:
                    # Случайное действие, чтобы продвинуть солвер
                    action = action_set[0]
                    obs, action_set, _, done, info = env.step(action)
                    continue

                scores = info["scores"]
                scores = np.nan_to_num(scores, nan=-1e9)
                best_var_idx = np.argmax(scores)
                
                # Если SB не дал оценки (все -inf) или индекс невалиден
                if scores[best_var_idx] <= -1e8 or best_var_idx not in action_set:
                    # Просто делаем шаг (ветвимся по первому доступному)
                    action = action_set[0]
                    obs, action_set, _, done, info = env.step(action)
                    continue

                # Сохраняем сэмпл
                data_item = {
                    "row_features": torch.tensor(obs.row_features, dtype=torch.float32),
                    "col_features": torch.tensor(obs.variable_features, dtype=torch.float32),
                    "edge_indices": torch.tensor(obs.edge_features.indices, dtype=torch.long),
                    "edge_attr": torch.tensor(obs.edge_features.values, dtype=torch.float32).unsqueeze(1),
                    "label_var_idx": torch.tensor(best_var_idx, dtype=torch.long),
                    "candidates": torch.tensor(action_set.astype(np.int64), dtype=torch.long),
                    "type": ptype
                }
                
                torch.save(data_item, os.path.join(save_dir, f"sample_{data_counter}.pt"))
                data_counter += 1
                samples_collected_here += 1
                
                # 3. Делаем шаг в среде (имитируем выбор эксперта)
                # Мы говорим солверу: "Ветвись по переменной best_var_idx"
                # И переходим к следующему узлу
                obs, action_set, _, done, info = env.step(best_var_idx)
            
        except Exception as e:
            print(f"Error on instance {i}: {e}")
            pass
        
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    print(f"Success: Collected {data_counter} samples from {num_instances} instances.")

if __name__ == "__main__":
    # Собираем 50 задач * 10 сэмплов = 500 сэмплов (быстро)
    # Или 100 задач * 20 сэмплов = 2000 сэмплов (лучше)
    collect_data(500, "dataset_train", n_size=10, k_dim=3, samples_per_instance=20)
    collect_data(100, "dataset_val", n_size=10, k_dim=3, samples_per_instance=20)