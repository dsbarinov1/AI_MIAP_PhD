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
    data_ctr: int = 0
):
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Среда
    observation_function = ecole.observation.NodeBipartite()
    information_function = {
        "scores": ecole.observation.StrongBranchingScores(pseudo_candidates=False)
    }
    
    # Параметры SCIP
    # Убрали heuristics/emphasis, так как он вызывал ошибку
    scip_params = {
        "presolving/maxrounds": 0,           # Главное: отключить пресолвинг
        "presolving/maxrestarts": 0,         # Отключить рестарты
        "separating/maxrounds": 0,           # Отключить cuts
        "separating/maxroundsroot": 0,
        "propagating/maxrounds": 0,
        "propagating/maxroundsroot": 0,
        "limits/time": time_limit,
    }
    
    env = ecole.environment.Branching(
        observation_function=observation_function,
        information_function=information_function,
        scip_params=scip_params
    )
    
    gen = MIAPGenerator(n=n_size, k=k_dim)
    data_counter = data_ctr
    temp_file = f"temp_miap_{os.getpid()}.mps" 

    print(f"Starting collection in '{save_dir}' (N={n_size})...")
    
    for i in tqdm(range(num_instances)):
        try:
            # 2. Генерация
            if i % 2 == 0:
                c_tensor = gen.generate_random_uniform()
                ptype = "random"
            else:
                c_tensor = gen.generate_euclidean()
                ptype = "euclidean"
            
            # Строим модель
            model, _ = gen.build_scip_model(c_tensor)
            
            # Сохраняем в MPS
            model.writeProblem(temp_file)
            
            # 3. Решение
            # Ecole сама загружает файл
            obs, action_set, _, done, info = env.reset(temp_file)
            
            # Если задача решена сразу или obs пустой
            if done or obs is None:
                continue

            # 4. Обработка оценок (Target)
            scores = info["scores"]
            scores = np.nan_to_num(scores, nan=-1e9) # NaN -> min_val
            best_var_idx = np.argmax(scores)
            
            # Проверка: лучший кандидат должен быть валидным (не -inf) и доступным для ветвления
            if scores[best_var_idx] <= -1e8 or best_var_idx not in action_set:
                continue

            # 5. Сохранение в тензоры
            # ИСПРАВЛЕНИЕ: Используем правильные имена из документации
            # row_features -> Признаки ограничений
            # variable_features -> Признаки переменных (было col_features)
            
            data_item = {
                "row_features": torch.tensor(obs.row_features, dtype=torch.float32),
                "col_features": torch.tensor(obs.variable_features, dtype=torch.float32), # <-- FIX
                "edge_indices": torch.tensor(obs.edge_features.indices, dtype=torch.long),
                "edge_attr": torch.tensor(obs.edge_features.values, dtype=torch.float32).unsqueeze(1),
                "label_var_idx": torch.tensor(best_var_idx, dtype=torch.long),
                "scores": torch.tensor(scores, dtype=torch.float32), # Save RAW scores for Soft Targets
                "candidates": torch.tensor(action_set.astype(np.int64), dtype=torch.long),
                "type": ptype
            }
            
            torch.save(data_item, os.path.join(save_dir, f"sample_{data_counter}.pt"))
            data_counter += 1
            
        except Exception as e:
            # print(f"Skip {i}: {e}") # Можно раскомментировать для отладки
            pass
        
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    print(f"Success: Collected {data_counter}/{num_instances} samples.")

if __name__ == "__main__":
    # Запуск
    collect_data(4000, "dataset_train", n_size=10, k_dim=3, data_ctr=4426)
    collect_data(1000, "dataset_val", n_size=10, k_dim=3, data_ctr=885)