import ecole
import pyscipopt
import numpy as np
import random
import os

class MIAPGenerator:
    def __init__(self, n: int, k: int = 3, seed: int = 42):
        self.n = n
        self.k = k
        self.rng = np.random.default_rng(seed)

    def generate_random_uniform(self):
        shape = tuple([self.n] * self.k)
        return self.rng.random(shape)

    def build_model(self, cost_tensor):
        model = pyscipopt.Model("MIAP")
        vars_dict = {}
        it = np.nditer(cost_tensor, flags=['multi_index'])
        for _ in it:
            idx = it.multi_index
            cost = cost_tensor[idx]
            vname = f"x{'_'.join(map(str, idx))}" 
            vars_dict[idx] = model.addVar(name=vname, vtype="B", obj=cost)
        model.setMinimize()

        for d in range(self.k):
            for i in range(self.n):
                vars_in_con = [v for idx, v in vars_dict.items() if idx[d] == i]
                model.addCons(pyscipopt.quicksum(vars_in_con) == 1)
        
        # Грязное ограничение
        all_vars = list(vars_dict.values())
        subset = random.sample(all_vars, len(all_vars)//5) 
        model.addCons(pyscipopt.quicksum(subset) <= len(subset)//2)
        
        return model

def debug_instance():
    print("\n--- DEBUGGING SCIP BEHAVIOR (SAFE MODE) ---")
    
    # Увеличим N, чтобы точно не решилось в корне
    N = 12 
    K = 3
    print(f"Generating MIAP N={N}, K={K} ({N**K} variables)...")
    
    gen = MIAPGenerator(n=N, k=K)
    c_tensor = gen.generate_random_uniform()
    model = gen.build_model(c_tensor)

    mps_file = "debug.mps"
    model.writeProblem(mps_file)
    print(f"Model saved to {mps_file}")

    print("\n--- Running Ecole ---")
    
    # ТОЛЬКО БАЗОВЫЕ ПАРАМЕТРЫ
    scip_params = {
        "presolving/maxrounds": 0,      # Нет пресолвинга
        "presolving/maxrestarts": 0,    # Нет рестартов
        "separating/maxrounds": 0,      # Нет cuts
        "separating/maxroundsroot": 0,
        "propagating/maxrounds": 0,     # Нет пропагации
        "propagating/maxroundsroot": 0,
        "display/verblevel": 4          # Логи
    }
    
    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        information_function={"scores": ecole.observation.StrongBranchingScores(pseudo_candidates=False)},
        scip_params=scip_params
    )
    
    print("Resetting Ecole env...")
    try:
        # Запускаем
        obs, _, _, done, info = env.reset(mps_file)
        
        if done:
            print("ALERT: Ecole solved it immediately (Done=True).")
            print("Причина: Задача слишком простая или эвристики SCIP слишком сильные.")
        else:
            print("\n>>> SUCCESS: Observation received! <<<")
            if obs is not None:
                print(f"Nodes (Variables): {obs.variable_features.shape[0]}")
                print(f"Nodes (Constraints): {obs.row_features.shape[0]}")
                scores = info.get('scores', [])
                valid_scores = [s for s in scores if not np.isnan(s)]
                print(f"Candidates for branching: {len(valid_scores)}")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    debug_instance()