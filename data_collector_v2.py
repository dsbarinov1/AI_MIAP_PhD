import os
import torch
import ecole
import numpy as np
import pyscipopt
import random
from tqdm import tqdm
from generators import MIAPGenerator 

def collect_data_v2(
    num_instances: int, 
    save_dir: str, 
    n_size: int = 10, 
    k_dim: int = 3,
    time_limit: float = 60.0
):
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Environment with Hybrid Observation
    # NodeBipartite: for Graph Structure + Static Features
    # Pseudocosts: for Dynamic History
    observation_function = {
        "bipartite": ecole.observation.NodeBipartite(),
        "pseudocosts": ecole.observation.Pseudocosts()
    }
    
    information_function = {
        "scores": ecole.observation.StrongBranchingScores(pseudo_candidates=False)
    }
    
    # SCIP Parameters
    scip_params = {
        "presolving/maxrounds": 0,           
        "presolving/maxrestarts": 0,         
        "separating/maxrounds": 0,           
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
    data_counter = 0
    temp_file = f"temp_miap_v2_{os.getpid()}.mps" 

    print(f"Starting V2 collection (Manual Fix) in '{save_dir}' (N={n_size})...")
    
    # Indices to KEEP from original NodeBipartite (removing constants 1-6, 18)
    keep_indices = [0] + list(range(7, 18))
    
    for i in tqdm(range(num_instances)):
        try:
            # 2. Generation
            if i % 2 == 0:
                c_tensor = gen.generate_random_uniform()
                ptype = "random"
            else:
                c_tensor = gen.generate_euclidean()
                ptype = "euclidean"
            
            model, _ = gen.build_scip_model(c_tensor)
            model.writeProblem(temp_file)
            
            # 3. Solve
            obs, action_set, _, done, info = env.reset(temp_file)
            
            if done or obs is None:
                continue

            # 4. Process Targets
            scores = info["scores"]
            scores = np.nan_to_num(scores, nan=-1e9)
            best_var_idx = np.argmax(scores)
            
            if scores[best_var_idx] <= -1e8 or best_var_idx not in action_set:
                continue

            # 5. Feature Engineering
            bip = obs["bipartite"]
            pseudo = obs["pseudocosts"] # Shape [N_vars]
            
            # a) Filter Variable Features
            vars_static = bip.variable_features[:, keep_indices]
            
            # b) Add Pseudocosts (FIXED: Handle NaNs)
            pseudo_clean = np.nan_to_num(pseudo, nan=0.0)
            vars_dynamic = pseudo_clean.reshape(-1, 1)
            
            # Concatenate: [Static (12) + Dynamic (1)] = 13 features
            vars_final = np.hstack([vars_static, vars_dynamic])
            
            # 6. Save
            data_item = {
                "row_features": torch.tensor(bip.row_features, dtype=torch.float32),
                "col_features": torch.tensor(vars_final, dtype=torch.float32), 
                "edge_indices": torch.tensor(bip.edge_features.indices, dtype=torch.long),
                "edge_attr": torch.tensor(bip.edge_features.values, dtype=torch.float32).unsqueeze(1),
                "label_var_idx": torch.tensor(best_var_idx, dtype=torch.long),
                "scores": torch.tensor(scores, dtype=torch.float32),
                "candidates": torch.tensor(action_set.astype(np.int64), dtype=torch.long),
                "type": ptype
            }
            
            torch.save(data_item, os.path.join(save_dir, f"sample_{data_counter}.pt"))
            data_counter += 1
            
        except Exception as e:
            # print(f"Skip {i}: {e}") 
            pass
        
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    print(f"Success: Collected {data_counter}/{num_instances} samples in {save_dir}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, default=500)
    parser.add_argument("--num_val", type=int, default=100)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    collect_data_v2(args.num_train, "dataset_v2_train", n_size=args.n, k_dim=args.k)
    collect_data_v2(args.num_val, "dataset_v2_val", n_size=args.n, k_dim=args.k)
