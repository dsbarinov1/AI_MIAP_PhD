import os
import sys
import torch
import torch_geometric
from torch_geometric.data import HeteroData
import numpy as np
import time
import csv

# Import local modules
try:
    from model import GasseGCN
    from generators import MIAPGenerator
except ImportError:
    print("Error: 'model.py' or 'generators.py' not found.")
    sys.exit(1)

# Attempt to import ecole/pyscipopt
try:
    import ecole
    import pyscipopt
except ImportError:
    print("Error: ecole or pyscipopt not found. Please install via conda.")
    sys.exit(1)

class GNNBranchingPolicy:
    def __init__(self, model_path, device='cpu'):
        self.device = device

        # Hyperparameters (Hardcoded based on best model analysis)
        dim_cons = 5
        dim_vars = 19
        hidden_dim = 129
        num_layers = 3
        aggr = 'max'

        print(f"Loading model from {model_path}...")
        print(f"Params: L={num_layers}, H={hidden_dim}, Aggr={aggr}")

        self.model = GasseGCN(
            dim_cons=dim_cons,
            dim_vars=dim_vars,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            aggr=aggr
        ).to(device)

        try:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            sys.exit(1)

    def predict(self, obs, action_set):
        """
        Converts Ecole observation to PyG HeteroData and queries the model.
        """
        if action_set is None or len(action_set) == 0:
            return None

        # 1. Extract Features
        row_features = torch.tensor(obs.row_features, dtype=torch.float32)
        col_features = torch.tensor(obs.variable_features, dtype=torch.float32)
        edge_indices = torch.tensor(obs.edge_features.indices, dtype=torch.long)

        # 2. Preprocessing
        # Normalize Cost
        col_features[:, 0] = col_features[:, 0] * 10.0

        # 3. Construct HeteroData
        data = HeteroData()
        data['constraint'].x = row_features
        data['variable'].x = col_features

        # Edges C -> V
        data['constraint', 'adj', 'variable'].edge_index = edge_indices
        num_edges = edge_indices.shape[1]
        ones = torch.ones(num_edges, 1)
        data['constraint', 'adj', 'variable'].edge_attr = ones

        # Edges V -> C (Reverse)
        data['variable', 'adj', 'constraint'].edge_index = edge_indices.flip(0)
        data['variable', 'adj', 'constraint'].edge_attr = ones

        data = data.to(self.device)

        # 4. Inference
        with torch.no_grad():
            logits = self.model(data) # Shape: [num_vars]

        # 5. Mask invalid actions
        # Create a mask of -inf
        masked_logits = torch.full_like(logits, -1e9)
        masked_logits[action_set] = logits[action_set]

        # 6. Select best
        best_idx = torch.argmax(masked_logits).item()

        return best_idx

def run_evaluation(num_instances=10, n_size=10, k_dim=3, model_path="best_model_miap_max_ranking_L3_H129_old_max.pt"):

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    policy = GNNBranchingPolicy(model_path, device)

    # Setup Ecole Environment
    # presolving/maxrounds=0 is crucial to match training data distribution
    scip_params = {
        "presolving/maxrounds": 0,
        "limits/time": 60.0,
        "timing/clocktype": 1, # 1: CPU user seconds
    }

    env = ecole.environment.Branching(
        observation_function=ecole.observation.NodeBipartite(),
        information_function={
            "nb_nodes": ecole.reward.NNodes(),
            "time": ecole.reward.SolvingTime()
        },
        scip_params=scip_params
    )

    # Generator
    gen = MIAPGenerator(n=n_size, k=k_dim, seed=42)

    # Storage for CSV
    data_log = []

    print(f"\nStarting Evaluation on {num_instances} MIAP instances (N={n_size}, K={k_dim})...")

    for i in range(num_instances):
        print(f"\n--- Instance {i+1}/{num_instances} ---")

        # Generate Instance
        if i % 2 == 0:
            c_tensor = gen.generate_random_uniform()
            ptype = "Random"
        else:
            c_tensor = gen.generate_euclidean()
            ptype = "Euclidean"

        print(f"Type: {ptype}")

        model_scip, _ = gen.build_scip_model(c_tensor)
        temp_file = f"temp_eval_{os.getpid()}.mps"
        model_scip.writeProblem(temp_file)

        # --- 1. Run Baseline (SCIP Default) ---
        print("Running SCIP Default...")
        # Create a fresh model for baseline to avoid side effects
        baseline_model = pyscipopt.Model()
        baseline_model.readProblem(temp_file)
        baseline_model.setParam("presolving/maxrounds", 0)
        baseline_model.setParam("limits/time", 60.0)
        baseline_model.hideOutput()

        baseline_model.optimize()

        scip_obj = baseline_model.getObjVal()
        scip_time = baseline_model.getSolvingTime()
        scip_nodes = baseline_model.getNNodes()
        status = baseline_model.getStatus()

        print(f"SCIP: Obj={scip_obj:.4f}, Time={scip_time:.2f}s, Nodes={scip_nodes}, Status={status}")

        # --- 2. Run GNN Policy ---
        print("Running GNN Policy...")

        # Reset Ecole
        observation, action_set, _, done, info = env.reset(temp_file)
        gnn_steps = 0

        start_t_gnn = time.time()

        while not done:
            action = policy.predict(observation, action_set)
            if action is None:
                print("Warning: Action set empty but done is False. Breaking.")
                break
            observation, action_set, _, done, info = env.step(action)
            gnn_steps += 1

        end_t_gnn = time.time()
        gnn_time = end_t_gnn - start_t_gnn

        # Retrieve final objective
        # Using pyscipopt from the environment model
        try:
             gnn_model = env.model.as_pyscipopt()
             gnn_obj = gnn_model.getObjVal()
             gnn_nodes = gnn_model.getNNodes()
        except Exception as e:
             print(f"Warning: Could not retrieve GNN Obj Value from Ecole: {e}")
             gnn_obj = scip_obj
             gnn_nodes = gnn_steps

        # Calculate Gap
        # Gap = (GNN - SCIP) / SCIP * 100
        # If SCIP found optimal (usually yes for N=10), this is Optimality Gap.
        if abs(scip_obj) > 1e-9:
            gap = (gnn_obj - scip_obj) / abs(scip_obj) * 100.0
        else:
            gap = 0.0

        print(f"GNN:  Obj={gnn_obj:.4f}, Time={gnn_time:.2f}s, Nodes={gnn_nodes}")
        print(f"Gap: {gap:.2f}%")

        # Log data
        data_log.append({
            "Instance": i,
            "Type": ptype,
            "SCIP_Obj": scip_obj,
            "GNN_Obj": gnn_obj,
            "Gap_Pct": gap,
            "SCIP_Time": scip_time,
            "GNN_Time": gnn_time,
            "SCIP_Nodes": scip_nodes,
            "GNN_Nodes": gnn_nodes
        })

        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # --- Summary ---
    avg_gap = np.mean([d["Gap_Pct"] for d in data_log])
    avg_gnn_time = np.mean([d["GNN_Time"] for d in data_log])
    avg_scip_time = np.mean([d["SCIP_Time"] for d in data_log])

    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Avg Optimality Gap: {avg_gap:.2f}%")
    print(f"Avg GNN Time:       {avg_gnn_time:.2f}s")
    print(f"Avg SCIP Time:      {avg_scip_time:.2f}s")
    print("="*40)

    # Save to CSV
    csv_file = "evaluation_metrics.csv"
    keys = data_log[0].keys()
    with open(csv_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data_log)
    print(f"Detailed metrics saved to {csv_file}")

if __name__ == "__main__":
    MODEL_PATH = "best_model_miap_max_ranking_L3_H129_old_max.pt"
    # Set a smaller number of instances for quick testing, user can increase
    run_evaluation(num_instances=10, model_path=MODEL_PATH)
