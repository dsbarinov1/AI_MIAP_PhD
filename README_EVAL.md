# MIAP GNN Solver Evaluation

This directory contains scripts to evaluate the trained Graph Neural Network (GNN) on Multi-Index Assignment Problem (MIAP) instances using the SCIP solver via Ecole.

## Requirements

The evaluation requires a Python environment with `ecole`, `pyscipopt`, `torch`, and `torch_geometric`.
The recommended way to install Ecole is via Conda, as it depends on the SCIP optimization suite.

### Step 1: Create Environment (if not exists)

```bash
conda create -n miap_eval python=3.10
conda activate miap_eval
```

### Step 2: Install Dependencies

```bash
# Install Ecole and PySCIPOpt
conda install -c conda-forge ecole pyscipopt

# Install PyTorch (adjust for your CUDA version if needed)
pip install torch torchvision torchaudio

# Install Torch Geometric
pip install torch_geometric

# Install other utilities
pip install numpy tqdm
```

### Step 3: Verify Environment

Run the provided check script to ensure all dependencies are correctly installed:

```bash
python check_env_requirements.py
```

## Running the Evaluation

The main evaluation script is `evaluate_solver.py`. It performs the following:
1. Loads the best trained model checkpoint (`best_model_miap_max_ranking_L3_H129_old_max.pt`).
2. Generates random and structured (Euclidean) MIAP instances.
3. Solves each instance twice:
   - **Baseline:** Default SCIP solver.
   - **GNN-Guided:** SCIP with branching decisions made by the GNN.
4. Calculates metrics:
   - **Optimality Gap (%):** Percentage deviation of the GNN solution from the optimal solution found by SCIP.
   - **Solving Time:** Time taken by each method.
   - **Nodes:** Number of branch-and-bound nodes explored.
5. Saves detailed metrics to `evaluation_metrics.csv`.

To run the evaluation:

```bash
python evaluate_solver.py
```

The script will output a summary to the console and save detailed logs to `evaluation_metrics.csv`.

## Configuration

You can modify `evaluate_solver.py` to change:
- `num_instances`: Number of test instances (default: 10).
- `n_size`: Problem size $N$ (default: 10).
- `k_dim`: Number of indices (default: 3).
- `model_path`: Path to the trained model checkpoint.

## Troubleshooting

- **"ecole not found"**: Ensure you activated the conda environment where ecole is installed.
- **"Model file not found"**: Ensure `best_model_miap_max_ranking_L3_H129_old_max.pt` is in the same directory as the script.
