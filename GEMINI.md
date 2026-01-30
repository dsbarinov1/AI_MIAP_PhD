# MIAP GNN Solver

Implements a Bipartite Graph Neural Network (GNN) framework for "Learning to Branch" in Mixed Integer Assignment Problems (MIAP) using Imitation Learning.

## 📊 Current Dataset Statistics
**Total Samples:** 532 branching states extracted from SCIP.
*   **Training:** 441 samples.
*   **Validation:** 91 samples.
*   *Note:* Dataset generation filters for valid branching states (non-leaf nodes) where Strong Branching provides scores.

## 📂 Project Structure

### Core Components

*   **`model.py`** — **Network Architecture**
    *   **Class:** `GasseMultiAggGCN`.
    *   **Type:** Bipartite GCN (Constraints $\leftrightarrow$ Variables).
    *   **Key Features:**
        *   Multi-aggregation message passing (`mean`, `max`, `sum`).
        *   Residual connections & LayerNorm for stability.
        *   Jumping Knowledge (concatenating features from all layers).
        *   Dropout for regularization.

*   **`train.py`** — **Training Loop**
    *   **Loss Functions:**
        *   `ranking` (Default): Pairwise Margin Ranking Loss ($Score_{best} > Score_{other} + \delta$).
        *   `nll`: Negative Log Likelihood (Softmax classification).
    *   **Metrics:** Accuracy@1 (Top-1 match), Accuracy@5 (Top-5 match).
    *   **Logging:** TensorBoard integration.

*   **`generators.py`** — **Problem Generation**
    *   **Class:** `MIAPGenerator`.
    *   **Task:** Axial 3-Index Assignment Problem ($N \times N \times N$).
    *   **Distributions:**
        *   `random_uniform`: Uncorrelated costs $U[0,1]$.
        *   `euclidean`: Structured costs based on 2D coordinates (satisfies triangle inequality).
    *   **Output:** PySCIPOpt Model objects.

*   **`data_collector.py`** — **Expert Demonstrations**
    *   **Oracle:** SCIP Solver with Strong Branching.
    *   **Interface:** Uses **Ecole** to extract bipartite state representations.
    *   **Output:** PyTorch Geometric (`.pt`) files.

*   **`dataset.py`** — **Data Loading**
    *   **Class:** `MIAPDataset` (PyG Dataset).
    *   **Processing:** Z-score normalization of features, One-Hot/Random feature injection (optional for symmetry breaking).

## 🚀 Workflow

### 1. Generate Dataset
Generates instances, solves them using SCIP/Strong Branching, and saves features.

```bash
# Edit __main__ in data_collector.py to change N (size) or count
python data_collector.py
```

### 2. Train Model
Trains the GNN to predict the variable selected by Strong Branching.

```bash
# Recommended baseline configuration
python train.py \
  --epochs 100 \
  --batch_size 32 \
  --hidden 128 \
  --layers 3 \
  --aggr max \
  --loss ranking \
  --device cuda
```

### 3. Monitor
Real-time metrics visualization.

```bash
tensorboard --logdir=runs
```

## 💾 Data Format
Each `.pt` sample contains a `HeteroData` object:
*   `data['variable'].x`: Variable features (Solution value, Reduced cost, etc.).
*   `data['constraint'].x`: Constraint features (Dual solution, etc.).
*   `data['variable', 'adj', 'constraint'].edge_index`: Bipartite adjacency.
*   `data['variable'].y`: Target index (Best variable chosen by Strong Branching).
*   `data['variable'].cand_mask`: Boolean mask of valid branching candidates.
```