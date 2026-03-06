#!/usr/bin/env bash
# Run data collection then training. Requires conda env miap_phd (see setup_env.sh).
# Usage: bash run_full_pipeline.sh [--quick]  (--quick: fewer instances and epochs)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ENV_NAME="${CONDA_ENV:-miap_phd}"

if ! command -v conda &>/dev/null; then
  echo "conda not in PATH. Run: source \$HOME/miniconda3/etc/profile.d/conda.sh (or your conda path)"
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME" || { echo "Create env first: bash setup_env.sh"; exit 1; }

QUICK=""
for x in "$@"; do
  [ "$x" = "--quick" ] && QUICK=1
done

if [ -n "$QUICK" ]; then
  N_TRAIN=50
  N_VAL=20
  N_TEST=20
  EPOCHS=5
  MAX_STEPS=5
else
  N_TRAIN=500
  N_VAL=100
  N_TEST=100
  EPOCHS=30
  MAX_STEPS=15
fi

echo "[pipeline] Collecting train ($N_TRAIN instances)..."
python data_collector.py --num_instances "$N_TRAIN" --save_dir dataset_train --split_name train --seed 101 --n_size 10 --k_dim 3 --max_steps_per_instance "$MAX_STEPS"
echo "[pipeline] Collecting val ($N_VAL instances)..."
python data_collector.py --num_instances "$N_VAL" --save_dir dataset_val --split_name val --seed 202 --n_size 10 --k_dim 3 --max_steps_per_instance "$MAX_STEPS"
echo "[pipeline] Collecting test ($N_TEST instances)..."
python data_collector.py --num_instances "$N_TEST" --save_dir dataset_test --split_name test --seed 303 --n_size 10 --k_dim 3 --max_steps_per_instance "$MAX_STEPS"

echo "[pipeline] Training ($EPOCHS epochs)..."
python train.py --train_dir dataset_train --val_dir dataset_val --test_dir dataset_test --epochs "$EPOCHS" --batch_size 32 --seed 101

echo "[pipeline] Done. Check runs/ for logs and summary.json"
