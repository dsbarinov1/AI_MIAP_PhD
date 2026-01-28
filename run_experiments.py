import itertools
import subprocess
import os
import sys
import time

def run_experiments():
    # Grid Definition
    # Reduce size for demonstration, but user can expand
    aggrs = ["add", "mean", "max", "cat"] # Min is not implemented in model logic, removed
    losses = ["nll", "ranking", "bce", "focal"]
    layers_list = [2, 3] # [1, 2, 3, 4, 5]
    hiddens = [64] # [32, 64, 128]
    batch_sizes = [32] # [16, 32, 64]

    # Total combinations
    combinations = list(itertools.product(aggrs, losses, layers_list, hiddens, batch_sizes))
    print(f"Total experiments to run: {len(combinations)}")

    # CSV Header
    results_file = "experiment_results.csv"
    with open(results_file, "w") as f:
        f.write("Aggr,Loss,Layers,Hidden,BatchSize,Status,LogDir\n")

    for aggr, loss, layers, hidden, bs in combinations:
        run_name = f"exp_{aggr}_{loss}_L{layers}_H{hidden}_BS{bs}"
        print(f"=== Running {run_name} ===")

        cmd = [
            sys.executable, "train.py",
            "--epochs", "10",
            "--aggr", aggr,
            "--loss", loss,
            "--layers", str(layers),
            "--hidden", str(hidden),
            "--batch_size", str(bs),
            "--log_dir", f"runs/{run_name}"
        ]

        try:
            # Run and wait
            subprocess.run(cmd, check=True)
            status = "Success"
        except subprocess.CalledProcessError:
            status = "Failed"
            print(f"!!! Failed: {run_name}")

        with open(results_file, "a") as f:
            f.write(f"{aggr},{loss},{layers},{hidden},{bs},{status},runs/{run_name}\n")

        time.sleep(1) # Cooldown

if __name__ == "__main__":
    run_experiments()
