import argparse
import os
import random

import ecole
import numpy as np
import torch
from tqdm import tqdm

from generators import MIAPGenerator


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sanitize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)


DIFFICULTY_N = {"easy": 10, "medium": 15, "hard": 20}
DIFFICULTY_STEPS = {"easy": 15, "medium": 25, "hard": 40}


def collect_data(
    num_instances: int,
    save_dir: str,
    n_size: int = 10,
    k_dim: int = 3,
    time_limit: float = 60.0,
    seed: int = 42,
    split_name: str = "train",
    max_steps_per_instance: int = 10,
    add_dirty_constraint: bool = False,
    dirty_fraction: float = 0.1,
    target_samples: int = None,
):
    """
    Collect branching state-action pairs. If target_samples is set, sample instances
    (with replacement) until at least target_samples samples are collected (Gasse-style).
    """
    os.makedirs(save_dir, exist_ok=True)
    set_global_seed(seed)

    observation_function = ecole.observation.NodeBipartite()
    information_function = {
        "scores": ecole.observation.StrongBranchingScores(pseudo_candidates=False)
    }

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
        scip_params=scip_params,
    )
    env.seed(seed)

    gen = MIAPGenerator(n=n_size, k=k_dim, seed=seed)
    sample_rng = np.random.default_rng(seed)

    data_counter = 0
    temp_file = f"temp_miap_{split_name}_{os.getpid()}.mps"

    stats = {
        "instances_seen": 0,
        "instances_terminal_at_reset": 0,
        "instances_with_exception": 0,
        "states_collected": 0,
        "states_stepped": 0,
        "states_skipped_empty_action_set": 0,
        "states_skipped_invalid_score": 0,
    }

    use_target_samples = target_samples is not None and target_samples > 0
    print(
        f"[collector] split={split_name} seed={seed} N={n_size} K={k_dim} "
        f"instances={num_instances} max_steps={max_steps_per_instance} "
        f"dirty={add_dirty_constraint}"
        + (f" target_samples={target_samples}" if use_target_samples else "")
    )

    def make_instance(instance_id):
        """Deterministic instance from seed + instance_id."""
        inst_gen = MIAPGenerator(n=n_size, k=k_dim, seed=seed + instance_id)
        if (seed + instance_id) % 2 == 0:
            return inst_gen.generate_random_uniform(), "random"
        return inst_gen.generate_euclidean(), "euclidean"

    if use_target_samples:
        pbar = tqdm(desc="samples", total=target_samples)
    else:
        pbar = tqdm(range(num_instances), desc="instances")

    iteration = 0
    while True:
        if use_target_samples and data_counter >= target_samples:
            break
        if not use_target_samples and iteration >= num_instances:
            break
        instance_id = sample_rng.integers(0, num_instances) if use_target_samples else iteration
        iteration += 1
        stats["instances_seen"] += 1
        if not use_target_samples:
            pbar.update(1)

        try:
            c_tensor, ptype = make_instance(instance_id)
            gen = MIAPGenerator(n=n_size, k=k_dim, seed=seed + instance_id)
            model, _ = gen.build_scip_model(
                c_tensor,
                add_dirty_constraint=add_dirty_constraint,
                dirty_fraction=dirty_fraction,
            )
            model.writeProblem(temp_file)

            obs, action_set, _, done, info = env.reset(temp_file)
            if done or obs is None:
                stats["instances_terminal_at_reset"] += 1
                continue

            step_id = 0
            while not done and obs is not None and step_id < max_steps_per_instance:
                if action_set is None or len(action_set) == 0:
                    stats["states_skipped_empty_action_set"] += 1
                    break

                action_set_np = np.asarray(action_set, dtype=np.int64)
                scores = sanitize_scores(info["scores"])
                cand_scores = scores[action_set_np]

                if cand_scores.size == 0:
                    stats["states_skipped_empty_action_set"] += 1
                    break

                best_local_idx = int(np.argmax(cand_scores))
                best_var_idx = int(action_set_np[best_local_idx])
                best_score = float(cand_scores[best_local_idx])

                if best_score <= -1e8:
                    stats["states_skipped_invalid_score"] += 1
                    break

                data_item = {
                    "row_features": torch.tensor(obs.row_features, dtype=torch.float32),
                    "col_features": torch.tensor(
                        obs.variable_features, dtype=torch.float32
                    ),
                    "edge_indices": torch.tensor(
                        obs.edge_features.indices, dtype=torch.long
                    ),
                    "edge_attr": torch.tensor(
                        obs.edge_features.values, dtype=torch.float32
                    ).unsqueeze(1),
                    "label_var_idx": torch.tensor(best_var_idx, dtype=torch.long),
                    "candidates": torch.tensor(action_set_np, dtype=torch.long),
                    "type": ptype,
                    "split": split_name,
                    "instance_id": instance_id,
                    "step_id": step_id,
                    "seed": seed,
                    "n_size": n_size,
                    "k_dim": k_dim,
                    "best_score": best_score,
                }

                sample_name = f"sample_{data_counter:08d}.pt"
                torch.save(data_item, os.path.join(save_dir, sample_name))
                data_counter += 1
                stats["states_collected"] += 1
                if use_target_samples:
                    pbar.update(1)
                    pbar.set_postfix({"samples": data_counter})

                expert_action = best_var_idx
                obs, action_set, _, done, info = env.step(expert_action)
                step_id += 1
                stats["states_stepped"] += 1

        except Exception as e:
            stats["instances_with_exception"] += 1
            if stats["instances_with_exception"] <= 5:
                print(f"[collector][warn] instance {instance_id} skipped: {e}")

    pbar.close()
    if os.path.exists(temp_file):
        os.remove(temp_file)

    avg_states_per_instance = stats["states_collected"] / max(1, stats["instances_seen"])
    print(
        "[collector][summary] "
        f"saved={stats['states_collected']} "
        f"avg_states_per_instance={avg_states_per_instance:.3f} "
        f"terminal_at_reset={stats['instances_terminal_at_reset']} "
        f"empty_action_set={stats['states_skipped_empty_action_set']} "
        f"invalid_score={stats['states_skipped_invalid_score']} "
        f"exceptions={stats['instances_with_exception']}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_instances", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="dataset_train")
    parser.add_argument("--n_size", type=int, default=10)
    parser.add_argument("--k_dim", type=int, default=3)
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=list(DIFFICULTY_N),
        default=None,
        help="Override n_size and max_steps: easy=10/15, medium=15/25, hard=20/40",
    )
    parser.add_argument("--time_limit", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_name", type=str, default="train")
    parser.add_argument("--max_steps_per_instance", type=int, default=10)
    parser.add_argument(
        "--target_samples",
        type=int,
        default=None,
        help="If set, collect until this many samples (Gasse-style, sample instances with replacement)",
    )
    parser.add_argument("--add_dirty_constraint", action="store_true")
    parser.add_argument("--dirty_fraction", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_size = args.n_size
    max_steps = args.max_steps_per_instance
    if args.difficulty:
        n_size = DIFFICULTY_N[args.difficulty]
        max_steps = DIFFICULTY_STEPS[args.difficulty]
    collect_data(
        num_instances=args.num_instances,
        save_dir=args.save_dir,
        n_size=n_size,
        k_dim=args.k_dim,
        time_limit=args.time_limit,
        seed=args.seed,
        split_name=args.split_name,
        max_steps_per_instance=max_steps,
        add_dirty_constraint=args.add_dirty_constraint,
        dirty_fraction=args.dirty_fraction,
        target_samples=args.target_samples,
    )
