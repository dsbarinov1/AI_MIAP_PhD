import argparse
import json
import os
import random
import tempfile
import time

import ecole
import numpy as np
import pyscipopt
import torch

from dataset import observation_to_data
from generators import MIAPGenerator


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def summarize_numeric(values):
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def solve_with_default_scip(
    model: pyscipopt.Model,
    time_limit: float = 60.0,
):
    model.setParam("limits/time", time_limit)
    model.optimize()

    status = str(model.getStatus())
    nodes = int(model.getNNodes())
    solve_time = float(model.getSolvingTime())

    obj_val = None
    try:
        obj_val = float(model.getObjVal())
    except Exception:
        obj_val = None

    return {
        "status": status,
        "nodes": nodes,
        "solve_time_sec": solve_time,
        "objective": obj_val,
    }


def evaluate_default_scip(
    num_instances: int,
    n_size: int,
    k_dim: int,
    seed: int,
    time_limit: float,
    add_dirty_constraint: bool,
    dirty_fraction: float,
    instance_paths: list = None,
):
    """If instance_paths is provided, solve those .mps files; else generate instances in memory."""
    set_global_seed(seed)
    records = []
    wall_start = time.time()

    if instance_paths is not None:
        for instance_id, path in enumerate(instance_paths):
            if not os.path.isfile(path):
                continue
            model = pyscipopt.Model()
            model.readProblem(path)
            result = solve_with_default_scip(model, time_limit=time_limit)
            result["instance_id"] = instance_id
            result["instance_type"] = "from_file"
            records.append(result)
    else:
        generator = MIAPGenerator(n=n_size, k=k_dim, seed=seed)
        split_rng = np.random.default_rng(seed)
        for instance_id in range(num_instances):
            if split_rng.random() < 0.5:
                cost_tensor = generator.generate_random_uniform()
                instance_type = "random"
            else:
                cost_tensor = generator.generate_euclidean()
                instance_type = "euclidean"
            model, _ = generator.build_scip_model(
                cost_tensor,
                add_dirty_constraint=add_dirty_constraint,
                dirty_fraction=dirty_fraction,
            )
            result = solve_with_default_scip(model, time_limit=time_limit)
            result["instance_id"] = instance_id
            result["instance_type"] = instance_type
            records.append(result)

    wall_time = time.time() - wall_start
    solved_statuses = {"optimal", "timelimit"}
    solved_records = [r for r in records if r["status"] in solved_statuses]
    node_values = [r["nodes"] for r in records]
    time_values = [r["solve_time_sec"] for r in records]

    summary = {
        "num_instances": num_instances,
        "solved_ratio_status_based": float(len(solved_records) / max(1, len(records))),
        "nodes": summarize_numeric(node_values),
        "solve_time_sec": summarize_numeric(time_values),
        "wall_time_sec_total": float(wall_time),
    }

    return records, summary


def _make_ecole_env(time_limit: float, seed: int):
    """Build Ecole Branching env with same SCIP params as data_collector."""
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
        observation_function=ecole.observation.NodeBipartite(),
        information_function={},
        scip_params=scip_params,
    )
    env.seed(seed)
    return env


def solve_with_gnn_policy(
    instance_path: str,
    model: torch.nn.Module,
    device: torch.device,
    time_limit: float,
    seed: int,
):
    """
    Solve one instance with the learned GNN branching policy in Ecole.
    Returns dict with status, nodes, solve_time_sec, objective, branching_steps.
    """
    env = _make_ecole_env(time_limit=time_limit, seed=seed)
    t0 = time.perf_counter()
    obs, action_set, _, done, info = env.reset(instance_path)

    branching_steps = 0
    while not done and obs is not None:
        if action_set is None or len(action_set) == 0:
            break
        data = observation_to_data(obs, action_set)
        data = data.to(device)
        with torch.no_grad():
            logits = model(data)
        masked_logits = logits.masked_fill(~data.cand_mask, -1e9)
        global_node_idx = int(torch.argmax(masked_logits).item())
        variable_index = int(global_node_idx - data.num_cons)
        if variable_index not in action_set:
            variable_index = int(action_set[0])
        obs, action_set, _, done, info = env.step(variable_index)
        branching_steps += 1

    solve_time_sec = time.perf_counter() - t0
    status = "optimal" if done else "timelimit"
    nodes = branching_steps

    return {
        "status": status,
        "nodes": nodes,
        "solve_time_sec": solve_time_sec,
        "objective": None,
        "branching_steps": branching_steps,
    }


def evaluate_gnn_policy(
    checkpoint_path: str,
    dataset_dir: str,
    instance_paths: list,
    n_size: int,
    time_limit: float,
    seed: int,
    device: str,
):
    """Run solver with GNN policy on the same instances as default SCIP. instance_paths[i] is path to i-th instance. dim_c/dim_v from dataset_dir first sample."""
    from model import GasseGCN

    sample_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".pt")])
    if not sample_files:
        raise ValueError(f"No .pt samples in dataset_dir for feature dims: {dataset_dir}")
    raw_sample = torch.load(os.path.join(dataset_dir, sample_files[0]), map_location="cpu")
    dim_c = raw_sample["row_features"].shape[1]
    dim_v = raw_sample["col_features"].shape[1]
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    hidden_dim = int(state_dict["cons_embedding.0.weight"].shape[0])
    model = GasseGCN(dim_cons=dim_c, dim_vars=dim_v, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    records = []
    for i, path in enumerate(instance_paths):
        if not os.path.isfile(path):
            continue
        result = solve_with_gnn_policy(
            instance_path=path,
            model=model,
            device=torch.device(device),
            time_limit=time_limit,
            seed=seed + 1000 + i,
        )
        result["instance_id"] = i
        records.append(result)

    node_values = [r["nodes"] for r in records]
    time_values = [r["solve_time_sec"] for r in records]
    solved_statuses = {"optimal", "timelimit"}
    solved_records = [r for r in records if r.get("status", "unknown") in solved_statuses]

    summary = {
        "num_instances": len(records),
        "solved_ratio_status_based": float(len(solved_records) / max(1, len(records))),
        "nodes": summarize_numeric(node_values),
        "solve_time_sec": summarize_numeric(time_values),
    }
    return records, summary


def evaluate_checkpoint_imitation(
    checkpoint_path: str,
    dataset_dir: str,
    batch_size: int,
    device: str,
):
    import torch
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from dataset import MIAPDataset
    from model import GasseGCN

    ds = MIAPDataset(dataset_dir)
    if len(ds) == 0:
        raise ValueError(f"Dataset is empty: {dataset_dir}")

    raw_sample = torch.load(os.path.join(dataset_dir, ds.files[0]), map_location="cpu")
    dim_c = raw_sample["row_features"].shape[1]
    dim_v = raw_sample["col_features"].shape[1]

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    hidden_dim = int(state_dict["cons_embedding.0.weight"].shape[0])
    model = GasseGCN(dim_cons=dim_c, dim_vars=dim_v, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    loader = PyGDataLoader(ds, batch_size=batch_size, shuffle=False)
    top1 = 0
    top5 = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            masked_logits = logits.masked_fill(~batch.cand_mask, -1e9)
            ptr = batch.ptr.cpu().numpy()
            targets = batch.y.view(-1).cpu().numpy()
            masked_logits_cpu = masked_logits.cpu()
            cand_mask_cpu = batch.cand_mask.cpu()

            for i in range(len(ptr) - 1):
                start, end = ptr[i], ptr[i + 1]
                graph_logits = masked_logits_cpu[start:end]
                graph_cand_mask = cand_mask_cpu[start:end]
                num_candidates = int(graph_cand_mask.sum().item())
                if num_candidates == 0:
                    continue
                target = int(targets[i])

                if int(torch.argmax(graph_logits).item()) == target:
                    top1 += 1

                k = min(5, num_candidates)
                _, topk = torch.topk(graph_logits, k)
                if bool((topk == target).any().item()):
                    top5 += 1

                total += 1

    return {
        "num_samples": int(total),
        "acc_top1": float(top1 / max(1, total)),
        "acc_top5": float(top5 / max(1, total)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_instances", type=int, default=50)
    parser.add_argument("--n_size", type=int, default=10)
    parser.add_argument("--k_dim", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time_limit", type=float, default=60.0)
    parser.add_argument("--add_dirty_constraint", action="store_true")
    parser.add_argument("--dirty_fraction", type=float, default=0.1)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default="baseline_eval_results.json")
    return parser.parse_args()


def _generate_instance_paths(num_instances, n_size, k_dim, seed, add_dirty_constraint, dirty_fraction):
    """Generate instances and write to temp dir; return list of paths and the temp dir (caller should cleanup)."""
    set_global_seed(seed)
    generator = MIAPGenerator(n=n_size, k=k_dim, seed=seed)
    split_rng = np.random.default_rng(seed)
    tmpdir = tempfile.mkdtemp(prefix="miap_eval_")
    paths = []
    for i in range(num_instances):
        if split_rng.random() < 0.5:
            cost_tensor = generator.generate_random_uniform()
        else:
            cost_tensor = generator.generate_euclidean()
        model, _ = generator.build_scip_model(
            cost_tensor,
            add_dirty_constraint=add_dirty_constraint,
            dirty_fraction=dirty_fraction,
        )
        path = os.path.join(tmpdir, f"instance_{i:05d}.mps")
        model.writeProblem(path)
        paths.append(path)
    return paths, tmpdir


if __name__ == "__main__":
    args = parse_args()

    instance_paths, tmpdir = _generate_instance_paths(
        num_instances=args.num_instances,
        n_size=args.n_size,
        k_dim=args.k_dim,
        seed=args.seed,
        add_dirty_constraint=args.add_dirty_constraint,
        dirty_fraction=args.dirty_fraction,
    )

    records, default_summary = evaluate_default_scip(
        num_instances=args.num_instances,
        n_size=args.n_size,
        k_dim=args.k_dim,
        seed=args.seed,
        time_limit=args.time_limit,
        add_dirty_constraint=args.add_dirty_constraint,
        dirty_fraction=args.dirty_fraction,
        instance_paths=instance_paths,
    )

    output = {
        "config": vars(args),
        "default_scip_summary": default_summary,
        "default_scip_records": records,
    }

    if args.checkpoint_path and args.dataset_dir:
        try:
            output["imitation_checkpoint_metrics"] = evaluate_checkpoint_imitation(
                checkpoint_path=args.checkpoint_path,
                dataset_dir=args.dataset_dir,
                batch_size=args.batch_size,
                device=args.device,
            )
        except Exception as e:
            output["imitation_checkpoint_metrics_error"] = str(e)

        try:
            gnn_records, gnn_summary = evaluate_gnn_policy(
                checkpoint_path=args.checkpoint_path,
                dataset_dir=args.dataset_dir,
                instance_paths=instance_paths,
                n_size=args.n_size,
                time_limit=args.time_limit,
                seed=args.seed,
                device=args.device,
            )
            output["gnn_policy_summary"] = gnn_summary
            output["gnn_policy_records"] = gnn_records
        except Exception as e:
            output["gnn_policy_error"] = str(e)

    try:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved evaluation results to {args.output_json}")
