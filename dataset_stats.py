"""
Сбор статистики по датасету (папка с .pt сэмплами) для анализа данных перед обучением.
Использование: python dataset_stats.py --dir dataset_train [--out dataset_train_stats.json]
"""
import argparse
import json
import os
import sys

import torch


def collect_dataset_stats(dirpath: str):
    """
    Сканирует папку с .pt файлами (сэмплы data_collector) и возвращает словарь статистик.
    """
    files = sorted([f for f in os.listdir(dirpath) if f.endswith(".pt")])
    if not files:
        return {"error": "no .pt files", "dir": dirpath}

    n_cons_list = []
    n_vars_list = []
    n_candidates_list = []
    step_ids = []
    types = []
    n_sizes = []
    k_dims = []
    best_scores = []

    for f in files:
        path = os.path.join(dirpath, f)
        try:
            d = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            continue
        row = d.get("row_features")
        col = d.get("col_features")
        cand = d.get("candidates")
        if row is not None:
            n_cons_list.append(row.shape[0])
        if col is not None:
            n_vars_list.append(col.shape[0])
        if cand is not None:
            n_candidates_list.append(cand.numel() if torch.is_tensor(cand) else len(cand))
        if "step_id" in d:
            step_ids.append(int(d["step_id"]))
        if "type" in d:
            types.append(str(d["type"]))
        if "n_size" in d:
            n_sizes.append(int(d["n_size"]))
        if "k_dim" in d:
            k_dims.append(int(d["k_dim"]))
        if "best_score" in d:
            try:
                best_scores.append(float(d["best_score"]))
            except Exception:
                pass

    def desc(name, arr):
        if not arr:
            return None
        arr = list(arr)
        return {
            "count": len(arr),
            "min": min(arr),
            "max": max(arr),
            "mean": round(sum(arr) / len(arr), 4),
        }

    type_counts = {}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1

    step_counts = {}
    for s in step_ids:
        step_counts[str(s)] = step_counts.get(str(s), 0) + 1

    stats = {
        "dir": dirpath,
        "num_samples": len(files),
        "num_loaded": len(n_cons_list),
        "n_cons": desc("n_cons", n_cons_list),
        "n_vars": desc("n_vars", n_vars_list),
        "n_candidates": desc("n_candidates", n_candidates_list),
        "step_id": desc("step_id", step_ids),
        "step_id_distribution": dict(sorted(step_counts.items(), key=lambda x: int(x[0]))),
        "type_distribution": type_counts,
        "n_size_distribution": desc("n_size", n_sizes) if n_sizes else None,
        "k_dim_distribution": desc("k_dim", k_dims) if k_dims else None,
        "best_score": desc("best_score", best_scores) if best_scores else None,
    }
    return stats


def format_stats(stats: dict) -> str:
    """Форматирует словарь статистик в читаемый текст."""
    if "error" in stats:
        return f"Error: {stats['error']} dir={stats.get('dir', '')}"
    lines = [
        f"Dataset: {stats['dir']}",
        f"  num_samples: {stats['num_samples']} (loaded: {stats['num_loaded']})",
    ]
    for key in ["n_cons", "n_vars", "n_candidates", "step_id", "n_size_distribution", "best_score"]:
        v = stats.get(key)
        if v and isinstance(v, dict):
            lines.append(f"  {key}: {v}")
    if stats.get("type_distribution"):
        lines.append(f"  type_distribution: {stats['type_distribution']}")
    if stats.get("step_id_distribution"):
        lines.append(f"  step_id_distribution: {stats['step_id_distribution']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics for .pt sample directories")
    parser.add_argument("--dir", type=str, required=True, help="Directory with .pt files")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()
    if not os.path.isdir(args.dir):
        print(f"Not a directory: {args.dir}", file=sys.stderr)
        sys.exit(1)
    stats = collect_dataset_stats(args.dir)
    print(format_stats(stats))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
