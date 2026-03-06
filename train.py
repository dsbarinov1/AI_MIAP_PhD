import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import softmax

from dataset import MIAPDataset
from model import GasseGCN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--train_dir", type=str, default="dataset_train")
    parser.add_argument("--val_dir", type=str, default="dataset_val")
    parser.add_argument("--test_dir", type=str, default=None, help="If set, evaluate on test at end and add to summary")
    parser.add_argument("--save_path", type=str, default="best_model.pt")
    parser.add_argument("--log_dir", type=str, default=None, help="Defaults to runs/miap_YYYYMMDD_HHMMSS")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TeeLogger:
    """Дублирует print в файл и stdout."""
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def compute_acc_metrics(model, loader, device):
    """Возвращает (acc1, acc5, count) по датасету."""
    model.eval()
    acc1 = acc5 = count = 0
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
                pred_top1 = int(torch.argmax(graph_logits).item())
                if pred_top1 == target:
                    acc1 += 1
                k = min(5, num_candidates)
                _, topk = torch.topk(graph_logits, k)
                if bool((topk == target).any().item()):
                    acc5 += 1
                count += 1
    return acc1, acc5, count


def train():
    args = get_args()
    set_global_seed(args.seed)

    if args.log_dir is None:
        args.log_dir = os.path.join("runs", "miap_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.log_dir, exist_ok=True)
    if args.save_path == "best_model.pt" or os.path.dirname(args.save_path) == "":
        args.save_path = os.path.join(args.log_dir, os.path.basename(args.save_path))
    log_file = os.path.join(args.log_dir, "train.log")
    csv_path = os.path.join(args.log_dir, "epochs.csv")
    summary_path = os.path.join(args.log_dir, "summary.json")

    tee = TeeLogger(log_file)
    sys.stdout = tee

    print(f"--- Training on {args.device} (BS={args.batch_size}, seed={args.seed}) ---")
    print(f"Log dir: {args.log_dir}")
    writer = SummaryWriter(args.log_dir)

    train_ds = MIAPDataset(args.train_dir)
    val_ds = MIAPDataset(args.val_dir)
    print(f"Dataset: {len(train_ds)} train, {len(val_ds)} val")

    if len(train_ds) == 0:
        raise ValueError(f"Train dataset is empty: {args.train_dir}")
    if len(val_ds) == 0:
        raise ValueError(f"Validation dataset is empty: {args.val_dir}")
    if len(train_ds) < args.batch_size:
        print(f"[WARN] Train samples ({len(train_ds)}) < batch_size ({args.batch_size}); 1 batch/epoch, training may be unstable.")
    if len(train_ds) < 100:
        print(f"[WARN] Few train samples ({len(train_ds)}). For reproducible metrics consider collecting more (e.g. 1000+ or --target_samples 100000).")

    try:
        from dataset_stats import collect_dataset_stats, format_stats
        for name, ddir in [("train", args.train_dir), ("val", args.val_dir)]:
            st = collect_dataset_stats(ddir)
            print(f"\n--- Dataset stats [{name}] {ddir} ---")
            print(format_stats(st))
            stats_json = os.path.join(args.log_dir, f"dataset_stats_{name}.json")
            with open(stats_json, "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[dataset_stats skipped] {e}")

    train_loader = PyGDataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    try:
        raw_sample = torch.load(os.path.join(args.train_dir, train_ds.files[0]), weights_only=True)
    except TypeError:
        raw_sample = torch.load(os.path.join(args.train_dir, train_ds.files[0]))
    dim_c = raw_sample["row_features"].shape[1]
    dim_v = raw_sample["col_features"].shape[1]
    print(f"Detected Input Dims (from train): Cons={dim_c}, Vars={dim_v}")
    try:
        val_sample = torch.load(os.path.join(args.val_dir, val_ds.files[0]), weights_only=True)
    except TypeError:
        val_sample = torch.load(os.path.join(args.val_dir, val_ds.files[0]))
    vc, vv = val_sample["row_features"].shape[1], val_sample["col_features"].shape[1]
    if vc != dim_c or vv != dim_v:
        print(f"[WARN] Val feature dims (Cons={vc}, Vars={vv}) differ from train (Cons={dim_c}, Vars={dim_v}). Model uses train dims.")

    model = GasseGCN(dim_cons=dim_c, dim_vars=dim_v, hidden_dim=args.hidden).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_epoch = -1

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["epoch", "train_loss", "train_acc1", "train_acc5", "val_acc1", "val_acc5"])

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        steps = 0

        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()

            logits = model(batch)
            masked_logits = logits.masked_fill(~batch.cand_mask, -1e9)

            probs = softmax(masked_logits, batch.batch)
            ptr = batch.ptr[:-1]
            targets_local = batch.y.view(-1)
            if ptr.numel() != targets_local.numel():
                raise RuntimeError(
                    f"Batch mismatch: ptr={ptr.numel()} targets={targets_local.numel()}"
                )

            global_targets = ptr + targets_local
            target_probs = probs[global_targets]
            loss = -torch.log(target_probs + 1e-9).mean()

            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            steps += 1

        avg_loss = loss_sum / steps if steps else 0.0

        train_acc1, train_acc5, train_count = compute_acc_metrics(model, train_loader, args.device)
        train_acc1 = train_acc1 / train_count if train_count else 0.0
        train_acc5 = train_acc5 / train_count if train_count else 0.0

        val_acc1, val_acc5, val_count = compute_acc_metrics(model, val_loader, args.device)
        acc1 = val_acc1 / val_count if val_count else 0.0
        acc5 = val_acc5 / val_count if val_count else 0.0

        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            csv.writer(csvfile).writerow([
                epoch + 1, f"{avg_loss:.6f}", f"{train_acc1:.4f}", f"{train_acc5:.4f}",
                f"{acc1:.4f}", f"{acc5:.4f}",
            ])

        print(
            f"Epoch {epoch + 1}/{args.epochs}: Loss {avg_loss:.4f} | "
            f"Train Acc@1: {train_acc1:.4f} Acc@5: {train_acc5:.4f} | "
            f"Val Acc@1: {acc1:.4f} Acc@5: {acc5:.4f}"
        )
        writer.add_scalar("Train/Loss", avg_loss, epoch)
        writer.add_scalar("Train/Acc_Top1", train_acc1, epoch)
        writer.add_scalar("Train/Acc_Top5", train_acc5, epoch)
        writer.add_scalar("Val/Acc_Top1", acc1, epoch)
        writer.add_scalar("Val/Acc_Top5", acc5, epoch)

        if acc1 > best_acc:
            best_acc = acc1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> new best val Acc@1, saved to {args.save_path}")

    writer.close()

    summary = {
        "config": vars(args),
        "best_epoch": best_epoch,
        "best_val_acc1": round(best_acc, 6),
        "final_epoch": args.epochs,
        "last_epoch_train_loss": round(avg_loss, 6),
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "log_dir": args.log_dir,
        "save_path": args.save_path,
    }

    if args.test_dir and os.path.isdir(args.test_dir):
        test_ds = MIAPDataset(args.test_dir)
        if len(test_ds) > 0:
            try:
                ckpt = torch.load(args.save_path, map_location=args.device, weights_only=True)
            except TypeError:
                ckpt = torch.load(args.save_path, map_location=args.device)
            model.load_state_dict(ckpt)
            test_loader = PyGDataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
            test_acc1, test_acc5, test_count = compute_acc_metrics(model, test_loader, args.device)
            summary["test_acc1"] = round(test_acc1 / test_count, 6) if test_count else None
            summary["test_acc5"] = round(test_acc5 / test_count, 6) if test_count else None
            summary["test_samples"] = test_count
            print(f"\n--- Test ({args.test_dir}) ---")
            print(f"Test Acc@1: {summary['test_acc1']:.4f} Acc@5: {summary['test_acc5']:.4f} (n={test_count})")
        else:
            summary["test_error"] = "empty test_dir"
    else:
        summary["test_dir"] = args.test_dir

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nBest Val Acc@1: {best_acc:.4f} (epoch {best_epoch})")
    print(f"Saved best model to: {args.save_path}")
    print(f"Summary: {summary_path}")
    print(f"Epochs CSV: {csv_path}")
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    train()
