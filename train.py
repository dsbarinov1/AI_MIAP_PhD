import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import softmax
from dataset import MIAPDataset
from model import GasseGCN
import numpy as np

# --- Loss Functions ---
def compute_ranking_loss(logits, batch, margin=0.1):
    loss = 0
    ptr = batch['variable'].ptr
    targets_local = batch['variable'].y.reshape(-1)
    cands_mask = batch['variable'].cand_mask

    ptr_cpu = ptr.cpu().numpy()
    targets_local_cpu = targets_local.cpu().numpy()

    num_graphs = len(ptr) - 1

    for i in range(num_graphs):
        start, end = ptr_cpu[i], ptr_cpu[i+1]
        g_logits = logits[start:end]
        t_idx = targets_local_cpu[i]

        # Candidate mask for this graph
        g_cands = cands_mask[start:end]

        # Get target score
        t_score = g_logits[t_idx]

        # Get non-target scores (candidates only)
        # We need indices where cands=True but idx != t_idx
        indices = torch.nonzero(g_cands).squeeze(-1)
        nt_indices = indices[indices != t_idx]

        if len(nt_indices) > 0:
            nt_scores = g_logits[nt_indices]
            t_score_exp = t_score.expand_as(nt_scores)
            # max(0, -target + nontarget + margin)
            loss += F.margin_ranking_loss(t_score_exp, nt_scores, torch.ones_like(nt_scores), margin=margin)

    return loss / num_graphs

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    # inputs: Logits
    # targets: 0/1 (same shape)
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    f_loss = alpha * (1-pt)**gamma * bce_loss
    return f_loss.mean()

def compute_bce_loss(logits, batch, focal=False):
    # Construct binary targets (1 for best, 0 for other candidates)
    # We only care about Candidates.

    mask = batch['variable'].cand_mask
    logits_cands = logits[mask]

    # Construct labels for these candidates
    # We need to map global indices to local graph-wise comparison?
    # Actually, simpler: create a global target vector

    # 1. Create global target vector [N_nodes]
    targets = torch.zeros_like(logits)

    # Get global target indices
    ptr = batch['variable'].ptr[:-1]
    targets_local = batch['variable'].y.reshape(-1)
    global_targets = ptr + targets_local

    targets[global_targets] = 1.0

    # 2. Filter by candidates
    targets_cands = targets[mask]

    # 3. Compute Loss
    if focal:
        return focal_loss(logits_cands, targets_cands)
    else:
        # Use pos_weight to handle imbalance?
        # Typically 1 positive vs ~10-20 negatives.
        pos_weight = torch.tensor([10.0], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits_cands, targets_cands, pos_weight=pos_weight)

# --- Training ---

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=129)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--aggr", type=str, default="max", choices=["add", "mean", "max", "min", "cat"])
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "leaky_relu", "tanh", "elu"])
    parser.add_argument("--loss", type=str, default="ranking", choices=["nll", "ranking", "bce", "focal"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="dataset", help="Base path for dataset (appends _train and _val)")
    return parser.parse_args()

def train():
    args = get_args()
    
    if args.tag != None:  
        run_name = f"miap_{args.aggr}_{args.loss}_L{args.layers}_H{args.hidden}_{args.tag}"
    else:
        run_name = f"miap_{args.aggr}_{args.loss}_L{args.layers}_H{args.hidden}"
    log_dir = args.log_dir if args.log_dir else f"runs/{run_name}"

    print(f"--- Training {run_name} on {args.device} ---")
    writer = SummaryWriter(log_dir)
    
    train_ds = MIAPDataset(f"{args.data_path}_train", force_edge_one=True, normalize=True)
    val_ds = MIAPDataset(f"{args.data_path}_val", force_edge_one=True, normalize=True)

    if len(train_ds) == 0: return

    sample = train_ds[0]
    dim_c = sample['constraint'].x.shape[1]
    dim_v = sample['variable'].x.shape[1]
    
    print(f"Detected dimensions -> Constraints: {dim_c}, Variables: {dim_v}")

    train_loader = PyGDataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    model = GasseGCN(dim_cons=dim_c, dim_vars=dim_v, hidden_dim=args.hidden, num_layers=args.layers, aggr=args.aggr, activation=args.activation).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        # --- TRAIN ---
        model.train()
        train_loss_sum = 0
        train_acc1_sum = 0
        train_acc5_sum = 0
        train_steps = 0
        
        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()
            
            logits = model(batch)
            
            # For Accuracy Calculation (always needed)
            with torch.no_grad():
                logits_masked = logits.clone()
                logits_masked[~batch['variable'].cand_mask] = -1e9
                ptr = batch['variable'].ptr.cpu().numpy()
                targets = batch['variable'].y.reshape(-1).cpu().numpy()
                logits_cpu = logits_masked.cpu()

                acc1 = 0
                acc5 = 0
                count = len(targets)

                for i in range(len(ptr) - 1):
                    start, end = ptr[i], ptr[i+1]
                    g_logits = logits_cpu[start:end]
                    target = targets[i]
                    if torch.argmax(g_logits) == target: acc1 += 1
                    k = min(5, len(g_logits))
                    _, topk = torch.topk(g_logits, k)
                    if target in topk: acc5 += 1

                train_acc1_sum += (acc1 / count)
                train_acc5_sum += (acc5 / count)

            # Masking for Loss if needed
            if args.loss == "nll":
                 logits[~batch['variable'].cand_mask] = -1e9

            # Compute Loss
            if args.loss == "nll":
                probs = softmax(logits, batch['variable'].batch)
                ptr = batch['variable'].ptr[:-1]
                targets_local = batch['variable'].y.reshape(-1)
                global_targets = ptr + targets_local
                target_probs = probs[global_targets]
                loss = -torch.log(target_probs + 1e-9).mean()
            elif args.loss == "ranking":
                loss = compute_ranking_loss(logits, batch)
            elif args.loss == "bce":
                loss = compute_bce_loss(logits, batch, focal=False)
            elif args.loss == "focal":
                loss = compute_bce_loss(logits, batch, focal=True)

            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_steps += 1
            
        avg_train_loss = train_loss_sum / train_steps
        avg_train_acc1 = train_acc1_sum / train_steps
        avg_train_acc5 = train_acc5_sum / train_steps
        curr_lr = optimizer.param_groups[0]['lr']
        
        # --- VALIDATION ---
        model.eval()
        val_loss_sum = 0
        val_acc1_sum = 0
        val_acc5_sum = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(args.device)
                logits = model(batch)
                
                # Validation Loss (using same metric as train for consistency)
                if args.loss == "nll":
                     logits_loss = logits.clone()
                     logits_loss[~batch['variable'].cand_mask] = -1e9
                     probs = softmax(logits_loss, batch['variable'].batch)
                     ptr = batch['variable'].ptr[:-1]
                     targets = batch['variable'].y.reshape(-1)
                     global_targets = ptr + targets
                     loss = -torch.log(probs[global_targets] + 1e-9).mean()
                elif args.loss == "ranking":
                    loss = compute_ranking_loss(logits, batch)
                elif args.loss == "bce":
                    loss = compute_bce_loss(logits, batch, focal=False)
                elif args.loss == "focal":
                    loss = compute_bce_loss(logits, batch, focal=True)

                val_loss_sum += loss.item()
                
                # Accuracy
                logits[~batch['variable'].cand_mask] = -1e9
                ptr = batch['variable'].ptr.cpu().numpy()
                targets = batch['variable'].y.reshape(-1).cpu().numpy()
                logits_cpu = logits.cpu()
                
                acc1 = 0
                acc5 = 0
                count = len(targets)
                for i in range(len(ptr) - 1):
                    start, end = ptr[i], ptr[i+1]
                    g_logits = logits_cpu[start:end]
                    target = targets[i]
                    if torch.argmax(g_logits) == target: acc1 += 1
                    k = min(5, len(g_logits))
                    _, topk = torch.topk(g_logits, k)
                    if target in topk: acc5 += 1

                val_acc1_sum += (acc1 / count)
                val_acc5_sum += (acc5 / count)
                val_steps += 1
        
        avg_val_loss = val_loss_sum / val_steps
        avg_val_acc1 = val_acc1_sum / val_steps
        avg_val_acc5 = val_acc5_sum / val_steps

        # Step Scheduler
        scheduler.step(avg_val_acc1)
        
        print(f"Ep {epoch+1:02d} | "
              f"T_Loss: {avg_train_loss:.4f} T_Acc@1: {avg_train_acc1:.4f} T_Acc@5: {avg_train_acc5:.4f} | "
              f"V_Loss: {avg_val_loss:.4f} V_Acc@1: {avg_val_acc1:.4f} V_Acc@5: {avg_val_acc5:.4f} | "
              f"LR: {curr_lr:.1e}", flush=True)
        
        writer.add_scalar("Train/Loss", avg_train_loss, epoch)
        writer.add_scalar("Train/Acc@1", avg_train_acc1, epoch)
        writer.add_scalar("Train/Acc@5", avg_train_acc5, epoch)
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/Acc@1", avg_val_acc1, epoch)
        writer.add_scalar("Val/Acc@5", avg_val_acc5, epoch)
        writer.add_scalar("Info/LR", curr_lr, epoch)
        
        if avg_val_acc1 > best_acc:
            best_acc = avg_val_acc1
            torch.save(model.state_dict(), f"best_model_{run_name}.pt")

    writer.close()

if __name__ == "__main__":
    train()
