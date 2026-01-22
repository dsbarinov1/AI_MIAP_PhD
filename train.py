import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import softmax
from dataset import MIAPDataset
from model import GasseGCN
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--aggr", type=str, default="max", choices=["add", "mean", "max"])
    parser.add_argument("--loss", type=str, default="ranking", choices=["nll", "ranking"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def train():
    args = get_args()
    print(f"--- Training on {args.device} (BS={args.batch_size}, Aggr={args.aggr}, Loss={args.loss}) ---")
    
    writer = SummaryWriter(f"runs/miap_gasse_{args.aggr}_{args.loss}")
    
    # Use normalize=True and force_edge_one=True
    train_ds = MIAPDataset("dataset_train", force_edge_one=True, normalize=True)
    val_ds = MIAPDataset("dataset_val", force_edge_one=True, normalize=True)
    
    if len(train_ds) == 0:
        print("Dataset is empty!")
        return

    sample = train_ds[0]
    dim_c = sample['constraint'].x.shape[1]
    dim_v = sample['variable'].x.shape[1]
    print(f"Dims: Cons={dim_c}, Vars={dim_v}")
    
    train_loader = PyGDataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    model = GasseGCN(dim_cons=dim_c, dim_vars=dim_v, hidden_dim=args.hidden, num_layers=args.layers, aggr=args.aggr).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0
        steps = 0
        
        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()
            
            logits = model(batch)
            
            cand_mask = batch['variable'].cand_mask
            logits[~cand_mask] = -1e9
            
            batch_idx = batch['variable'].batch
            probs = softmax(logits, batch_idx)
            
            # ptr_full has N+1 elements [0, Size1, Size1+Size2, ...]
            ptr_full = batch['variable'].ptr
            ptr_start = ptr_full[:-1] # [0, Size1, ...]
            
            targets_local = batch['variable'].y.reshape(-1) # Ensure 1D [B]
            
            global_targets = ptr_start + targets_local

            if args.loss == "nll":
                target_probs = probs[global_targets]
                loss = -torch.log(target_probs + 1e-9).mean()
            else: # ranking
                # Pairwise Ranking Loss
                # We want Score(Target) > Score(NonTarget) + Margin
                loss = 0
                num_graphs = len(ptr_start)

                # Iterate graphs (vectorizing this is hard due to variable number of candidates)
                ptr_cpu = ptr_full.cpu().numpy()
                targets_local_cpu = targets_local.cpu().numpy()
                cands_mask = batch['variable'].cand_mask

                for i in range(num_graphs):
                    start, end = ptr_cpu[i], ptr_cpu[i+1] # Global indices for this graph

                    # Graph logits
                    g_logits = logits[start:end]

                    # Target index (local)
                    t_idx = targets_local_cpu[i]
                    t_score = g_logits[t_idx]

                    # Candidate mask for this graph
                    g_cands = cands_mask[start:end]

                    # Non-target candidates
                    # Indices where g_cands is True AND idx != t_idx
                    g_cands_indices = torch.nonzero(g_cands).squeeze(-1)
                    nt_indices = g_cands_indices[g_cands_indices != t_idx]

                    if len(nt_indices) > 0:
                        nt_scores = g_logits[nt_indices]

                        # Expand target score to match number of non-targets
                        t_scores_expanded = t_score.expand_as(nt_scores)

                        # Loss: max(0, -target + nontarget + margin)
                        curr_loss = F.margin_ranking_loss(t_scores_expanded, nt_scores, torch.ones_like(nt_scores), margin=0.1)
                        loss += curr_loss

                if num_graphs > 0:
                    loss = loss / num_graphs

            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            steps += 1
            
        avg_loss = loss_sum / steps if steps else 0
        
        # --- VALIDATION ---
        model.eval()
        val_acc1 = 0
        val_acc5 = 0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(args.device)
                logits = model(batch)
                
                cand_mask = batch['variable'].cand_mask
                logits[~cand_mask] = -1e9
                
                ptr = batch['variable'].ptr.cpu().numpy()
                targets = batch['variable'].y.reshape(-1).cpu().numpy()
                logits_cpu = logits.cpu()
                
                for i in range(len(ptr) - 1):
                    start, end = ptr[i], ptr[i+1]
                    graph_logits = logits_cpu[start:end]
                    target = targets[i]
                    
                    if torch.argmax(graph_logits) == target:
                        val_acc1 += 1
                        
                    k = min(5, len(graph_logits))
                    _, topk = torch.topk(graph_logits, k)
                    if target in topk:
                        val_acc5 += 1
                        
                    val_count += 1
        
        acc1 = val_acc1 / val_count if val_count else 0
        acc5 = val_acc5 / val_count if val_count else 0
        
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Val Acc@1: {acc1:.4f} | Val Acc@5: {acc5:.4f}")
        
        writer.add_scalar("Train/Loss", avg_loss, epoch)
        writer.add_scalar("Val/Acc_Top1", acc1, epoch)
        writer.add_scalar("Val/Acc_Top5", acc5, epoch)
        
        if acc1 > best_acc:
            best_acc = acc1
            torch.save(model.state_dict(), "best_model_gasse.pt")

    writer.close()
    print(f"Best Val Acc@1: {best_acc:.4f}")

if __name__ == "__main__":
    train()
