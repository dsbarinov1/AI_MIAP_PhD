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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def train():
    args = get_args()
    print(f"--- Training on {args.device} (BS={args.batch_size}) ---")
    
    writer = SummaryWriter("runs/miap_gasse_experiment")
    
    train_ds = MIAPDataset("dataset_train")
    val_ds = MIAPDataset("dataset_val")
    
    if len(train_ds) == 0:
        print("Dataset is empty!")
        return

    # Check dimensions from one sample
    sample = train_ds[0]
    dim_c = sample['constraint'].x.shape[1]
    dim_v = sample['variable'].x.shape[1]
    print(f"Dims: Cons={dim_c}, Vars={dim_v}")
    
    train_loader = PyGDataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    model = GasseGCN(dim_cons=dim_c, dim_vars=dim_v, hidden_dim=args.hidden).to(args.device)
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
            
            # Masking non-candidates
            cand_mask = batch['variable'].cand_mask
            logits[~cand_mask] = -1e9
            
            # Softmax per graph
            batch_idx = batch['variable'].batch
            probs = softmax(logits, batch_idx)
            
            # Target Selection
            # batch['variable'].ptr gives start index of each graph in the concatenated batch
            ptr = batch['variable'].ptr[:-1]
            targets_local = batch['variable'].y.reshape(-1) # Ensure 1D [B]
            
            global_targets = ptr + targets_local
            
            target_probs = probs[global_targets]
            loss = -torch.log(target_probs + 1e-9).mean()
            
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
                
                # Split back to graphs
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
