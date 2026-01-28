import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax
from dataset import MIAPDataset
from model import GasseHeteroGCN
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_dir", type=str, default="dataset_train")
    parser.add_argument("--val_dir", type=str, default="dataset_val")
    parser.add_argument("--save_path", type=str, default="best_hetero_model.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_batch_accuracy(logits, batch_obj):
    """
    Вспомогательная функция для расчета точности в гетерогенном батче.
    """
    correct = 0
    total = 0
    
    # Клонируем и маскируем для расчета argmax
    temp_logits = logits.detach().clone()
    temp_logits[~batch_obj['variable'].cand_mask] = -1e9
    
    ptr = batch_obj['variable'].ptr.cpu().numpy()
    y = batch_obj['variable'].y.flatten().cpu().numpy()
    logits_cpu = temp_logits.cpu().numpy()
    
    for i in range(len(ptr)-1):
        graph_logits = logits_cpu[ptr[i]:ptr[i+1]]
        if np.argmax(graph_logits) == y[i]:
            correct += 1
        total += 1
    return correct, total

def train():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    writer = SummaryWriter("runs/miap_refined_v2")
    
    train_ds = MIAPDataset(args.train_dir)
    val_ds = MIAPDataset(args.val_dir)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    sample = train_ds[0]
    model = GasseHeteroGCN(
        dim_cons=sample['constraint'].x.shape[1],
        dim_vars=sample['variable'].x.shape[1],
        hidden_dim=args.hidden
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

    best_val_acc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            logits = model(batch)
            mask = batch['variable'].cand_mask
            
            # Расчет Loss
            # Ставим -1e9 только для софтмакса в лоссе
            masked_logits_for_loss = logits.clone()
            masked_logits_for_loss[~mask] = -1e9
            
            probs = softmax(masked_logits_for_loss, batch['variable'].batch)
            ptr = batch['variable'].ptr[:-1]
            targets = batch['variable'].y.flatten() + ptr
            
            loss = -torch.log(probs[targets] + 1e-9).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Расчет Train Accuracy
            c, t = compute_batch_accuracy(logits, batch)
            train_correct += c
            train_total += t

        train_acc = train_correct / train_total
        
        # Валидация
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0
        
        with torch.no_grad():
            for v_batch in val_loader:
                v_batch = v_batch.to(device)
                v_logits = model(v_batch)
                
                # Val Loss
                v_mask = v_batch['variable'].cand_mask
                v_logits_masked = v_logits.clone()
                v_logits_masked[~v_mask] = -1e9
                v_probs = softmax(v_logits_masked, v_batch['variable'].batch)
                v_ptr = v_batch['variable'].ptr[:-1]
                v_targets = v_batch['variable'].y.flatten() + v_ptr
                v_loss = -torch.log(v_probs[v_targets] + 1e-9).mean()
                val_loss_sum += v_loss.item()

                # Val Acc
                c, t = compute_batch_accuracy(v_logits, v_batch)
                val_correct += c
                val_total += t
        
        val_acc = val_correct / val_total
        avg_train_loss = total_loss / len(loader)
        avg_val_loss = val_loss_sum / len(val_loader)
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch:03d} | TrainLoss: {avg_train_loss:.4f} | TrainAcc: {train_acc:.4f} | ValAcc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")
        
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)

    writer.close()
    print(f"--- Finished. Best Val Acc: {best_val_acc:.4f} ---")

if __name__ == "__main__":
    train()