import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
    parser.add_argument("--lr", type=float, default=0.0005) # Чуть поменьше
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def train():
    args = get_args()
    print(f"--- Training on {args.device} (BS={args.batch_size}) ---")
    
    # TensorBoard: логи будут в папке runs/miap_experiment
    writer = SummaryWriter("runs/miap_experiment")
    
    train_ds = MIAPDataset("dataset_train")
    val_ds = MIAPDataset("dataset_val")
    
    print(f"Dataset: {len(train_ds)} train, {len(val_ds)} val")
    
    # DataLoader
    train_loader = PyGDataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # Определяем реальные размерности фичей (без паддинга)
    # Нам нужно заглянуть в сырой файл, чтобы узнать сколько там было до dataset.py
    # Или просто знать константы Ecole.
    # Ecole NodeBipartite по умолчанию: Rows=5, Cols=19 (или около того).
    # Давайте возьмем из dataset (мы там знаем оригинальные dim_c и dim_v, 
    # но dataset.get() уже возвращает padded).
    
    # ХАК: Возьмем первый сырой файл и посмотрим
    raw_sample = torch.load(os.path.join("dataset_train", train_ds.files[0]))
    dim_c = raw_sample['row_features'].shape[1]
    dim_v = raw_sample['col_features'].shape[1]
    print(f"Detected Input Dims: Cons={dim_c}, Vars={dim_v}")
    
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
            
            # --- ХИТРЫЙ LOSS С БАТЧИНГОМ ---
            # 1. Зануляем (ставим -inf) все, что НЕ кандидат
            logits[~batch.cand_mask] = -1e9
            
            # 2. Считаем Softmax по ГРАФАМ (batch.batch)
            # Это превращает логиты в вероятности, суммирующиеся в 1 внутри каждого графа
            probs = softmax(logits, batch.batch)
            
            # 3. Нам нужна вероятность правильного класса (Target)
            # batch.y - это локальный индекс правильной переменной + сдвиг (из dataset.py)
            # Но при батчинге PyG просто конкатенирует атрибуты.
            # batch.y будет вектором [B].
            # Нам нужно найти глобальный индекс в батче, соответствующий target узлу.
            
            # В dataset.py мы сделали: y = label_var_idx + num_cons.
            # Это индекс внутри ОДНОГО графа.
            # При батчинге: global_index = ptr[graph_id] + y[graph_id]
            
            ptr = batch.ptr[:-1] # Индексы начал графов
            # batch.y имеет размер [B, 1] или [B]
            targets_local = batch.y.squeeze()
            
            global_targets = ptr + targets_local
            
            # Берем вероятности правильных ответов
            target_probs = probs[global_targets]
            
            # NLL Loss: -log(p)
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
                logits[~batch.cand_mask] = -1e9
                
                # Разбиваем обратно на графы для метрик
                # (Можно делать векторно, но циклом проще для понимания)
                
                # Получаем списки узлов для каждого графа
                # ptr: [0, N1, N1+N2, ...]
                ptr = batch.ptr.cpu().numpy()
                targets = batch.y.squeeze().cpu().numpy()
                
                logits_cpu = logits.cpu()
                
                for i in range(len(ptr) - 1):
                    start, end = ptr[i], ptr[i+1]
                    graph_logits = logits_cpu[start:end]
                    
                    # Target внутри графа
                    # В dataset.py y уже сдвинут на num_cons, так что он корректен для graph_logits
                    target = targets[i]
                    
                    # Top 1
                    if torch.argmax(graph_logits) == target:
                        val_acc1 += 1
                        
                    # Top 5
                    # Проверяем, есть ли хотя бы 5 кандидатов
                    k = min(5, len(graph_logits))
                    _, topk = torch.topk(graph_logits, k)
                    if target in topk:
                        val_acc5 += 1
                        
                    val_count += 1
        
        acc1 = val_acc1 / val_count if val_count else 0
        acc5 = val_acc5 / val_count if val_count else 0
        
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Val Acc@1: {acc1:.4f} | Val Acc@5: {acc5:.4f}")
        
        # Пишем в TensorBoard
        writer.add_scalar("Train/Loss", avg_loss, epoch)
        writer.add_scalar("Val/Acc_Top1", acc1, epoch)
        writer.add_scalar("Val/Acc_Top5", acc5, epoch)
        
        if acc1 > best_acc:
            best_acc = acc1
            torch.save(model.state_dict(), "best_model.pt")

    writer.close()
    print(f"Best Val Acc@1: {best_acc:.4f}")

if __name__ == "__main__":
    train()