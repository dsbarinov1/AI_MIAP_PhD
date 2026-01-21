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
import random

def get_args():
    parser = argparse.ArgumentParser(description="Train Gasse GCN for MIAP")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_dir", type=str, default="dataset_train")
    parser.add_argument("--val_dir", type=str, default="dataset_val")
    parser.add_argument("--save_path", type=str, default="best_model.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, loader, device):
    """
    Функция валидации. Считает точность на всем датасете.
    """
    model.eval()
    val_acc1 = 0
    val_acc5 = 0
    val_count = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Если батч пустой или битый
            if batch.x.shape[0] == 0: continue

            logits = model(batch)
            
            # Маскируем невалидных кандидатов
            logits[~batch.cand_mask] = -1e9
            
            # Переносим на CPU для подсчета метрик
            ptr = batch.ptr.cpu().numpy()
            # flatten() гарантирует 1D массив, даже если батч=1
            targets = batch.y.view(-1).cpu().numpy() 
            logits_cpu = logits.cpu()
            
            # Итерируемся по графам в батче
            # len(ptr) - 1 = количество графов
            for i in range(len(ptr) - 1):
                start, end = ptr[i], ptr[i+1]
                graph_logits = logits_cpu[start:end]
                
                # Если граф пустой или что-то пошло не так
                if graph_logits.shape[0] == 0: continue

                target = targets[i]
                
                # Top 1 Accuracy
                if torch.argmax(graph_logits) == target:
                    val_acc1 += 1
                    
                # Top 5 Accuracy
                k = min(5, len(graph_logits))
                _, topk = torch.topk(graph_logits, k)
                if target in topk:
                    val_acc5 += 1
                    
                val_count += 1
    
    # Считаем среднее
    acc1 = val_acc1 / val_count if val_count > 0 else 0.0
    acc5 = val_acc5 / val_count if val_count > 0 else 0.0
    
    return acc1, acc5

def train():
    args = get_args()
    set_seed(args.seed)
    
    print(f"--- Training on {args.device} (BS={args.batch_size}, LR={args.lr}) ---")
    
    # Инициализация TensorBoard
    writer = SummaryWriter("runs/miap_experiment")
    
    # Загрузка данных
    train_ds = MIAPDataset(args.train_dir)
    val_ds = MIAPDataset(args.val_dir)
    
    print(f"Dataset: {len(train_ds)} train, {len(val_ds)} val")
    
    # DataLoader'ы
    train_loader = PyGDataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # Определение размерности признаков
    # Берем первый сэмпл. 
    # В model.py мы ожидаем, что последняя колонка - это тип узла.
    # Значит feature_width = total_width - 1
    sample = train_ds[0]
    feature_width = sample.x.shape[1] - 1
    print(f"Feature width detected: {feature_width}")
    
    # Инициализация модели
    # dim_cons и dim_vars одинаковые, так как мы сделали паддинг в dataset.py
    model = GasseGCN(dim_cons=feature_width, dim_vars=feature_width, hidden_dim=args.hidden).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Scheduler: уменьшает LR, если валидация не растет
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0
        steps = 0
        
        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()
            
            if batch.x.shape[0] == 0: continue
            
            logits = model(batch)
            
            # --- Маскирование ---
            logits[~batch.cand_mask] = -1e9
            
            # --- Расчет Loss ---
            # 1. Softmax по каждому графу отдельно
            probs = softmax(logits, batch.batch)
            
            # 2. Находим глобальные индексы правильных ответов
            ptr = batch.ptr[:-1] 
            targets_local = batch.y.view(-1) # view(-1) безопасен для BS=1
            
            # Сдвигаем локальные индексы таргетов на начало графа в батче
            global_targets = ptr + targets_local
            
            # 3. Берем вероятности правильных ответов
            target_probs = probs[global_targets]
            
            # 4. NLL Loss (-log p)
            loss = -torch.log(target_probs + 1e-9).mean()
            
            loss.backward()
            
            # Gradient Clipping (спасает от взрывов)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            loss_sum += loss.item()
            steps += 1
            
        avg_loss = loss_sum / steps if steps > 0 else 0.0
        
        # --- Валидация ---
        val_acc1, val_acc5 = evaluate(model, val_loader, args.device)
        
        # Шаг шедулера
        scheduler.step(val_acc1)
        
        # Логирование
        lr_curr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | Loss {avg_loss:.4f} | Val Acc@1: {val_acc1:.4f} | Val Acc@5: {val_acc5:.4f} | LR {lr_curr:.1e}")
        
        # TensorBoard
        writer.add_scalar("Train/Loss", avg_loss, epoch)
        writer.add_scalar("Val/Acc_Top1", val_acc1, epoch)
        writer.add_scalar("Val/Acc_Top5", val_acc5, epoch)
        
        # Сохранение лучшей
        if val_acc1 > best_acc:
            best_acc = val_acc1
            torch.save(model.state_dict(), args.save_path)
            print(f"  >>> New best model saved (Acc: {best_acc:.4f})")

    writer.close()
    print(f"Training finished. Best Val Acc@1: {best_acc:.4f}")

if __name__ == "__main__":
    train()