import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax
from dataset import MIAPDataset
from model import GasseGCN

def sanity_check():
    print("--- SANITY CHECK: Overfitting on 1 batch ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Берем маленький кусочек данных (первые 8 примеров)
    full_ds = MIAPDataset("dataset_train")
    small_ds = [full_ds[i] for i in range(8)] 
    
    # 2. Создаем лоадер
    loader = DataLoader(small_ds, batch_size=8, shuffle=False) # Весь батч сразу
    
    # 3. Инициализируем модель
    sample = small_ds[0]
    # Учитываем, что последняя колонка - это тип узла, мы ее не подаем в embedding
    # В model.py мы это уже обрабатываем
    dim_c = 5 # Примерно (зависит от твоих данных)
    dim_v = sample.x.shape[1] - 1 - dim_c # Вычисляем остаток
    # Проще взять из sample и довериться модели, она сама режет по типу
    
    # ВАЖНО: В model.py мы хардкодили dim_c/dim_v в конструкторе?
    # Давай проверим, как мы инициализируем.
    # Лучше передать реальные размеры.
    
    # В твоем dataset.py x имеет размер [N, max_dim + 1].
    # Мы должны передать это корректно.
    
    input_dim = sample.x.shape[1] # Это full width
    
    # Инициализируем модель
    # ВНИМАНИЕ: Нужно убедиться, что dimensions в GasseGCN совпадают с данными
    # В dataset.py мы делали паддинг.
    # Сейчас сделаем универсально:
    model = GasseGCN(dim_cons=input_dim-1, dim_vars=input_dim-1, hidden_dim=128).to(device)
    
    # 4. Оверфиттим!
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Агрессивный LR
    
    print(f"Target indices: {[d.y.item() for d in small_ds]}")
    
    for epoch in range(200): # 200 эпох долбим одни и те же 8 примеров
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            logits = model(batch)
            
            # Маскирование
            logits[~batch.cand_mask] = -1e9
            
            # Loss
            ptr = batch.ptr[:-1]
            targets_local = batch.y.squeeze()
            global_targets = ptr + targets_local
            
            probs = softmax(logits, batch.batch)
            target_probs = probs[global_targets]
            loss = -torch.log(target_probs + 1e-9).mean()
            
            loss.backward()
            optimizer.step()
            
            # Acc
            pred = logits.argmax() # Это неправильно для батча, но для лога пойдет
            # Правильный подсчет Acc@1 для батча
            acc = 0
            logits_cpu = logits.detach().cpu()
            ptr_cpu = batch.ptr.cpu()
            targets_cpu = targets_local.cpu()
            
            for i in range(len(small_ds)):
                start, end = ptr_cpu[i], ptr_cpu[i+1]
                graph_logits = logits_cpu[start:end]
                if graph_logits.argmax() == targets_cpu[i]:
                    acc += 1
            acc_pct = acc / len(small_ds)
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.6f} | Acc {acc_pct:.2f}")
            
        if acc_pct == 1.0 and loss.item() < 0.01:
            print("\n>>> SANITY CHECK PASSED! Model can memorize data. <<<")
            return

    print("\n>>> SANITY CHECK FAILED! Model cannot learn even 8 examples. <<<")
    print("Likely causes: Broken Gradients, Bad Architecture, or Dead Features.")

if __name__ == "__main__":
    sanity_check()