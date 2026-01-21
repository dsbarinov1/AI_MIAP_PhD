import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax
from dataset import MIAPDataset
from model import GasseHeteroGCN

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = MIAPDataset("dataset_train")
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    # Определение размерностей
    sample = train_ds[0]
    dim_v = sample['variable'].x.shape[1]
    dim_c = sample['constraint'].x.shape[1]
    
    model = GasseHeteroGCN(dim_c, dim_v).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            logits = model(batch) # Только для переменных
            
            # 1. Маскируем только кандидатов
            mask = batch['variable'].cand_mask
            logits[~mask] = -1e9
            
            # 2. Softmax по ГРАФАМ внутри узлов 'variable'
            probs = softmax(logits, batch['variable'].batch)
            
            # 3. Находим глобальные индексы таргетов
            # batch['variable'].ptr указывает на начало каждого графа в списке переменных
            ptr = batch['variable'].ptr[:-1]
            targets = batch['variable'].y + ptr
            
            target_probs = probs[targets]
            loss = -torch.log(target_probs + 1e-9).mean()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")

if __name__ == "__main__":
    train()