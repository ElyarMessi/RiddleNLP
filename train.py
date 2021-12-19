import os
from model import RiddleModel
from data_loader import Data_set, DataLoader
# from loss import UBLoss
from tqdm import tqdm
from datetime import datetime
from collections import deque

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_head = 'model'
os.makedirs(model_head, exist_ok=True)
save_to = os.path.join(model_head, f'best_val_model_pytorch')
dataset = Data_set('data/train.csv', 'data/wiki_info_v2.json', is_test=False)
val_dataset = Data_set('data/val.csv', 'data/wiki_info_v2.json', is_test=False)
# criterion = UBLoss(n_positive=dataset.n_positive, n_negative=dataset.n_negative)
criterion = nn.CrossEntropyLoss()
print(dataset.error_line)
dataloader = DataLoader(dataset, batch_size=1)  # x只输出一个，batch_size不知道为什么不影响x；y正常
val_loader = DataLoader(val_dataset, batch_size=1)

model = RiddleModel()
model.to(device)

lr = 0.002
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
Epoch_num = 5

train_losses = np.zeros(Epoch_num)
test_losses = np.zeros(Epoch_num)
best_test_loss = np.inf
best_test_epoch = 0
for epoch in range(Epoch_num):
    model.train()
    print("Epoch:", epoch)
    t0 = datetime.now()
    train_loss = deque()
    for id, (x, label) in tqdm(enumerate(dataloader), desc="Training Process:"):
        label = label.to(device)
        y = model(x)

        optimizer.zero_grad()
        # print(f'pred: {torch.argmax(y)}')
        # print(f'true: {torch.argmax(label)}')

        # 计算loss 样本不平衡这里可以调loss
        loss = criterion(y, label)
        loss.backward()

        # 更新网络
        optimizer.step()
        # optim.step()
        train_loss.append(loss.item())
    train_loss = np.mean(train_loss)
    # 保存模型
    model.eval()
    test_loss = deque()
    for id, (x, label) in tqdm(enumerate(val_loader), desc="Val: "):
        label = label.to(device)
        y = model(x)
        loss = criterion(y, label)
        test_loss.append(loss.item())
    test_loss = np.mean(test_loss)

    train_losses[epoch] = train_loss
    test_losses[epoch] = test_loss

    if test_loss < best_test_loss:
        torch.save(model, save_to)
        best_test_loss = test_loss
        best_test_epoch = epoch
        print('model saved')

    dt = datetime.now() - t0
    print(f'Epoch {epoch + 1}/{Epoch_num}, Train Loss: {train_loss:.4f}, \
      Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')

torch.save(model, f'{save_to}_last_epoch')
