from data_loader import Data_set,DataLoader
from collections import deque
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = Data_set('data/train.csv', 'data/wiki_info_v2.json',is_test=False)
dataloader = DataLoader(dataset,batch_size=5, shuffle=False)
val_dataset = Data_set('data/val.csv', 'data/wiki_info_v2.json', is_test=False)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)
test_dataset = Data_set('data/test.csv', 'data/wiki_info_v2.json', is_test=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

model = torch.load('model/best_val_model_pytorch_2e-07')
# model = torch.load('model/best_val_model_pytorch_2e-07_train')
model.to(device)

result = np.zeros((1, 2))
counter = 0
freq = 50
# for x, y in val_loader:
#     counter += 1
#     y = y.view(1, -1).to(device)
#     pred = model(x)
#     pred_y = torch.argmax(pred, dim=1)
#     pred_y = pred_y.detach().cpu().numpy()[0]
#     true_ans = torch.argmax(y)
#     if counter % freq == 0:
#         print(f'pred: {pred_y}, true: {true_ans}, raw: {pred}')
#     if pred_y == true_ans:
#         result[0][0] += 1
#     else:
#         result[0][1] += 1
#
# print(result)
ans = deque()
for x, y in test_loader:
    pred = model(x)
    pred_y = torch.argmax(pred, dim=1)
    pred_y = pred_y.detach().cpu().numpy()[0]
    ans.append(pred_y)
    # print(f'pred: {pred_y}, raw: {pred}')

if len(ans) == len(test_loader):
    with open('res.txt', 'w') as f:
        for a in ans:
            f.write(f'{a}\n')
else:
    raise ValueError(f'output loss {len(test_loader) - len(ans)}')

