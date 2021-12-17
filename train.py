from RiddleGuess.model import RiddleModel
from RiddleGuess.data_loader import Data_set,DataLoader
from tqdm import tqdm

dataset = Data_set('data/train.csv', 'data/wiki_info_v2.json',is_test=False)
print(dataset.error_line)
dataloader = DataLoader(dataset,batch_size=4)

model = RiddleModel()

Epoch_num = 5
for epoch in range(Epoch_num):
    print("Epoch:",epoch)
    for id, (x, label) in tqdm(enumerate(dataloader),desc="Training Process:"):
        y = model(x)
        # print(y)
        # print(label)

        # 计算loss 样本不平衡这里可以调loss
        # loss = ...

        # 更新网络
        # optim.step()
    # 保存模型