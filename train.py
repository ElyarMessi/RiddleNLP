import time
import numpy as np
from pathlib import Path
from datetime import timedelta
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from tqdm import tqdm

from model import RiddleModel, init_network
from data_loader import Data_set, DataLoader
from config import Config


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train(config: Config):
    start_time = time.time()
    train_dataset = Data_set('data/train.csv', 'data/wiki_info_v2.json', is_test=False)
    val_dataset = Data_set('data/val.csv', 'data/wiki_info_v2.json', is_test=False)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=config.batch_size)
    val_loader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=config.batch_size)
    print(train_dataset.error_line)

    # model init
    model = RiddleModel(config)
    init_network(model)
    model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_f = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 4]).cuda())

    # iter setting
    total_batch = 1
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    # log
    log_path = Path(config.log_dir)
    log_path.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=config.log_dir + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    for epoch in range(config.epoch_num):
        print('Epoch [{}/{}]'.format(epoch + 1, config.epoch_num))
        # scheduler.step() # 学习率衰减
        for i, (x, labels) in tqdm(enumerate(train_loader), desc="Training Process"):
            probs = model(x)
            model.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            loss = loss_f(probs, labels)
            loss.backward()
            optimizer.step()
            # 以下计算 valid data 上的效果
            if total_batch % config.report_step == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predict = torch.max(probs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(config, model, val_loader)
                improve = ''
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.model_path)
                    improve = '*'
                    last_improve = total_batch
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.3},  Train Acc: {2:>6.3%},  Val Loss: {3:>5.3},  Val Acc: {4:>6.3%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集 loss 超过 require_improvement batch 没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()


def evaluate(config: Config, model, data_iter, test=False):
    """
    测试效果
    :param config:
    :param model:
    :param data_iter:
    :param test: 汇报混淆矩阵
    :return:
    """
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    loss_f = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, labels in data_iter:
            probs = model(x)
            labels = torch.LongTensor(labels).cuda()
            loss = loss_f(probs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(probs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.labels, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


if __name__ == '__main__':
    config = Config()
    train(config)
