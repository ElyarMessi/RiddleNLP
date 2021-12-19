from collections import deque
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import json

from typing import Tuple, List


class Data_set(Dataset):
    def __init__(self, filename, wiki_filename, is_test=False):
        self.x_list = deque()
        self.y_list = deque()
        self.is_test = is_test
        self.wiki_data = {}
        self.error_line = 0  # csv错误行
        self.tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext')
        self.n_positive = 0
        self.n_negative = 0

        with open(wiki_filename, 'r', encoding='utf-8') as file:
            self.wiki_data = json.load(file)

        with open(filename, 'r', encoding='utf-8') as file:
            for line in file.readlines()[1:]:
                # 剔除有错误的行 即逗号多一个少一个
                if not is_test and line.count(',') != 6:
                    self.error_line += 1
                    continue
                if is_test and line.count(',') != 5:
                    self.error_line += 1
                    continue

                terms = line.split(',')
                riddle_string = terms[0]
                l_ = riddle_string.find('（')
                r_ = riddle_string.find('）')
                # 谜语
                riddle_ = riddle_string[:l_ - 1]
                # 提示（打一···）
                tip = riddle_string[l_ + 1:r_]
                if not is_test:
                    label = int(terms[6])

                # --------------------------------------------------------------------------------------------
                # x_list的数据格式为：（谜语，提示，候选项，候选项的wiki解释）
                # 二分类
                # --------------------------------------------------------------------------------------------
        #         for i in range(1, 6):
        #             self.x_list.append([riddle_, tip, terms[i], self.wiki_data.get(terms[i], 0)])
        #             if not is_test:
        #                 if i - 1 == label:
        #                     self.y_list.append(1)
        #                     self.n_positive += 1
        #                 else:
        #                     self.y_list.append(0)
        #                     self.n_negative += 1
        #
        # if self.n_negative == self.n_positive:
        #     self.is_unbalance = True
        # else:
        #     self.is_unbalance = False

                # --------------------------------------------------------------------------------------------
                # x_list的数据格式为：（谜语，提示，候选项，候选项的wiki解释）
                # 多分类?
                # --------------------------------------------------------------------------------------------
                sample = []
                sample_ans = []
                for i in range(1, 6):
                    bert_input = self.tokenizer.encode_plus(tip + riddle_, terms[i] + self.wiki_data.get(terms[i], 0),
                                                            add_special_tokens=True,
                                                            padding='max_length',
                                                            max_length=256,
                                                            truncation='only_second')

                    input_ids = torch.tensor(bert_input['input_ids'])
                    token_type_ids = torch.tensor(bert_input['token_type_ids'])
                    attention_mask = torch.tensor(bert_input['attention_mask'])
                    sample.append([input_ids, token_type_ids, attention_mask])
                    if not is_test:
                        if i - 1 == label:
                            sample_ans.append(1)
                            self.n_positive += 1
                        else:
                            sample_ans.append(0)
                            self.n_negative += 1
                sample_ans = torch.FloatTensor(sample_ans)
                self.x_list.append(sample)
                self.y_list.append(sample_ans)

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, item) -> Tuple[Tuple, List]:
        # if self.is_test:
        #     return self.x_list[item], None
        # print(type(self.x_list[item]))
        riddle, tip, ans, ans_wiki = self.x_list[item]
        # X = self.x_list[item]  # input_ids, token_type_ids, attention_mask
        bert_input = self.tokenizer.encode_plus(tip + riddle, ans + ans_wiki,
                                                    add_special_tokens=True,
                                                    padding='max_length',
                                                    max_length=256,
                                                    truncation='only_second')

        input_ids = torch.tensor(bert_input['input_ids'])
        token_type_ids = torch.tensor(bert_input['token_type_ids'])
        attention_mask = torch.tensor(bert_input['attention_mask'])
        # print(riddle)
        # print(tip)
        if not self.is_test:
            label = self.y_list[item]
        else:
            label = None

        # --------------------------------------------------------------------------------------------
        # 重点调整区域
        # 这里可以调整模型输入，目前的设置是，提示拼接谜语作为bert句子对的第一个输入，候选项拼接候选项的解释作为第二个输入
        # padding的参数也可以调节
        # --------------------------------------------------------------------------------------------

        return (input_ids, token_type_ids, attention_mask), label

# dataset = Data_set('data/train.csv', 'data/wiki_info_v2.json',is_test=False)
# print(dataset.error_line)
# dataloader = DataLoader(dataset,batch_size=4)
#
# for id, (x,y) in enumerate(dataloader):
#     for i in range(3):
#         print(x[i])
#     print(y)
#     if id >= 10:
#         break

# dataset = Data_set('data/test.csv', 'data/wiki_info_v2.json',is_test=True)
# print(dataset.error_line)
# dataloader = DataLoader(dataset)
#
# for id, (x,y) in enumerate(dataloader):
#     print(x)
#     #print(y)
#     if id >= 10:
#         break
