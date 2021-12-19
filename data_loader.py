import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from config import Config


class Data_set(Dataset):
    def __init__(self, filename, wiki_filename, config: Config, is_test=False):
        self.riddle_list = []  # list of riddle
        self.choice_list = []  # list of ([cand0], [cand1], ... [cand4])
        self.y_list = []
        self.is_test = is_test
        self.wiki_data = {}
        self.error_line = 0  # csv错误行
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.riddle_max_len = config.riddle_max_len
        self.choice_max_len = config.choice_max_len

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
                    # 正确的答案是term[label+1]
                    label = int(terms[6])

                # x_list的数据格式为：（谜语，提示，候选项，候选项的wiki解释）
                self.riddle_list.append([riddle_, tip])
                choices = [[terms[i], self.wiki_data.get(terms[i], 0)] for i in range(1, 6)]
                self.choice_list.append(choices)
                self.y_list.append(label)  # 0 ~ 4

    def __len__(self):
        return len(self.riddle_list)

    def __getitem__(self, item):
        riddle, tip = self.riddle_list[item]
        choices = self.choice_list[item]
        label = None
        if not self.is_test:
            label = self.y_list[item]

        # --------------------------------------------------------------------------------------------
        # 重点调整区域
        # 这里可以调整模型输入，目前的设置是，提示拼接谜语作为bert句子对的第一个输入，候选项拼接候选项的解释作为第二个输入
        # padding的参数也可以调节
        # --------------------------------------------------------------------------------------------
        riddle_input = self.tokenizer.encode_plus(tip, riddle, add_special_tokens=True,
                                                  padding='max_length', max_length=self.riddle_max_len,
                                                  truncation='only_second')
        riddle_input_ids = torch.tensor(riddle_input['input_ids']).cuda()
        riddle_attention_mask = torch.tensor(riddle_input['attention_mask']).cuda()

        choices_input_ids = []
        choices_attention_mask = []
        for choice in choices:
            choice_text, choice_wiki = choice
            choice_input = self.tokenizer.encode_plus(choice_text, choice_wiki, add_special_tokens=True,
                                                      padding='max_length', max_length=self.choice_max_len,
                                                      truncation='only_second')
            choices_input_ids.append(choice_input['input_ids'])
            choices_attention_mask.append(choice_input['attention_mask'])

        choices_input_ids = torch.tensor(choices_input_ids).cuda()
        choices_attention_mask = torch.tensor(choices_attention_mask).cuda()

        return riddle_input_ids, riddle_attention_mask, choices_input_ids, choices_attention_mask, label
