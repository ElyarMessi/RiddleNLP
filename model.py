import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel
from config import Config

model_name = 'hfl/chinese-bert-wwm-ext'


class RiddleModel(nn.Module):
    def __init__(self, config: Config):
        super(RiddleModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.linear1 = nn.Linear(768,128)
        self.linear2 = nn.Linear(128,2)

    def forward(self,x):
        input_ids, token_type_ids, attention_mask = x
        hidden_state, pooled_output = self.bert(input_ids=input_ids,attention_mask=attention_mask,
                                                token_type_ids=token_type_ids,return_dict=False)
        # 这里需要考虑是直接用pooled_output还是hidden_state
        # 目前这里先用pooled_output了
        output = self.dropout(pooled_output)
        output = self.linear1(output)
        output = self.linear2(output)
        output = F.softmax(output, dim=1)
        return output


def init_network(model, initializer_range=0.02):
    for n, p in list(model.named_parameters()):
        if 'bert' not in n:
            p.data.normal_(0, initializer_range)
