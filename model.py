import torch
import torch.nn as nn
from transformers import BertModel

model_name = 'chinese-bert-wwm-ext'


class RiddleModel(nn.Module):
    def __init__(self):
        super(RiddleModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.linear1 = nn.Linear(768,128)
        self.linear2 = nn.Linear(128,2)

    def forward(self,x):
        input_ids, token_type_ids, attention_mask = x
        hidden_state, pooled_output = self.bert(input_ids=input_ids,attention_mask=attention_mask,
                                                token_type_ids=token_type_ids,return_dict=False)
        # 这里需要考虑是直接用pooled_output还是hidden_state
        # 目前这里先用pooled_output了
        output = self.linear1(pooled_output)
        output = self.linear2(output)
        output = torch.sigmoid(output)

        return output