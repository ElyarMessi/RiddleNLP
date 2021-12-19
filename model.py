import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel
from config import Config

model_name = 'hfl/chinese-bert-wwm-ext'


class RiddleModel(nn.Module):
    def __init__(self, config: Config):
        super(RiddleModel, self).__init__()
        self.choice_num = config.choice_num
        self.bert = BertModel.from_pretrained(model_name)
        self.riddle_dropout = nn.Dropout(config.dropout)
        self.choice_dropout = nn.Dropout(config.dropout)
        self.riddle_linear = nn.Linear(768, config.hidden_size)  # for riddle
        self.choice_linear = nn.Linear(768, config.hidden_size)  # for choice

    def forward(self, riddle, riddle_attention, choices, choices_attention):
        """
        [riddle, tip] <-> [choice1, wiki], ..., [choice4, wiki]
        :param riddle: [batch, riddle_len]
        :param riddle_attention: [batch, riddle_len]
        :param choices: [batch, 5, choice_len]
        :param choices_attention: [batch, 5, choice_len]
        :return:
        """
        batch_size = riddle.size(0)

        _, riddle_pooled_output = self.bert(input_ids=riddle,
                                            attention_mask=riddle_attention,
                                            return_dict=False)
        riddle_encoded = self.riddle_dropout(riddle_pooled_output)  # [batch, hidden]

        choices_flatten = choices.flatten(0, 1)  # [batch * 5, choice_len]
        choices_attention_flatten = choices_attention.flatten(0, 1)  # [batch * 5, choice_len]
        _, choice_pooled_output = self.bert(input_ids=choices_flatten,
                                            attention_mask=choices_attention_flatten,
                                            return_dict=False)
        choices_encoded = choice_pooled_output.reshape(batch_size, self.choice_num, -1)  # [batch, 5, hidden]
        choices_encoded = self.choice_dropout(choices_encoded)

        riddle_encoded = self.riddle_linear(riddle_encoded)  # [batch, riddle_linear_size]
        choices_encoded = self.choice_linear(choices_encoded)  # [batch, 5, choice_linear_size]
        logits = torch.bmm(choices_encoded, riddle_encoded.unsqueeze(2))
        logits = logits.squeeze()
        probs = F.softmax(logits, dim=1)
        return probs


def init_network(model, initializer_range=0.02):
    for n, p in list(model.named_parameters()):
        if 'bert' not in n:
            p.data.normal_(0, initializer_range)
