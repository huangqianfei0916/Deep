"""
Copyright (c) 2024 by huangqianfei@tju.edu.cn All Rights Reserved. 
Author: huangqianfei@tju.edu.cn
Date: 2024-08-31 19:25:13
Description: 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, ErnieModel, PreTrainedModel



class ClassificationHead(nn.Module):
    """
    Perform sentence-level classification tasks.
    """

    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states):

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = F.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class ErnieEncode(nn.Module):

    def __init__(
        self,
        ernie,
        num_classes1,
        num_classes2,
        dropout=0.1,
    ):
        super(ErnieEncode, self).__init__()
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.ernie = ernie
        self.dropout = nn.Dropout(
            dropout if dropout is not None else self.ernie.config.hidden_dropout_prob
        )
        # print(self.ernie.config.hidden_size)
        self.seq_classifier1 = ClassificationHead(
            self.ernie.config.hidden_size,
            self.ernie.config.hidden_size,
            num_classes1,
            dropout,
        )
        self.seq_classifier2 = ClassificationHead(
            self.ernie.config.hidden_size,
            self.ernie.config.hidden_size,
            num_classes2,
            dropout,
        )

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):

        outputs = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        first_token_tensor = outputs[0][:, 0]
        second_token_tensor = outputs[0][:, 1]

        logits1 = self.seq_classifier1(first_token_tensor)
        logits2 = self.seq_classifier2(second_token_tensor)
        return logits1, logits2