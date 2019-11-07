import torch.nn as nn
import torch.nn.functional as F
import torch

# ======================================================================================================================
# 定义网络模型
# ======================================================================================================================
class lstmnet(nn.Module):

    def __init__(self, emb_weights, vocab_size, dim, config):
        super(lstmnet, self).__init__()

        self.config = config

        # embedding and LSTM layers
        self.embedding = nn.Embedding.from_pretrained(embeddings=emb_weights, freeze=config.emb_freeze)

        self.lstm = nn.LSTM(input_size=config.input_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            batch_first=config.batch_first,
                            bidirectional=config.bidir)
        for p in self.parameters():
            p.requires_grad=False

        # dropout layer
        self.dropout = nn.Dropout(config.dropout)

        if config.bidir:
            self.l1 = nn.Linear(config.hidden_size * 2, config.num_classes)

        else:
            self.l1 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):

        x = x.long()

        out = self.embedding(x)
        out, _ = self.lstm(out)

        out = out[:, -1, :]
        out=self.dropout(out)
        out = self.l1(out)

        return out
