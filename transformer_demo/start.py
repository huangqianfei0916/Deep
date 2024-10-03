'''
Author: huangqianfei
Date: 2023-01-01 14:16:58
LastEditTime: 2023-01-14 15:45:52
Description: 
'''
import os
import sys

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(CUR_DIR + "/")

import torch
import time
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, ErnieModel
from pathlib import Path

from data_tools import Reader
from transformer.Models import Transformer
from transformer import Constants
from transformer.Optim import ScheduledOptim
from utils import cal_performance
from utils import save_model

gpu = torch.cuda.is_available()
print("GPU available: ", gpu)
print("CuDNN: ", torch.backends.cudnn.enabled)
print('GPUs：', torch.cuda.device_count())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
torch.cuda.manual_seed_all(2024)

class TransformerTask(object):
    """transformer task"""
    def __init__(self, model_path, save_dir):
        self.save_dir = save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        self.model = Transformer(
                n_src_vocab=39980, 
                n_trg_vocab=39980,
                src_pad_idx=0, 
                trg_pad_idx=0)

        # print("================================================")
        # for key, value in self.model.named_parameters():
        #     print(key)
        #     print(value.shape)
        # print("================================================")

        self.criterion = nn.CrossEntropyLoss()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

        freeze = False
        if freeze:
            model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        else:
            model_parameters = self.model.parameters()
        
        self.optimizer = ScheduledOptim(
                            torch.optim.Adam(model_parameters, betas=(0.9, 0.98), eps = 1e-09), 
                            lr_mul=2,
                            d_model=512, 
                            n_warmup_steps=100)


    def now(self):
        """get the current time"""
        return str(time.strftime('%Y-%m-%d %H:%M:%S'))

    def train_batch(self, item):
        """train batch"""
        text1_ids, text1_attention_mask, text2_ids, text2_attention_mask = item
        src_id = text1_ids.to(device)
        tgt_id = text2_ids.to(device)

        gold = tgt_id[:, 1:].contiguous().view(-1)
        tgt_id = tgt_id[:, :-1]

        # forward
        pred = self.model(src_id, tgt_id)
        loss, n_correct, n_word = cal_performance(pred, gold, 0, smoothing=True)
        
        return loss, n_correct, n_word

    def train(self, train_path, n_epoch=2):
        """train"""
        self.model = self.model.to(device)
        step = 0

        for epoch in range(n_epoch):
            self.model = self.model.train()
            
            data_set = Reader(
                train_path,
                tokenizer=self.tokenizer,
                max_token=80,
                shuffle=True,
            )

            dataloader = DataLoader(
                data_set,
                collate_fn=data_set.collate_fn,
                shuffle=True,
                batch_size=64,
            )

            for i, item in enumerate(dataloader):
                loss, n_correct, n_word = self.train_batch(item)
                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step_and_update_lr()
                step += 1

                train_loss = loss.item()

                if step % 20 == 0:
                    print("step: %d, Train Loss: %.4f, n_correct: %d, n_word: %d" % (step, train_loss, n_correct, n_word))
                    model_path =  self.save_dir + f"/model-{step}/"
                    save_model(self.model, self.tokenizer, self.optimizer, step, model_path)
            
            print("step: %d, Train Loss: %.4f, n_correct: %d, n_word: %d" % (step, train_loss, n_correct, n_word))
            model_path =  self.save_dir + f"/model-{step}/"
            save_model(self.model, self.tokenizer, self.optimizer, step, model_path)



if __name__ == '__main__':
    """main"""
    model_path = "/Users/huangqianfei01/Desktop/learn/learn_nlp/transformer_demo/tokenizer/"
    train_path = CUR_DIR + "/data/part_query.txt"
    task = TransformerTask(
        model_path,
        save_dir="model_checkpoint"
        )
    
    task.train(train_path, n_epoch=2)
