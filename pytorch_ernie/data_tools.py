"""
Copyright (c) 2024 by huangqianfei@tju.edu.cn All Rights Reserved. 
Author: huangqianfei@tju.edu.cn
Date: 2024-08-31 19:22:36
Description: 
"""
import os
import random

import torch
import subprocess
import collections
import numpy as np

from transformers import BertTokenizer, ErnieModel
from torch.utils.data import DataLoader


CUR_DIR = os.path.abspath(os.path.dirname(__file__))


class Reader():


    def __init__(self, train_path = "", max_token = 64, shuffle=False, tokenizer=None):

        self.label1_index = {}
        for index, item in enumerate(self.label1):
            self.label1_index[item] = index
        self.label2_index = {}
        for index, item in enumerate(self.label2):
            self.label2_index[item] = index
        self.shuffle = shuffle
        self.max_token = max_token
        self.tokenizer = tokenizer

        self.reader = os.popen(f"cat {train_path}")
        self.deque = collections.deque()
        self._lines = int(subprocess.getoutput(f"wc -l {train_path}").strip().split(" ")[0])
        print(self._lines)

    def __getitem__(self, index):
        if not self.deque:
            docs = self.reader.read(1024 * 1024 * 1).splitlines()
            docs = self.parse_data(docs)
            self.deque.extend(docs)
        
        item = self.deque.popleft()
        doc = item[0]
        label1 = item[1]
        label2_list = item[2:]
        doc = "fxbnlu" + doc

        label1_index = self.label1_index[label1]
        # label2_index = self.label2_index[label2]
        label2_flag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for label2 in label2_list:
            if label2 in self.label2_index:
                index_flag = self.label2_index[label2]
                label2_flag[index_flag] = 1
        
        return doc, label1_index, label2_flag

    def parse_data(self, docs):
        """get data index"""
        docs = [x.split("\t") for x in docs]
        if self.shuffle:
            random.shuffle(docs)

        data = []
        for item in docs:
            if len(item) >= 3 and item[1] in self.label1_index and item[2] in self.label2_index:
                data.append(item)
            else:
                data.append([item[0], self.label1[0], self.label2[0]])
    
        return data


    def __len__(self):
        return self._lines

    def collate_fn(self, batch):
        """pad batch"""
        doc_list = [item[0] for item in batch]
        labels1 = [item[1] for item in batch]
        labels2 = [item[2] for item in batch]

        max_len = max(len(inst) for inst in doc_list)
        max_len = min(max_len, self.max_token)

        srcs = self.tokenizer(
            doc_list,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        input_ids = srcs.input_ids
        attention_mask = srcs.attention_mask
        
        labels1 = torch.from_numpy(np.array(labels1))
        labels2 = torch.from_numpy(np.array(labels2)).float()

        return input_ids, labels1, labels2, attention_mask



if __name__ == "__main__":

    model_path = "/Users/huangqianfei/.transformer/ernie-3.0-mini-zh"

    tokenizer = BertTokenizer.from_pretrained(model_path)
    line = "我爱北京天安门, fxbnlu unaffable"
    list = tokenizer.tokenize(line)
    print(list)

    train_path = CUR_DIR + "/data/train_data.txt"
    data_set = Reader(
        train_path,
        tokenizer=tokenizer,
        max_token=50,
        shuffle=True,
    )


    dataloader = DataLoader(
        data_set,
        collate_fn=data_set.collate_fn,
        shuffle=True,
        batch_size=12,
    )
    print(len(dataloader))

    for step, batch in enumerate(dataloader):

        input_ids, labels1, labels2, attention_mask = batch
        shape = input_ids.shape
 
        for i in range(3):
            src = " ".join(tokenizer.convert_ids_to_tokens(input_ids[i]))
            print(src)

            line = tokenizer.decode(input_ids[i])
            words = [x if x != "[PAD]" else "" for x in line.split()]
            query = "".join(words).strip()
            print(query, "->", labels1[i].item(), "->", labels2[i].item())
        exit()
    