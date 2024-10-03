"""
Copyright (c) 2024 by huangqianfei@tju.edu.cn All Rights Reserved. 
Author: huangqianfei@tju.edu.cn
Date: 2024-08-31 19:22:36
Description: 
"""
import os
import random

import torch
import json
import subprocess
import collections
import numpy as np

from transformers import BertTokenizer, ErnieModel
from torch.utils.data import DataLoader


CUR_DIR = os.path.abspath(os.path.dirname(__file__))


class Reader():

    def __init__(self, train_path = "", max_token = 64, shuffle=False, tokenizer=None):

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
        text1 =  item[0]
        text2 = item[1]

        return text1, text2

    def parse_data(self, docs):
        """get data index"""

        data = []
        for line in docs:
            line = line.strip()
            try:
                json_res = json.loads(line)
            except:
                continue
            
            text1 = json_res["text1"]
            text2 = json_res["text2"]
            
            data.append([text1, text2])

        if self.shuffle:
            random.shuffle(data)
        
        return data


    def __len__(self):
        return self._lines

    def collate_fn(self, batch):
        """pad batch"""
        text1_list = [item[0] for item in batch]
        text2_list = [item[1] for item in batch]

        max_len1 = max(len(inst) for inst in text1_list)
        max_len2 = max(len(inst) for inst in text2_list)
        max_len = max(max_len1, max_len2) + 2

        max_len = min(max_len, self.max_token)

        text1_srcs = self.tokenizer(
            text1_list,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        text1_ids = text1_srcs.input_ids
        text1_attention_mask = text1_srcs.attention_mask

        text2_srcs = self.tokenizer(
            text2_list,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        text2_ids = text2_srcs.input_ids
        text2_attention_mask = text2_srcs.attention_mask

        return text1_ids, text1_attention_mask, text2_ids, text2_attention_mask



if __name__ == "__main__":

    model_path = "/Users/huangqianfei01/Desktop/learn/learn_nlp/transformer_demo/tokenizer/"

    tokenizer = BertTokenizer.from_pretrained(model_path)
    line = "我爱北京天安门, fxbnlu unaffable"
    list = tokenizer.tokenize(line, max_length=10, padding='max_length')
    print(list)

    train_path = CUR_DIR + "/data/part_query.txt"
    data_set = Reader(
        train_path,
        tokenizer=tokenizer,
        max_token=100,
        shuffle=True,
    )

    dataloader = DataLoader(
        data_set,
        collate_fn=data_set.collate_fn,
        shuffle=True,
        batch_size=1,
    )
    print(len(dataloader))

    for step, batch in enumerate(dataloader):

        text1_ids, text1_attention_mask, text2_ids, text2_attention_mask = batch
        print(text1_ids[0])
        src = "".join(tokenizer.convert_ids_to_tokens(text1_ids[0]))
        print(src)
        dst = "".join(tokenizer.convert_ids_to_tokens(text2_ids[0]))
        print(dst)
        line = tokenizer.decode(text1_ids[0])
        print(line)
        exit()
    