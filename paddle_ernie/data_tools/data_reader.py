"""
Copyright (c) 2024 by huangqianfei@tju.edu.cn All Rights Reserved. 
Author: huangqianfei@tju.edu.cn
Date: 2024-08-31 19:22:36
Description: 
"""
import os
import sys
import random

import paddle
import subprocess
import collections
import numpy as np

from paddlenlp.transformers import ErnieTokenizer

CUR_DIR = os.path.abspath(os.path.dirname(__file__))


class Reader():
    label1 = ['class11', 'class12']
    label2 = ['class21', 'class22', 'class23']


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

    def __getitem__(self, index):
        if not self.deque:
            docs = self.reader.read(1024 * 1024 * 5).splitlines()
            docs = self.parse_data(docs)
            self.deque.extend(docs)

        item = self.deque.popleft()
        doc, label1, label2 = item
        doc = "fxbnlu" + doc

        label1_index = self.label1_index[label1]
        label2_index = self.label2_index[label2]
        return doc, label1_index, label2_index

    def parse_data(self, docs):
        """get data index"""
        docs = [x.split("\t") for x in docs]
        if self.shuffle:
            random.shuffle(docs)

        data = []
        for item in docs:
            if len(item) >= 3 and item[1] in self.label1_index and item[2] in self.label2_index:
                data.append(item)
    
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
            return_tensors="pd",
        )
        
        input_ids = srcs.input_ids
        labels1 = paddle.to_tensor(np.array(labels1))
        labels2 = paddle.to_tensor(np.array(labels2))
        
        return (input_ids, labels1, labels2)



if __name__ == "__main__":

    model_path = "/Users/huangqianfei/.paddlenlp/models/ernie-3.0-mini-zh"

    tokenizer = ErnieTokenizer.from_pretrained(model_path)
    line = "我爱北京天安门, fxbnlu unaffable"
    print(tokenizer.tokenize(line))
  

    train_path = CUR_DIR + "/../data/train_data.txt"
    data_set = Reader(
        train_path,
        tokenizer=tokenizer,
        max_token=64,
        shuffle=True,
    )


    dataloader = paddle.io.DataLoader(
        data_set,
        collate_fn=data_set.collate_fn,
        return_list=True,
        shuffle=True,
        batch_size=12,
        drop_last=True,
        num_workers=1,
    )

    for step, batch in enumerate(dataloader):
        (input_ids, labels1, labels2,) = batch

        shape = input_ids.shape
 
        for i in range(3):
            src = " ".join(tokenizer.convert_ids_to_tokens(input_ids[i]))
            print(src)

            line = tokenizer.decode(input_ids[i])
            words = [x if x != "[PAD]" else "" for x in line.split()]
            query = "".join(words).strip()
            print(query, "->", labels1[i].item(), "->", labels2[i].item())
        exit()
    