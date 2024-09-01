"""
Copyright (c) 2024 by huangqianfei@tju.edu.cn All Rights Reserved. 
Author: huangqianfei@tju.edu.cn
Date: 2024-08-31 16:51:42
Description: 
"""
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import torch.nn as nn

import torch.nn.functional as F


checkpoint = "/Users/huangqianfei/.transformer/ernie-3.0-mini-zh"
# 文本输入
raw_inputs = [
    "I love you",
    "I love you so much",
]

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(raw_inputs, max_length=8, padding=True, truncation=True, return_tensors="pt")

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
print(model.config.id2label)
# 输出 {0: 'NEGATIVE', 1: 'POSITIVE'} 表示越靠近0是负面的，越靠近1是正向的
# 调用模型输出
outputs = model(**inputs)
print(outputs.logits.shape)
## 将模型输出的预测值logits给到torch,返回我们可以看懂的数据
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
# 输出 tensor([[1.3436e-04, 9.9987e-01],[1.3085e-04, 9.9987e-01]], grad_fn=<SoftmaxBackward0>)
# 第一句话，NEGATIVE=0.0001, POSITIVE=0.99 所以第一句话大概率是正向的


