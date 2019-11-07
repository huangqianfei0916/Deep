import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pickle
from sklearn.model_selection import train_test_split
from LSTM.config.config import DefaultConfig

config = DefaultConfig()

fix_len = config.fix_length

word2id = config.word2id

"""加载word2index的dict"""
word2index = open(word2id, 'rb')
word_index_dict = pickle.load(word2index)
print(word_index_dict)

# ======================================================================================================================
# 返回该词的index，将每条记录转化成由index组成的list，判断其长度不足的补0
# ======================================================================================================================
def word2index(word):
    """将一个word转换成index"""
    if word in word_index_dict:
        return word_index_dict[word]
    else:
        return 0


def sentence2index(sentence):
    """将一个句子转换成index的list，并截断或补零"""
    word_list = sentence.strip().split()
    index_list = list(map(word2index, word_list))
    len_sen = len(index_list)
    if len_sen < fix_len:
        index_list = index_list + [0] * (fix_len - len_sen)
    else:
        index_list = index_list[:fix_len]
    return index_list

# ======================================================================================================================
# 划分数据集
# ======================================================================================================================
def get_data():
    f = open(config.data_path)
    documents = f.readlines()
    sentence = []
    for words in documents:
        s = sentence2index(words)
        sentence.append(s)

    x = np.array(sentence)

    """取出标签"""
    y = [0] * config.pos + [1] * config.neg
    y = np.array(y)

    train_x, val_x, train_y, val_y = train_test_split(
        x, y, test_size=0.1, random_state=0)

    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

    train_loader = DataLoader(train_data, shuffle=False, batch_size=config.batch_size)
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=config.batch_size)

    return train_loader, valid_loader

# ======================================================================================================================
# 划分数据集2
# ======================================================================================================================
def get_data2():
    f = open(config.data_path2)
    documents = f.readlines()
    sentence = []
    for words in documents:
        s = sentence2index(words)
        sentence.append(s)

    x = np.array(sentence)

    """取出标签"""
    y = [0] * config.pos + [1] * config.neg
    y = np.array(y)

    l = []
    for i in range(len(y)):
        l.append((x[i], y[i]))

    train_dataset, test_dataset = torch.utils.data.random_split(l, [config.total * 0.9, config.total * 0.1])
    train_data = DataLoader(train_dataset, config.batch_size, False)
    test_data = DataLoader(test_dataset, config.batch_size, False)

    return train_data, test_data
