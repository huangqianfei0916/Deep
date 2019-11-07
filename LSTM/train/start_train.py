import sys

sys.path.extend(["../../", "../", "./"])
import torch
import pickle
import time
from torch import nn
from datetime import datetime
from LSTM.net.lstm import lstmnet
from LSTM.data_processing.dataset import  get_data
from LSTM.config.config import DefaultConfig


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


# ======================================================================================================================
# 模型评估
# ======================================================================================================================

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = torch.max(output, 1)

    num_correct = (pred_label == label).sum().item()

    return num_correct / total


# ======================================================================================================================
# 训练模型
# ======================================================================================================================
def train(model, train_data, valid_data, config, optimizer, criterion):

    model = model.to(config.device)
    prev_time = datetime.now()

    for epoch in range(config.num_epochs):
        model = model.train()
        train_loss = 0
        train_acc = 0

        for im, label in train_data:
            im = im.to(config.device)
            label = label.long().to(config.device)

            # forward
            output = model(im)

            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)

        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            model = model.eval()
            for im, label in valid_data:
                with torch.no_grad():
                    im = im.to(config.device)
                    label = label.long().to(config.device)

                output = model(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                    "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(train_data),
                       train_acc / len(train_data), valid_loss / len(valid_data),
                       valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)


# ======================================================================================================================
# 主函数
# ======================================================================================================================

if __name__ == '__main__':
    # gpu测试
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: ", torch.backends.cudnn.enabled)
    print('GPUs：', torch.cuda.device_count())

    torch.manual_seed(2019)
    torch.cuda.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)

    torch.set_num_threads(4)
    config = DefaultConfig()

    f_vectors = open(config.vector_path, 'rb')
    emb_vectors = torch.Tensor(pickle.load(f_vectors))

    train_data, validate_data = get_data()
    vocab_size, dim = emb_vectors.shape

    model = lstmnet(emb_weights=emb_vectors,
                    vocab_size=vocab_size,
                    dim=dim,
                    config=config)

    criterion = nn.CrossEntropyLoss()

    if config.emb_freeze:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()

    optimzier = torch.optim.Adam(model_parameters, lr=config.lr, weight_decay=config.weight_decay)

    train(model, train_data, validate_data, config, optimzier, criterion)
