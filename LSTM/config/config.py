import torch

class DefaultConfig(object):
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # path
    vector_path = '..\\data\\6ma\\vectors.pkl'
    word2id = "..\\data\\6ma\\word2id.pkl"
    data_path = "..\\data\\6ma\\words.txt"
    data_path2 = "..\\data\\6ma\\words.txt"

    pos = 880
    neg = 880
    total = 1760
    fix_length = 39

    """模型结构参数"""
    model_name = 'LSTM'
    emb_freeze = True
    input_size = 50
    hidden_size = 100
    bidir = True
    batch_first = True
    num_layers = 1
    dropout = 0.5
    num_classes = 2

    """训练参数"""
    random_seed = 1
    use_gpu = True
    lr = 0.01
    weight_decay = 0
    num_epochs = 800
    data_shuffle = True
    batch_size = 32
