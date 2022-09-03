from operator import mod
import os
from statistics import mode

import networkx as nx
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

CUR_DIR = os.path.abspath(os.path.dirname(__file__))


def build_graph():
    # 节点1 \t 节点2 \t 边权重
    df = pd.read_csv(CUR_DIR + "/space_data.tsv", sep = "\t")
    G = nx.from_pandas_edgelist(df, "source", "target", edge_attr = True, create_using=nx.Graph())

    return G


def get_randomwalk(node, path_length, G):
    """walk"""
    random_walk = [node]
    
    for i in range(path_length - 1):
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))    
        if len(temp) == 0:
            break

        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node
        
    return random_walk


def train_model(G):
    """train model"""
    all_nodes = list(G.nodes())
    random_walks = []
    for n in tqdm(all_nodes):
        for i in range(5):
            random_walks.append(get_randomwalk(n, 10, G))

    print(len(random_walks))

    model = Word2Vec(random_walks, sg = 1, hs = 0, min_count = 1, window = 4, size = 100, iter = 20)

    print(model)

    return model


def plot_nodes(word_list, model):
    """draw"""
    X = model[word_list]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    
    plt.figure(figsize = (12, 9))
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(word_list):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
        
    plt.show()
    # plt.savefig("./deepwalk.png",dpi=300) 


if __name__ == "__main__":
    G = build_graph()

    model = train_model(G)
    print(model.similar_by_word('astronaut training'))

    terms = ['lunar escape systems','soviet moonshot', 'soyuz 7k-l1', 'moon landing',
         'space food', 'food systems on space exploration missions', 'meal, ready-to-eat',
         'space law', 'metalaw', 'moon treaty', 'legal aspects of computing',
         'astronaut training', 'reduced-gravity aircraft', 'space adaptation syndrome', 'micro-g environment']
    plot_nodes(terms, model)