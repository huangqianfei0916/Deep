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


def build_graph():
    # 节点1 \t 节点2 \t 边权重
    df = pd.read_csv("space_data.tsv", sep = "\t")
    G=nx.from_pandas_edgelist(df, "source", "target", edge_attr=True, create_using=nx.Graph())

    print(len(G))
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

    model = Word2Vec(window = 4, sg = 1, hs = 0,
                    negative = 10,
                    alpha=0.03, min_alpha=0.0007,
                    seed = 14)

    model.build_vocab(random_walks, progress_per=2)
    model.train(random_walks, total_examples = model.corpus_count, epochs=20, report_delay=1)

    print(model)

    model.similar_by_word('astronaut training')

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


if __name__ == "__main__":
    G = build_graph()
    get_randomwalk('space exploration', 10, G)
    model = train_model(G)
    terms = ['lunar escape systems', 'soviet moonshot', 'soyuz 7k-l1', 'moon landing',
         'space food', 'food systems on space exploration missions', 'meal, ready-to-eat',
         'space law', 'metalaw', 'moon treaty', 'legal aspects of computing',
         'astronaut training', 'reduced-gravity aircraft', 'space adaptation syndrome', 'micro-g environment']
    plot_nodes(terms, model)