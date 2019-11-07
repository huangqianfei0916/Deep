from gensim.models import word2vec
import gensim
import numpy as np
import pickle
import time
from sklearn import preprocessing

start_time = time.time()

# =======================================================================================================================
# 加载语料训练模型
# =======================================================================================================================

# sentences = word2vec.LineSentence("..\\data\\6ma\\words.txt")
#
# model = word2vec.Word2Vec(sentences, sg=1, hs=0, min_count=1, window=20, size=100)

# =======================================================================================================================
# 保存整个模型
# =======================================================================================================================

# model.save("atmodel6")
# model = word2vec.Word2Vec.load("vectors.txt")

# =======================================================================================================================
# 单独保存词向量
# =======================================================================================================================

# model.wv.save_word2vec_format("..\\data\\6ma\\6ma.model", binary=False)

model = gensim.models.KeyedVectors.load_word2vec_format("..\\data\\6ma\\ncp.model", binary=False)
# =======================================================================================================================
# 将单词和向量进行分开保存
# =======================================================================================================================
wv = model.wv
vocab_list = wv.index2word

"""将单词的下标和单词组成dict"""
word2id = {}
for id, word in enumerate(vocab_list):
    word2id[word] = id
print(word2id)

vectors = wv.vectors
print(vectors)


fw = open('..\\data\\6ma\\word2id.pkl', 'wb')
fv = open('..\\data\\6ma\\vectors.pkl', 'wb')
pickle.dump(word2id, fw)
pickle.dump(vectors, fv)
fw.close()
fv.close()

end_time = time.time()
print("耗时：{}s\n".format(end_time - start_time))
