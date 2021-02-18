import numpy as np
from collections import Counter

class BM25(object):
    def __init__(self, query_list, k1=2, k2=1, b=0.5):
        self.query_list = query_list
        self.query_number = len(query_list)
        self.avg_query_len = sum([len(doc) for doc in query_list]) / self.query_number
        self.f = []
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        df = {}
        for document in self.query_list:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log((self.query_number - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        query_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                        self.f[index][q] + self.k1 * (1 - self.b + self.b * query_len / self.avg_query_len))) * (
                                 qf[q] * (self.k2 + 1) / (qf[q] + self.k2))
        return score

    def get_query_score(self, query):
        score_list = []
        for i in range(self.query_number):
            score_list.append(self.get_score(i, query))
        return score_list

query_list = ["火锅加盟排行榜",
                 "奶茶加盟",
                 "桥头排骨加盟费",
                 "串串香加盟费多少",
                 "泡面小食堂加盟",
                 "大益茶加盟费多少"]
import jieba
query_list = [list(jieba.cut(doc)) for doc in query_list]
bm25 = BM25(query_list)
for item in bm25.query_list:
    print(item)
print("doc_num:",bm25.query_number)
print("avg_len:",bm25.avg_query_len)
print("f:")
for item in bm25.f:
    print(item)
print(bm25.idf)
query = "烧烤排骨加盟"
query = list(jieba.cut(query))
scores = bm25.get_query_score(query)
print(scores)
max_value = max(scores) #返回最大值
max_index = scores.index(max(scores)) # 返回最大值的索引
print("max_value:",max_value)
print("index:",max_index)
print("query",query)
print("doc:",query_list[max_index])
