import numpy as np
import jieba
'''
tf-idf
tf是一个列表，其中存放很多词典（词典个数等于doc数量），词典是意思是每个词在单个doc的词频；可能不同的词典出现相同的词（key）；
idf是一个词典，这个是针对整个语料的词典，存放每个词的idf值；
tf-idf的核心是：一个词在当前doc出现的频率很高，但是在整个语料其他doc出现频率很低，那么这个就可以很好的代表当前的doc；
'''
class TF_IDF(object):
    def __init__(self, query_list):
        self.query_list = query_list
        self.query_number = len(query_list)
        self.tf = []
        self.idf = {}
        df = {}
        for document in self.query_list:
            # temp同于计算单个词在每一个doc中出现的频率；
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1/len(document)
            self.tf.append(temp)
            # df的key是整个语料的词，value是这个词出现在整个语料的几个doc中；
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        # 计算idf
        for key, value in df.items():
            self.idf[key] = np.log(self.query_number / (value + 1))
    # 计算query在每个doc下的tf-idf，也就是判断query中的每个词是否在当前doc，若是在就累加对应词的tf-idf，知道query结束。
    def get_score(self, index, query):
        score = 0.0
        for q in query:
            if q not in self.tf[index]:
                continue
            score += self.tf[index][q] * self.idf[q]
        return score

    def get_query_score(self, query):
        score_list = []
        # 计算每个doc和当前的query之间的相似度，也就是最终的score
        for i in range(self.query_number):
            score_list.append(self.get_score(i, query))
        return score_list

if __name__ == "__main__":
    query_list = ["火锅加盟排行榜",
                    "奶茶加盟",
                    "桥头排骨加盟费",
                    "串串香加盟费多少",
                    "泡面小食堂加盟",
                    "大益茶加盟费多少"]

    query_list = [list(jieba.cut(doc)) for doc in query_list]
    tf_idf = TF_IDF(query_list)
    for item in tf_idf.query_list:
        print(item)
    print(tf_idf.query_number)
    print("tf:")
    for item in tf_idf.tf:
        print(item)
    print("idf:")
    print(tf_idf.idf)

    query = "烧烤排骨加盟"
    query = list(jieba.cut(query))
    print("概率：")
    scores = tf_idf.get_query_score(query)
    print(scores)
    max_value = max(scores) #返回最大值
    max_index = scores.index(max(scores)) # 返回最大值的索引
    print("max_value:",max_value)
    print("index:",max_index)
    print("query",query)
    print("doc:",query_list[max_index])




