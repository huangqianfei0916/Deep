# ======================================================================================================================
# 将fasta文件进行分词，修改fastafilepath（源文件）和wordfilepath（目标文件存储路径）的路径即可
# ======================================================================================================================
# kmer切分
# b = [document[i:i + 3] for i in range(len(document)) if i < len(document) - 2]
# b = re.findall(r'.{3}', string)

import re
import time

start_time=time.time()
fastafilepath = "..\\data\\4mc\\4mc.txt"
wordfilepath = "..\\data\\4mc\\4mcword.txt"

f = open(fastafilepath)
f1 = open(wordfilepath, "w")

documents = f.readlines()
string=""
flag=0
for document in documents:
    if document.startswith(">") and flag == 0:
        flag = 1
        continue
    elif document.startswith(">") and flag == 1:
        b = [string[i:i + 3] for i in range(len(string)) if i < len(string) - 2]
        word = " ".join(b)
        f1.write(word)
        f1.write("\n")
        string = ""
    else:
        string += document
        string = string.strip()
b = [string[i:i + 3] for i in range(len(string)) if i < len(string) - 2]
word = " ".join(b)
f1.write(word)
f1.write("\n")

print("训练结果已保存到{}该目录下！\n".format(wordfilepath))

end_time = time.time()
print("耗时：{}s\n".format(end_time - start_time))
f1.close()
f.close()

