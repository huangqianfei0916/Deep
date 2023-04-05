'''
Author: huangqianfei
Date: 2023-04-05 10:17:32
LastEditTime: 2023-04-05 16:59:49
Description: 正向最大匹配算法
'''

def FMM(sentence, dic):
    max_len = max([len(i) for i in dic])
    idx = 0
    sent_len = len(sentence)
    res = []
    while idx < sent_len:
        tmp_idx = idx + max_len
        while tmp_idx > idx:
            sent_part = sentence[idx:tmp_idx]
 
            if sent_part in dic or len(sent_part) == 1:
                res.append(sent_part)
                idx += tmp_idx - idx
                break

            else:
                tmp_idx -= 1

    return res


if __name__ == "__main__":
    sentence = "北京大学生前来应聘。"
    dic = ["北京大学", "生前", "前来", "来", "北京", "大学", "应聘", "大学生"]
    print(f"The cut result of fmm {FMM(sentence, dic)}")