"""
Copyright (c) 2024 by huangqianfei@tju.edu.cn All Rights Reserved. 
Author: huangqianfei@tju.edu.cn
Date: 2024-09-19 22:10:31
Description: 
"""



import torch

inputs = torch.FloatTensor([i for i in range(60)]).reshape(3,4,5)


print(inputs.shape)
print(inputs[:,0])
print(inputs[:,0,:])