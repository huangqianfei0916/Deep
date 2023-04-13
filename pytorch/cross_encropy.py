'''
Author: huangqianfei
Date: 2023-04-06 20:31:04
LastEditTime: 2023-04-13 20:27:13
Description: 
'''
import torch
import torch.nn as nn
--------------------soft label-------------------------
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
softmax = nn.Softmax()
x_softmax = softmax(input)
x_log = torch.log(x_softmax)
# tensor([[-1.6752, -1.0925, -4.2082, -1.6690, -1.2945],
#         [-1.6818, -0.6926, -3.0213, -2.2847, -1.8132],
#         [-3.6919, -3.9275, -0.1586, -4.3641, -2.4158]], grad_fn=<LogBackward0>)
c = x_log * target
d = torch.sum(c, dim = 1)
e = -torch.sum(d, dim = 0)
print(e / 3)
# tensor(2.5833, grad_fn=<DivBackward0>)

output = loss(input, target)
output.backward()
print(output)
# tensor(2.5833, grad_fn=<DivBackward1>)
--------------------hard label-------------------------
y = torch.empty(3, dtype=torch.long).random_(5)
loss_func = nn.NLLLoss()
res1 = loss_func(x_log, y)

res2 = loss(input, y)
print(res1)
print(res2)
# tensor(1.9085, grad_fn=<NllLossBackward0>)
# tensor(1.9085, grad_fn=<NllLossBackward0>)