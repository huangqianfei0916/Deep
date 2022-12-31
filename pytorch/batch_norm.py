import torch.nn as nn
import torch
if __name__ == '__main__':
  norm = nn.BatchNorm1d(4, affine=True)
  inputs = torch.FloatTensor([i for i in range(600)]).reshape(3,4,50)

  # norm = nn.BatchNorm1d(500, affine=True)
  # inputs = torch.FloatTensor([i for i in range(2000)]).reshape(4,500)

  print(inputs)
  output = norm(inputs)
  
  print(output)
  '''
  	tensor([[-1.0000, -1.0000, -1.0000, -1.0000],
    		[ 1.0000,  1.0000,  1.0000,  1.0000]])
  '''
print(norm.running_mean.shape)
print(norm.weight.shape)


