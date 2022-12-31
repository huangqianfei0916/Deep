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


# import torch.nn as nn
# import torch
# if __name__ == '__main__':
#   norm = nn.LayerNorm([4,5], elementwise_affine=True)
#   inputs = torch.FloatTensor([i for i in range(60)]).reshape(3,4,5)

#   norm = nn.LayerNorm(5, elementwise_affine=True)
#   inputs = torch.FloatTensor([i for i in range(20)]).reshape(4,5)

#   output = norm(inputs)
#   print(output)
#   '''
#   	tensor([[-1.3416, -0.4472,  0.4472,  1.3416],
#     		[-1.3416, -0.4472,  0.4472,  1.3416]],
#    				grad_fn=<NativeLayerNormBackward>)
#   '''
#   print(norm.weight.shape)