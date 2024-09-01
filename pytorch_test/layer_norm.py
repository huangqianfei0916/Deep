import torch.nn as nn
import torch
if __name__ == '__main__':
  norm = nn.LayerNorm([4,5], elementwise_affine=True)
  inputs = torch.FloatTensor([i for i in range(60)]).reshape(3,4,5)

  norm = nn.LayerNorm(5, elementwise_affine=True)
  inputs = torch.FloatTensor([i for i in range(20)]).reshape(4,5)
  print(inputs)

  output = norm(inputs)
  '''
  	tensor([[-1.3416, -0.4472,  0.4472,  1.3416],
    		[-1.3416, -0.4472,  0.4472,  1.3416]],
   				grad_fn=<NativeLayerNormBackward>)
  '''
  print(norm.weight.shape)
  print(norm.weight)
