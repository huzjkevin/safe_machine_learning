import torch
import torch.nn as nn
import sklearn.datasets as datasets
import pandas as pd
import os

# m = nn.LogSoftmax(dim=0)
# n = nn.LogSoftmax(dim=1)
# o = nn.LogSoftmax()
# input = torch.randn(2, 3)
#
# print(input)
# print()
#
# output1 = m(input)
# output2 = n(input)
# output3 = o(input)
#
# output11 = o(input[:, 0])
# output22 = m(input[0, :])
# output33 = o(input[:, 0])
# output44 = o(input[:, 0])
# print(output1)
# print(output2)
# print(output3)
# print()
#
# print(output11)
# print(output22)
# print(output33)
# print(output44)

bos = datasets.load_boston()
data = bos.data
target = bos.target

a = torch.utils.data.TensorDataset(data, target)
b = a.tensors

print(data)
print(target)