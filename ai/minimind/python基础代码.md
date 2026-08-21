# Python 基础代码

MiniMind 学习中用到的基础方法整理。

## arange

```Python
import torch

t=torch.arange(0,10,2)
print(t)  # Output: tensor([0, 2, 4, 6, 8])

t2=torch.arange(5,0,-1)
print(t2)  # Output: tensor([5, 4, 3, 2, 1])
```

类似range，生成有序的一组数

## cat

```Python
import torch

t1=torch.tensor([[1,2,3],[4,5,6]])
t2=torch.tensor([[7,8,9],[10,11,12]])
result=torch.cat((t1,t2),dim=0)
print(result)  
#tensor([[ 1,  2,  3],
#        [ 4,  5,  6],
#        [ 7,  8,  9],
#        [10, 11, 12]])
result2=torch.cat((t1,t2),dim=1)
print(result2)  
# tensor([[ 1,  2,  3,  7,  8,  9],
#         [ 4,  5,  6, 10, 11, 12]])
```

在对应维度将两个矩阵拼接

计算维度可以看作是从外面向里面数这是第几个括号

## Dropout

```Python
import torch
import torch.nn as nn

dropout_layer = nn.Dropout(p=0.5)

t1=torch.Tensor([1,2,3])
t2=dropout_layer(t1)
#为了保持期望不变，dropout会将没有丢弃的元素乘1/(1-p)
print(t2)#假设丢弃1,输出tensor([0 , 4., 6.])
```

随机丢弃，一定的随机性是为了提升模型的泛化能力

## Linear

```Python
import torch
import torch.nn as nn

layer = nn.Linear(in_features=3, out_features=5, bias=True)
t1 = torch.Tensor([1, 2, 3])  # shape: (3,)

t2 = torch.Tensor([[1, 2, 3]])  # shape: (1, 3)
# 这里应用的w和b是随机的，真实训练里会在optimizer上更新
output2 = layer(t2)             # shape: (1, 5)
print(output2)
```

线性层，这其实就是一组（输入×输出）个 y=kx+b。线性映射而已。

## outer

```Python
import torch
v1=torch.tensor([1,2,3])
v2=torch.tensor([4,5,6])
result=torch.outer(v1,v2)
print(result)
#输出
#tensor([[ 4,  5,  6],
#        [ 8, 10, 12],
#        [12, 15, 18]])
```

求两个向量的外积，outer要求输入张量必须是一维的

## transpose

```Python
import torch

t1=torch.Tensor([[1,2,3],[4,5,6]])
t1=t1.transpose(0,1)
print(t1)
#tensor([[1., 4.],
#        [2., 5.],
#        [3., 6.]])
```

交换维度，transpose所作仅仅只是交换视图，不复制数据，仅改变维度顺序。会对原本变量的内存上进行操作。

## unsqueeze

```Python
import torch
t1 = torch.Tensor([1, 2, 3])
t2 = t1.unsqueeze(0)
print(t2)
#tensor([[1., 2., 3.]])
#torch.Size([1, 3])
```

扩展维度，在所指示的维度扩展出一个新的维度。这里就可以看见原本的[3]变成了[1,3]

## view

```Python
import torch
t = torch.tensor([[ 1,  2,  3,  4,  5,  6],
                  [ 7,  8,  9, 10, 11, 12]])
t_view1 = t.view(3, 4)
print(t_view1)
t_view2 = t.view(4, 3)
print(t_view2)
#tensor([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12]])
#tensor([[ 1,  2,  3],
#        [ 4,  5,  6],
#        [ 7,  8,  9],
#        [10, 11, 12]])
```

对张量的形状进行重塑。功能类似于reshape，但是view对内存的要求更高。

reshape会对内存中不连续的数进行复制并放在其他地方。

view只会对原本内存中连续的数进行操作，不复制。例如经过transpose的操作就会破坏在内存中的连续性。

可以通过`t.contiguous()`重新调整内存（必要时复制数据到连续内存）；`t.is_contiguous()`只是检查是否连续，并不会改变内存布局。

## where

```Python
import torch

x=torch.tensor([1, 2, 3, 4, 5])
y=torch.tensor([10, 20, 30, 40, 50])

condition=(x>3)

result=torch.where(condition,x,y)
print(result)  # Output: tensor([10, 20, 30,  4,  5])
```

根据条件从两个张量中选择元素。设置的条件为x>3,where就会依据条件进行选择，因为下标3、4的元素（值4、5）符合条件，所以取前面的x；0、1、2不满足条件，取后面的y。

类似于c语言的 a？x：y
