# RMSNorm

---
归一化（normalization），均方根（RMS）变为1，组合起来就得到了RMSNorm。
RMSNorm是一个非常常用也非常简单的算法。进入或离开某一层时均可以加入一个RMSNorm来整理数据的分布，可以减少参数在整理数据上的负担，让每一层都更加关注于学习特征，同时稳定参数，防止在多层的传递中变得过大或者过小。

```Python
import torch
import torch.nn as nn

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```

---
公式：

$y_i = \frac{x_i}{\sqrt{\frac{1}{n} \sum_{j=1}^{n} (x_j)^2 + \varepsilon}} \cdot \gamma$

这里添加 ε 主要是为了防止分母为零（例如整行输入全为0时）导致的数值不稳定。一种增强鲁棒性措施。

最后乘 γ 是对处理后数据进行自行缩放，一般放入优化器中自动优化。

---
实际模型里面，归一化层有很多可能放置的位置，目前常见的都是在正常层前面进行归一化处理，以及transformer内部的qk矩阵映射后面进行归一化。
