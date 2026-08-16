# MiniMind——大模型入门笔记

关于MiniMind项目:[Github链接](https://github.com/jingyaogong/minimind?tab=readme-ov-file)

推荐学习视频：[视频链接](https://www.bilibili.com/video/BV1T2k6BaEeC?spm_id_from=333.788.videopod.episodes&vd_source=8704f44a88fba43c396e2910ec5bbdc8)

minimind学习总结，复盘。

# method（基础语法知识）

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
result=torch.cat((t1,t2),*dim*=0)
print(result)  
#tensor([[ 1,  2,  3],
#        [ 4,  5,  6],
#        [ 7,  8,  9],
#        [10, 11, 12]])
result2=torch.cat((t1,t2),*dim*=1)
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

dropout_layer = nn.Dropout(*p*=0.5)

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

layer = nn.Linear(*in_features*=3, *out_features*=5, *bias*=True)
t1 = torch.Tensor([1, 2, 3])  # shape: (3,)

t2 = torch.Tensor([[1, 2, 3]])  # shape: (1, 3)
# 这里应用的w和b是随机的，真实训练里会在optimizer上更新
output2 = layer(t2)             # shape: (1, 5)
print(output2)
```

线性层，这其实就是一组（*输入\*输出）*个y=kx\+b。线性映射而已。

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

扩展维度，在所指示的维度扩展出一个新的维度。这里就可以看见原本的\[3\]变成了\[1,3\]

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

view只会对原本内存中连续的数进行操作，不复制。例如经过transpose的操作就会破环在内存中的连续性。

可以通过`t.is_contiguous()`重新调整内存。

## where

```Python
import torch

x=torch.tensor([1, 2, 3, 4, 5])
y=torch.tensor([10, 20, 30, 40, 50])

condition=(x>3)

result=torch.where(condition,x,y)
print(result)  # Output: tensor([10, 20, 30,  4,  5])
```

根据条件从两个张量中选择元素。设置的条件为x\>3,where就会依据条件进行选择，因为0,1,2号元素符合条件，所以0,1,2号取前面的值，3,4不满足，取后面的值。

类似于c语言的 a？x：y

# 模型结构

![LLM\-structure\.jpg](/assets/minimind/minimind LLM-structure.jpg)

这里将会聚焦在图中白色框的部分，仅关注模型本身，暂时不涉及tokenizer。

模型结构分为两部分：

第一部分是模型的底层实现

第二部分是模型各部分的拼接

## 实现

### RMSNorm

```Python
class RMSNorm(*torch*.*nn*.Module):
    def __init__(*self*, *dim*: int, *eps*: float = 1e-5):
        super().__init__()
        *self*.eps = *eps*
        self.dim = dim
        *self*.weight = nn.Parameter(torch.ones(*dim*))

    def norm(*self*, *x*):
        return *x* * torch.rsqrt(*x*.pow(2).mean(-1, *keepdim*=True) + *self*.eps)

    def forward(*self*, *x*):
        return (*self*.weight * *self*.norm(*x*.float())).type_as(*x*)
```

归一化，将标准差变1。

公式：

$y_i = \frac{x_i}{\sqrt{\frac{1}{n} \sum_{j=1}^{n} (x_j)^2 + \varepsilon}} \cdot \gamma$

这里添加ε主要是为了防止分母过小导致的梯度爆炸，也是防止分母为0。一种增强鲁棒性措施。

最后乘γ是对处理后数据进行自行缩放，一般放入优化器中自动优化

关于RMSNorm，这是一个非常常用也非常简单的算法。进入或离开某一层时均可以加入一个RMSNorm来优化参数，可以让每一层都更加关注于学习特征，同时稳定参数，防止在多层的传递中变得过大或者过小。

### Transformer

#### GQA attention

多头注意力，通过注意力机制让大模型自主决定信息的权重并提取关键的信息。

这个就是transformer不同于传统网络的核心。

这一部分东西较多，我将仔细拆解。

##### 理论

attention的公式展示

$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V$

attention会将一个输入同时做3个不同的线性映射。这三个线性映射分别命名为Q,K,（query，key，value）

这里的Q是查询头，K是键，V是权重。

attention的实现逻辑大致就是用q查询k，然后将最契合的值从v中提取出来。

---

这里就可以看公式了

$\frac{QK^T}{\sqrt{d_k}}$这就是做查询。qk相乘之后就会得到不同位置的张量的重要程度。除根下k的个数来控制方差。

这里的q和k都要求是均值为0，方差为1的一组张量，以让相乘的结果矩阵的方差为1。

这样经过softmax之后，即可以突出重点，又不会一家独大最合适的数值。可以让模型更加稳定。



然后将相乘结果经过softmax进行运算

softmax公式，知道作用即可：

$\operatorname{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$

softmax会让所有数据的和为1，将原本的重要程度数据变成更加直观实用的比率。



最后与价值矩阵v相乘，提取出原本输入中所包含的信息。

得到最后的矩阵。

这个矩阵就包括了大模型自主提取的重要信息。

---

实际上仅仅这样还不足以写出attention，仍然有一些细节需要处理，详看[补充](https://my.feishu.cn/wiki/LKLEwXWQAiPF6Bke3KScwV0XnVO#share-NDbmdsiWzouFLDx32eicH5isnld)。

---

以上就是基本attention（MHA）的原理，但是这样的attention计算量十分巨大。也会产生巨大的kv\_cache\(历史记忆\)

因此为了减少计算量，就产生了MHA的优化版本GQA。

GQA相比MHA，减少了kv头的个数，采用多个q共享一个kv的策略。

实际效果和MHA相近，可以减少kv\_cache,优化显存占用



为什么这么做有效？

或许可以理解为有同一片资源，我们可以用不同的方式询问。多角度询问来减少信息的损失。



关键参数：

`h`：查询头数（Query heads）

`g`：KV 头数（Key/Value heads，即分组数）

组内 Q 头数：`h/g`（必须整除）

极端情况：

g=1 → 退化为 **MQA**

g=h → 退化为 **MHA**

实际工程上验证大概4个q对应1个kv是一个最佳参数

###### 补充1：mask

在实际的训练上，因为大模型需要根据以前信息预测下一个字，所以不能让模型看见未来信息。就需要在模型训练的时候添加一个mask掩码，将未来信息遮盖住。

实际推理的时候需要和训练保持一致，所以mask需要一直存在。

mask通常会加在qk相乘之后，将当前位置之后的数值设为负无穷，这样就可以在softmax之后被计算为0。

ps：这里的mask必须放在softmax之前，在softmax之后直接设0会导致原本的和为1关系被破坏。

![3Blue1Brown transformer注意力讲解.png](/assets/minimind/3Blue1Brown transformer注意力讲解.png)

###### 补充2：RoPE

位置编码器。为原本独立的token添加位置信息。

如果重新观察attention的计算过程，也许就会发现，输入的token都是各自独立的，token之间没有任何关系可以表示。但这肯定是不合适的，文字的排列本身就包含了大量的信息。因此我们需要通过一些手段给token添加一种可以表示位置的信息。

添加位置信息有两种方式：绝对位置编码和相对位置编码。

绝对位置编码，比如给每个token添加一个顺序的数字。

这种编码方式存在着一些缺陷，绝对位置的注意力容易被限制，泛化能力不足。因此现在一般使用相对位置编码。

相对位置编码，例如RoPE（旋转位置编码）。

RoPE会根据每个token的位置和token内部每个不同的维度将数据进行不同的旋转

---

RoPE的数学原理

对于位置m，向量**x**=\(*x*1,*x*2\)旋转：

![截图 2026\-04\-18 09\-29\-50\.png](/assets/minimind/旋转矩阵计算式.png)

同理位置 *n*：*Rn*\(**y**\)。

计算旋转后的内积：

![图片\.png](/assets/minimind/两两旋转的证明.png)

化简：

![图片\.png](/assets/minimind/相对位置证明.png)

最后可以发现，因为x,y均是已知的，所以两个token之间的关系仅仅由m\-n决定

---



$\text{freqs}_i = \frac{1}{\text{rope\_base}^{i / \text{dim}}}, \quad i = 0, 1, \dots, \frac{\text{dim}}{2} - 1$

这个就是rope的实现。

rope会同时操作q和k，将q和k各自相乘sin\(freqs\)和cos\(freqs\)。

从公式中可以看出，rope会对不同的token的同一维度旋转不同的角度。

对同一token的不同维度提供不同的频率的旋转。

高维度旋转慢，能够覆盖超长文本，感知全局

低维度旋转快，对附近的token十分敏感，能够捕捉附近信息。

这样子的差异就为token提供了覆盖远近不同的位置信息。

###### 补充2的补充：YaRN

YaRN就是对RoPE的优化。

原本RoPE的频率分布是一个前期突变十分剧烈，后期几乎固定不变的曲线。

这样子就导致后面维度的变化不明显，难以区分位置。这会影响到大模型的长文本能力

而yarn所作的就是优化这条频率曲线，如图：

![图片\.png](/assets/minimind/yarn效果.png)

图中可以明显看见yarn针对低频部分和rope有明显变化，这一部分主要就是让位置编码在低频部分也能够保持足够的区分度。

同时在高频和低频之间的部分，使用插值的方法平滑过渡曲线。

这样就得到了更加平滑合适的频率曲线。



##### attention代码和解读

```Python
#预计算频率，提前将sin和cos打表，减少训练时候的计算量
def precompute_freqs_cis(*dim*: int, *end*: int = int(32 * 1024), *rope_base*: float = 1e6, *rope_scaling*: dict = None):
    #这里的rope就决定了大模型的最大输出长度。相当于底子。
    #这个是根据数据集的样本平均长度决定的，通常会比平均长度稍微多一点
    #*L_*max≈rope_base^(2/*d)这里设定1e6说明标准rope最大支持32k上下文。使用yarn可以推广到64k-128k*
    freqs, attn_factor = 1.0 / (*rope_base* ** (torch.arange(0, *dim*, 2)[: (*dim* // 2)].float() / *dim*)), 1.0
    if *rope_scaling* is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            *rope_scaling*.get("original_max_position_embeddings", 2048), 
            *rope_scaling*.get("factor", 16),
            *rope_scaling*.get("beta_fast", 32.0), 
            *rope_scaling*.get("beta_slow", 1.0), 
            *rope_scaling*.get("attention_factor", 1.0)
        )#设定参数
        #仅当最大序列长度超过模型支持最大序列长度时开启
        if *end* / orig_max > 1.0:
            #自动计算低频，中频，高频范围。
            inv_dim = lambda *b*: (*dim* * math.log(orig_max / (*b* * 2 * math.pi))) / (2 * math.log(*rope_base*))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), *dim* // 2 - 1)
            #生成频率曲线
            ramp = torch.clamp((torch.arange(*dim* // 2, *device*=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(*end*, *device*=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], *dim*=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], *dim*=-1) * attn_factor
    return freqs_cos, freqs_sin
```

ps：有点一知半解，不要忘了再看看



```Python
#应用位置编码
def apply_rotary_pos_emb(*q*, *k*, *cos*, *sin*, *unsqueeze_dim*=1):
    def rotate_half(*x*): return torch.cat((-*x*[..., *x*.shape[-1] // 2:], *x*[..., : *x*.shape[-1] // 2]), *dim*=-1)
    q_embed = ((*q* * *cos*.unsqueeze(*unsqueeze_dim*)) + (rotate_half(*q*) * *sin*.unsqueeze(*unsqueeze_dim*))).to(*q*.dtype)
    k_embed = ((*k* * *cos*.unsqueeze(*unsqueeze_dim*)) + (rotate_half(*k*) * *sin*.unsqueeze(*unsqueeze_dim*))).to(*k*.dtype)
    return q_embed, k_embed
    
#复制多个k和v，让kv数量和q相等，做兼容
def repeat_kv(*x*: torch.Tensor, *n_rep*: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = *x*.shape
    if *n_rep* == 1: return *x*
    #expand复制，相对于直接相乘，会节省显存。这样子操作并不会真的多复制出来维度。
    return (*x*[:, :, :, None, :].expand(bs, slen, num_key_value_heads, *n_rep*, head_dim).reshape(bs, slen, num_key_value_heads * *n_rep*, head_dim))
```

这里要复制k和v，让他们的数量和q对等起来。

否则后面做点积的时候会因为数量不对而报错。



```Python
class Attention(*nn*.Module):
    def __init__(*self*, *config*: MiniMindConfig):
        super().__init__()
        #设定头相关参数
        *self*.num_key_value_heads = *config*.num_attention_heads if *config*.num_key_value_heads is None else *config*.num_key_value_heads
        *self*.n_local_heads = *config*.num_attention_heads
        *self*.n_local_kv_heads = *self*.num_key_value_heads
        #计算需要复制kv的倍数
        *self*.n_rep = *self*.n_local_heads // *self*.n_local_kv_heads
        #头的维度
        *self*.head_dim = *config*.head_dim
        #因果掩码设定
        *self*.is_causal = True
        #qk矩阵，由于本质就是线性映射，所以可以一个用线性层一次性全部初始化，之后再修改形状。
        *self*.q_proj = nn.Linear(*config*.hidden_size, *config*.num_attention_heads * *self*.head_dim, *bias*=False)
        *self*.k_proj = nn.Linear(*config*.hidden_size, *self*.num_key_value_heads * *self*.head_dim, *bias*=False)
        *self*.v_proj = nn.Linear(*config*.hidden_size, *self*.num_key_value_heads * *self*.head_dim, *bias*=False)
        #最后输出前的线性层
        *self*.o_proj = nn.Linear(*config*.num_attention_heads * *self*.head_dim, *config*.hidden_size, *bias*=False)
        #归一化
        *self*.q_norm = RMSNorm(*self*.head_dim, *eps*=*config*.rms_norm_eps)
        *self*.k_norm = RMSNorm(*self*.head_dim, *eps*=*config*.rms_norm_eps)
        #这些是其他处理，放在计划拼接的部分
        *self*.attn_dropout = nn.Dropout(*config*.dropout)
        *self*.resid_dropout = nn.Dropout(*config*.dropout)
        *self*.dropout = *config*.dropout
        #计算加速的库，知道就好了，计算点乘用的时候带上。
        *self*.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and *config*.flash_attn
```

初始化



```Python
def forward(*self*, *x*, *position_embeddings*, *past_key_value*=None, *use_cache*=False, *attention_mask*=None):
        #提取输入矩阵的形状，不着急，后面的数据流会有展示，可以对照。
        bsz, seq_len, _ = *x*.shape
        #这里做线性映射
        xq, xk, xv = *self*.q_proj(*x*), *self*.k_proj(*x*), *self*.v_proj(*x*)
        #这里的xq，xk，xv就包含了多头的结果（前面初始化把多头一次性全部包括了）。
        #这里修改形状，方便后面点积。
        xq = xq.view(bsz, seq_len, *self*.n_local_heads, *self*.head_dim)
        xk = xk.view(bsz, seq_len, *self*.n_local_kv_heads, *self*.head_dim)
        xv = xv.view(bsz, seq_len, *self*.n_local_kv_heads, *self*.head_dim)
        #xq，xk过一次归一化，稳定数据。
        xq, xk = *self*.q_norm(xq), *self*.k_norm(xk)
        #为xq和xk添加位置信息，在这里添加是因为xk之后会保存下来成为记忆。xq也需要根据位置信息做计算。
        #这里的*position_embeddings是是前面预计算的结果*
        cos, sin = *position_embeddings*
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        #如果有kv缓存
        if *past_key_value* is not None:
            #把历史kv和当前kv拼接，这样子attention就可以关注以前的信息
            xk = torch.cat([*past_key_value*[0], xk], *dim*=1)
            xv = torch.cat([*past_key_value*[1], xv], *dim*=1)
        #把处理后的kv作为输出，相当于整理的一下内存，并且把现在的kv也加入到了缓存里面。
        #这样子内存分配连续，更有利于gpu计算
        past_kv = (xk, xv) if *use_cache* else None
        #变换维度，并且匹配qkv的数量
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, *self*.n_rep).transpose(1, 2), repeat_kv(xv, *self*.n_rep).transpose(1, 2))
        #使用优化后的库
        if *self*.flash and (seq_len > 1) and (not *self*.is_causal or *past_key_value* is None) and (*attention_mask* is None or torch.all(*attention_mask* == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, *dropout_p*=*self*.dropout if *self*.training else 0.0, *is_causal*=*self*.is_causal)
        else:
            #计算注意力权重
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(*self*.head_dim)
            #因果掩码
            if *self*.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), *device*=scores.device).triu(1)
            #遮盖占位符。比如有些句子没法占满最大长度，就会用占位符占满
            if *attention_mask* is not None: scores += (1.0 - *attention_mask*.unsqueeze(1).unsqueeze(2)) * -1e9
            #将注意力权重和v点乘，得到最后的结果，然后做其他处理
            #这里对注意力权重做了一次dropout，这里的dropout主要是为了防止模型产生过度依赖，增强鲁棒性
            output = *self*.attn_dropout(F.softmax(scores.float(), *dim*=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = *self*.resid_dropout(*self*.o_proj(output))
        return output, past_kv

```

核心计算过程

#### Feed Foward Network

前馈神经网络，负责将attention提取到的信息进行推理。

```Python
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        *self*.dropout = nn.Dropout(*config*.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
```

这里仅关注ffn本身，残差连接和归一化在拼接模型里面解释。

ffn核心就是三个线性映射和一个非线性的激活函数。

ffn会先将输入进行升维，up\_linear和gate\_linear会同时升维并让gate\_linear通过激活函数做非线性变换。

将变换后的两组数据做对应位置相乘，再将结果通过down\_linear将高维信息总结成低维的结果。

##### 激活函数

激活函数就是为大模型带来推理能力的核心，如果没有激活函数，模型失去非线性，退化为单层网络。y=k1\(k2\*x\)=k3\*x



SiLU和ReLU激活函数图像

![screenshot\_20260417\_090036\.png](/assets/minimind/SiLU和ReLU激活函数.png)

这里用ReLU讲解会更加便于理解。激活函数就是在不同的位置为数据做不同的变换，这一点在ReLU上体现十分鲜明。

在大于0的部分，ReLU会将数据对应的乘一个x，数据越大，处理的幅度越大，所以ReLU对偏移较大的数据的敏感度较大。

在小于0的部分，ReLU就会将数据全部归为0，可以理解为将所有无效或者权重较低的信息过滤掉。这样就会突出有效的信息。

而SiLU就是平滑版的ReLU，平滑更加适合求导，解决在0点处导数不存在的问题，也解决ReLU在小于0处完全失去任何信息的问题（无效信息本身也是一种信息，他可以指导模型进行优化，但ReLU把他们全部都砍没了，SiLU的一点平滑处理解决了这个问题）。

当然对于激活函数，还有sigmod等等很多其他的函数，这些就不展开了。

##### 线性映射

ffn中的线性映射会先将输入数据升维到intermediate\_size大小。在处理之后再缩回原本的hidden\_size大小。

实际使用通常设置为hidden\_stated的4倍大小

可以理解为人类思考时候脑海内部的抽象空间。

更高维的信息可以抽象出更丰富的信息，并且减少在经过非线性变换的时候损失的信息。





## 拼接

一个block，先把FFN和GAQ拼接起来，这就是一个transformer

代码：

```Python
class MiniMindBlock(*nn*.Module):
    def __init__(*self*, *layer_id*: int, *config*: MiniMindConfig):
        super().__init__()
        *self*.self_attn = Attention(*config*)
        *self*.input_layernorm = RMSNorm(*config*.hidden_size, *eps*=*config*.rms_norm_eps)
        *self*.post_attention_layernorm = RMSNorm(*config*.hidden_size, *eps*=*config*.rms_norm_eps)
        *self*.mlp = FeedForward(*config*) if not *config*.use_moe else MOEFeedForward(*config*)

    def forward(*self*, *hidden_states*, *position_embeddings*, *past_key_value*=None, *use_cache*=False, *attention_mask*=None):
        #这里包含了两个残差连接，两种写法等价。
        #核心原理就是在开始之前先保存一下开始的状态，然后把变换之后的状态和开始的状态相加
        #为什么有效？可以理解为y=kx和y=kx+b。有了b，模型可以专心拟合曲率，减少拟合压力。
        residual = *hidden_states*
        *hidden_states*, present_key_value = *self*.self_attn(
            *self*.input_layernorm(*hidden_states*), *position_embeddings*,
            *past_key_value*, *use_cache*, *attention_mask*
        )
        *hidden_states* += residual
        *hidden_states* = *hidden_states* + *self*.mlp(*self*.post_attention_layernorm(*hidden_states*))
        return *hidden_states*, present_key_value
```

将transformer和其他层封装，生成一个模型的框架

```Python
class MiniMindModel(*nn*.Module):
    def __init__(*self*, *config*: MiniMindConfig):
        super().__init__()
        *self*.config = *config*
        *self*.vocab_size, *self*.num_hidden_layers = *config*.vocab_size, *config*.num_hidden_layers
        *self*.embed_tokens = nn.Embedding(*config*.vocab_size, *config*.hidden_size)
        *self*.dropout = nn.Dropout(*config*.dropout)
        #这是为了多次重复提取重要信息
        *self*.layers = nn.ModuleList([MiniMindBlock(l, *config*) for l in range(*self*.num_hidden_layers)])#生成num_hidden_layers个block，每个block里有一个Attention和一个FeedForward（或MOEFeedForward）。
        *self*.norm = RMSNorm(*config*.hidden_size, *eps*=*config*.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(*dim*=*config*.head_dim, *end*=*config*.max_position_embeddings, *rope_base*=*config*.rope_theta, *rope_scaling*=*config*.rope_scaling)
        *self*.register_buffer("freqs_cos", freqs_cos, *persistent*=False)
        *self*.register_buffer("freqs_sin", freqs_sin, *persistent*=False)

    def forward(*self*, *input_ids*, *attention_mask*=None, *past_key_values*=None, *use_cache*=False, ***kwargs*):
        batch_size, seq_length = *input_ids*.shape
        if hasattr(*past_key_values*, 'layers'): *past_key_values* = None
        *past_key_values* = *past_key_values* or [None] * len(*self*.layers)
        start_pos = *past_key_values*[0][0].shape[1] if *past_key_values*[0] is not None else 0
        hidden_states = *self*.dropout(*self*.embed_tokens(*input_ids*))
        position_embeddings = (*self*.freqs_cos[start_pos:start_pos + seq_length], *self*.freqs_sin[start_pos:start_pos + seq_length])
        presents = []

        for layer, past_key_value in zip(*self*.layers, *past_key_values*):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                *past_key_value*=past_key_value,
                *use_cache*=*use_cache*,
                *attention_mask*=*attention_mask*
            )
            presents.append(present)
        hidden_states = *self*.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in *self*.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss
```

将模型封装进训练框架，搭建好模型训练的框架。

```Python
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    def __init__(*self*, *config*: MiniMindConfig = None):
        *self*.config = *config* or MiniMindConfig()
        super().__init__(*self*.config)
        *self*.model = MiniMindModel(*self*.config)
        *self*.lm_head = nn.Linear(*self*.config.hidden_size, *self*.config.vocab_size, *bias*=False)
        *self*.model.embed_tokens.weight = *self*.lm_head.weight
    
    def forward(*self*, *input_ids*, *attention_mask*=None, *past_key_values*=None, *use_cache*=False, *logits_to_keep*=0, *labels*=None, ***kwargs*):
        hidden_states, *past_key_values*, aux_loss = *self*.model(*input_ids*, *attention_mask*, *past_key_values*, *use_cache*, ***kwargs*)
        slice_indices = slice(-*logits_to_keep*, None) if isinstance(*logits_to_keep*, int) else *logits_to_keep*
        logits = *self*.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if *labels* is not None:
            x, y = logits[..., :-1, :].contiguous(), *labels*[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), *ignore_index*=-100)
        return MoeCausalLMOutputWithPast(*loss*=loss, *aux_loss*=aux_loss, *logits*=logits, *past_key_values*=*past_key_values*, *hidden_states*=hidden_states)
    
```

# 预训练，微调和强化学习

这里的三个基本逻辑是相似的，都是在训练模型的能力，不过侧重不同

预训练是为了给模型训练出基本的语言能力，让模型识字，学基础知识。

微调是为了给模型训练基本的回答能力，让模型学会对话。

强化学习是为了给模型训练更加得体的表达能力，让模型的回答更加符合人类喜好。

从前到后，学习率逐级降低，模型会学习到更加精细的能力。

逐级逼近最优点。

## 预训练



因为预训练的特点，其数据集会提供大量原始的文本，使用一种填鸭式的方式给ai投喂大量语言的规律

```Plain Text
{"text": "给我生成一首有关秋天的诗歌。秋日早晨，清风拂面。\n金色的叶子，似火在燃烧。\n露珠晶莹，如珍珠般美丽。\n秋的气息，弥漫在空气中。\n余音袅袅，如鸟儿的歌唱。\n美丽的秋天，是大自然的馈赠。帮我想一些创意，给即将到来的公司年会准备节目。一些节目比如能否请一位表演者为我们表演一曲钢琴曲，或者请一位小提琴手为大家演奏一首古典曲目。如果想要画面更具有视觉冲击力，可以安排一个魔术师或者杂技演员的表演。另外，也可以设计一些小游戏或者有奖竞猜来增添活动的趣味性，这些小游戏可以和公司的文化、发展历程等相关。请问给我讲一个清净的法则。无为而治是一项清净的法则。即在处理问题时，不要强行干预，反而尽可能地减少干预，坚持自然的发展趋势。比如，让植物自然生长，照顾它们就行了，不需要过度地修剪和整齐地排列。类似地，让人们自由发展，而不是通过繁琐的管制、政策干涉等方式，去要求、指导人们的行为。这种无为而治的法则旨在维持一个有秩序的、和平的治理状态，却不需要大规模的干预和指挥。"}
{"text": "根据以下输入的问题，生成一句话回答。\n你觉得寿司好不好吃？作为一名AI，我没有味觉，无法品尝食物，因此也没有对寿司是否好吃的判断。明白了，那请你回答一个问题，猫科动物里最凶猛的是哪种？猫科动物中最凶猛的应该是老虎。它们是世界上最大的猫科动物，可以长达3米，重达680千克，拥有锐利的爪子和牙齿，是非常强大的捕食者。"}
```

这里可以看出数据集只有一个text的提示符，紧跟有大量的文本，文本之间的逻辑性也不是很强

这是因为ai的重点在先会说话，因为原始的ai是一组随机出来的数。需要先在茫茫的数据空间中找出最优点的大概位置，对模型表达的细节并不关注。这时候只需要大概能阅读的文本都是可以的



代码

训练前预热：

负责配置环境，配置训练状态和模型状态，是否使用某些参数等设置

```Python
if __name__ == "__main__":
    #命令行指令
    parser = argparse.ArgumentParser(*description*="MiniMind Pretraining")
    parser.add_argument("--save_dir", *type*=str, *default*="D:\\Microsoft VS Code\\code\\minimind\\out", *help*="模型保存目录")
    parser.add_argument('--save_weight', *default*='pretrain', *type*=str, *help*="保存权重的前缀名")
    parser.add_argument("--epochs", *type*=int, *default*=2, *help*="训练轮数")
    parser.add_argument("--batch_size", *type*=int, *default*=32, *help*="batch size")
    parser.add_argument("--learning_rate", *type*=float, *default*=5e-4, *help*="初始学习率")
    parser.add_argument("--device", *type*=str, *default*="cuda:0" if torch.cuda.is_available() else "cpu", *help*="训练设备")
    parser.add_argument("--dtype", *type*=str, *default*="bfloat16", *help*="混合精度类型")
    parser.add_argument("--num_workers", *type*=int, *default*=8, *help*="数据加载线程数")
    parser.add_argument("--accumulation_steps", *type*=int, *default*=8, *help*="梯度累积步数")
    parser.add_argument("--grad_clip", *type*=float, *default*=1.0, *help*="梯度裁剪阈值")
    parser.add_argument("--log_interval", *type*=int, *default*=100, *help*="日志打印间隔")
    parser.add_argument("--save_interval", *type*=int, *default*=1000, *help*="模型保存间隔")
    parser.add_argument('--hidden_size', *default*=768, *type*=int, *help*="隐藏层维度")
    parser.add_argument('--num_hidden_layers', *default*=8, *type*=int, *help*="隐藏层数量")
    parser.add_argument('--max_seq_len', *default*=340, *type*=int, *help*="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', *default*=0, *type*=int, *choices*=[0, 1], *help*="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", *type*=str, *default*="D:\\Microsoft VS Code\\code\\minimind\\dataset\\pretrain_t2t_mini.jsonl", *help*="预训练数据路径")
    parser.add_argument('--from_weight', *default*='none', *type*=str, *help*="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', *default*=0, *type*=int, *choices*=[0, 1], *help*="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", *action*="store_true", *help*="是否使用wandb")
    parser.add_argument("--wandb_project", *type*=str, *default*="MiniMind-Pretrain", *help*="wandb项目名")
    parser.add_argument("--use_compile", *default*=0, *type*=int, *choices*=[0, 1], *help*="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, *exist_ok*=True)
        #记录了关键参数，稍后创建模型时会把它传进去。
    lm_config = MiniMindConfig(*hidden_size*=args.hidden_size, *num_hidden_layers*=args.num_hidden_layers, *use_moe*=bool(args.use_moe))
        #检查记录点，决定是否续训
    ckp_data = lm_checkpoint(lm_config, *weight*=args.save_weight, *save_dir*='D:\\Microsoft VS Code\\code\\minimind\\checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(*dtype*=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(*project*=args.wandb_project, *name*=wandb_run_name, *id*=wandb_id, *resume*=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, *device*=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, *max_length*=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None #分布式计算模式相关配置
    scaler = torch.amp.GradScaler(*enabled*=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), *lr*=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)#优化策略
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, *device_ids*=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch) #多卡同步
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()#单卡随机
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, *batch_sampler*=batch_sampler, *num_workers*=args.num_workers, *pin_memory*=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
```



训练程序，这里调用了模型训练框架，计算了损失和进行了反向传播。

这里也负责保存模型，显示训练进度等操作

```Python
def train_epoch(*epoch*, *loader*, *iters*, *start_step*=0, *wandb*=None):
    start_time = time.time()
    last_step = *start_step*
    for step, (input_ids, labels) in enumerate(*loader*, *start*=*start_step* + 1):#标记这里，不是很明白enumerate是干什么的
        print(input_ids.shape)
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        #调整学习率
        lr = get_lr(*epoch* * *iters* + step, args.epochs * *iters*, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        #混合精度训练
        with autocast_ctx:
            res = model(input_ids, *labels*=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps#梯度累计

        scaler.scale(loss).backward()#放大损失，防止梯度下溢，因为低精度可能会损失小的梯度值

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(*set_to_none*=True)

        # 计算并记录损失、学习率、预计剩余时间等
        if step % args.log_interval == 0 or step == *iters*:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - *start_step*, 1) * (*iters* - step) // 60
            Logger(f'Epoch:[{*epoch* + 1}/{args.epochs}]({step}/{*iters*}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if *wandb*: *wandb*.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # 保存模型检查点
        if (step % args.save_interval == 0 or step == *iters*) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, *weight*=args.save_weight, *model*=model, *optimizer*=optimizer, *scaler*=scaler, *epoch*=*epoch*, *step*=step, *wandb*=*wandb*, *save_dir*='D:\\Microsoft VS Code\\code\\minimind\\checkpoints')
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    if last_step > *start_step* and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(*set_to_none*=True)

```

## 微调

## 强化学习

# 数据流解析



![未命名笔记 \(1\)\_20260418\_224517\_31089\_1\.png](/assets/minimind/数据流.png)

