# Attention（GQA）

多头注意力，通过注意力机制让大模型自主决定信息的权重并提取关键的信息。

这个就是transformer不同于传统网络的核心。

这一部分东西较多，下面仔细拆解。

## 理论

attention的公式展示

$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V$

attention会将一个输入同时做3个不同的线性映射。这三个线性映射分别命名为Q,K,V（query，key，value）。

这里的Q是查询头，K是键，V是权重。

attention的实现逻辑大致就是用q查询k，然后将最契合的值从v中提取出来。

---

这里就可以看公式了

$\frac{QK^T}{\sqrt{d_k}}$这就是做查询。qk相乘之后就会得到不同位置的张量的重要程度。除以根号下k的个数来控制方差。

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

实际上仅仅这样还不足以写出完整注意力，仍然有一些细节需要处理，详看补充。

---

## MHA → GQA

以上就是基本attention（MHA）的原理，但是这样的attention计算量十分巨大。也会产生巨大的kv\_cache（历史记忆）。

因此为了减少计算量，就产生了MHA的优化版本GQA。

GQA相比MHA，减少了kv头的个数，采用多个q共享一个kv的策略。

实际效果和MHA相近，可以减少kv\_cache，优化显存占用。

### 为什么这么做有效？

或许可以理解为对同一片资源，可以使用不同的方式查询。多角度询问来减少信息的损失，提取更加丰富的信息。

### 关键参数

`h`：查询头数（Query heads）

`g`：KV 头数（Key/Value heads，即分组数）

组内 Q 头数：`h/g`（必须整除）

极端情况：

- g=1 → 退化为 **MQA**
- g=h → 退化为 **MHA**

实际工程上验证大概4个q对应1个kv是一个最佳参数

## 补充1：mask

在实际的训练上，因为大模型需要根据以前信息预测下一个字，所以不能让模型看见未来信息。就需要在模型训练的时候添加一个mask掩码，将未来信息遮盖住。

实际推理的时候需要和训练保持一致，所以mask需要一直存在。

mask通常会加在qk相乘之后，将当前位置之后的数值设为负无穷，这样就可以在softmax之后被计算为0。

ps：这里的mask必须放在softmax之前，在softmax之后直接设0会导致原本的和为1关系被破坏。

![3Blue1Brown transformer注意力讲解.png](../../assets/minimind/3Blue1Brown%20transformer注意力讲解.png)

## 补充2：位置编码（RoPE / YaRN）

attention的输入token都是相互独立的，需要给token添加位置信息，详见 [RoPE + YaRN](./RoPE+YaRN.md)，这里不再展开。

## Attention 代码和解读

### repeat_kv：复制多个k和v

```Python
#复制多个k和v，让kv数量和q相等，做兼容
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    #expand复制，相对于直接相乘，会节省显存。这样子操作并不会真的多复制出来维度。
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))
```

这里要复制k和v，让他们的数量和q对等起来。

否则后面做点积的时候会因为数量不对而报错。

> 位置编码部分 `precompute_freqs_cis` 和 `apply_rotary_pos_emb` 的代码在 [RoPE + YaRN](./RoPE+YaRN.md#代码实现) 中已给出。

### Attention 类：初始化

```Python
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        #设定头相关参数
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        #计算需要复制kv的倍数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        #头的维度
        self.head_dim = config.head_dim
        #因果掩码设定
        self.is_causal = True
        #qk矩阵，由于本质就是线性映射，所以可以一个用线性层一次性全部初始化，之后再修改形状。
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        #最后输出前的线性层
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        #归一化
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        #这些是其他处理，放在计划拼接的部分
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        #计算加速的库，知道就好了，计算点乘用的时候带上。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
```

初始化

### Attention 类：核心计算过程

```Python
def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        #提取输入矩阵的形状，transformer的数据流有展示数据和变量的变化，可以对照。
        bsz, seq_len, _ = x.shape
        #这里做线性映射
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        #这里的xq，xk，xv就包含了多头的结果（前面初始化把多头一次性全部包括了）。
        #这里修改形状，方便后面点积。
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        #xq，xk过一次归一化，稳定数据。
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        #为xq和xk添加位置信息，在这里添加是因为xk之后会保存下来成为记忆。xq也需要根据位置信息做计算。
        #这里的position_embeddings是是前面预计算的结果
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        #如果有kv缓存
        if past_key_value is not None:
            #把历史kv和当前kv拼接，这样子attention就可以关注以前的信息
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        #把处理后的kv作为输出，相当于整理的一下内存，并且把现在的kv也加入到了缓存里面。
        #这样子内存分配连续，更有利于gpu计算
        past_kv = (xk, xv) if use_cache else None
        #变换维度，并且匹配qkv的数量
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        #使用优化后的库
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            #计算注意力权重
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            #因果掩码
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            #遮盖占位符。比如有些句子没法占满最大长度，就会用占位符占满
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            #将注意力权重和v点乘，得到最后的结果，然后做其他处理
            #这里对注意力权重做了一次dropout，这里的dropout主要是为了防止模型产生过度依赖，增强鲁棒性
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
```

核心计算过程
