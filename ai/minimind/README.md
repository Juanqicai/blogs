# MiniMind

关于MiniMind项目:[「大模型」2小时完全从0训练64M的小参数LLM！](https://github.com/jingyaogong/minimind?tab=readme-ov-file)

推荐木乔的视频：[【2025/Minimind】Only三小时！Pytorch从零手敲大模型，架构到训练全教程](https://www.bilibili.com/video/BV1T2k6BaEeC?spm_id_from=333.788.videopod.episodes&vd_source=8704f44a88fba43c396e2910ec5bbdc8)

（感谢木乔老师的精心准备和讲解，也感谢jingyaogong大佬辛苦准备的项目，感谢！）

这是一份minimind的学习总结，复盘。

这个项目主要学习模型主体结构以及一部分训练的知识。

所包含的知识点可以看下面索引速查~

## 索引

- [Python 基础代码](./python基础代码.md) —— PyTorch 基础方法整理（arange、cat、Dropout、Linear 等）
- [RMSNorm](./RMSNorm.md) —— 均方根归一化
- [RoPE + YaRN](./RoPE+YaRN.md) —— 旋转位置编码及其优化扩展
- [Attention（GQA）](./attention.md) —— 多头注意力，MHA → GQA，mask 与位置编码
- [激活函数](./激活函数.md) —— 各类激活函数总结
- [Feed Forward Network（FFN）](./ffn.md) —— 前馈神经网络，负责信息推理
- [Transformer 拼接](./transformer.md) —— 残差连接、Block 堆叠、模型封装
- [预训练](./预训练+微调.md) —— 训练代码解读

目前minimind的笔记主要集中在transformer和其相关的知识点上。

有关训练，tokenizer还有交互终端等其他的代码涉及较少。

建议可以先看[Transformer 拼接](./transformer.md)，里面会逐个索引到其他目录。

后面的[预训练](./预训练+微调.md)涉及较少也相对独立。

如果又看不懂的python基础代码也可以看看[Python 基础代码](./python基础代码.md)，不过我觉得现在直接问ai效果也一样。

## 感想

学的时候还没有接触ai智能体，现在感觉有点太死磕底层了，早知道其实可以不用那么无聊的去搞这些。

huggingface的transformer库封装了更加方便和高效，实际使用也可以直接交给ai来处理。

我应该不太可能接触到如此底层的调优或者其他修改吧？（大概）。

不过也还不错，至少学的那段时间还是很纯粹的，没有其他心思打扰。整理出来东西后感觉很舒服。
