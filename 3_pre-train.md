# NLP

[视频](https://www.bilibili.com/video/BV1BQ4y1R7V7?from=search&seid=2508773552982905078&spm_id_from=333.337.0.0) | [课件](https://github.com/wangshusen/DeepLearning)

## 前置内容复习

### 数据基本处理

#### One-hot编码

例如单词表中有N个单词，创建一个N维向量来表示一个单词

- 第n个单词（从1开始）用第n位置1的向量表示$[0,0,...,1,...,0]$​

- 全0向量表示未知或缺失的数据

缺点在于单词表较大时会长生很大的存储开销

#### Word Embedding

使用特征化的向量进行词语的表示，用更低维度的特征向量代替原来的超高维的 one-hot 向量，典型的模型有

- word2vec
- Glove

#### 文本处理基本步骤

##### 1. Tokenization

将文本分割成单词列表，需要考虑的问题：

- 去掉大小写？一般要去掉，但是可能会造成歧义，例如“Apple”
- 去掉stop words？the, a, of, ...
- 纠正拼写错误

##### 2. 计算词频

遍历单词列表，维护一个字典，键值对为`(单词，词频)`，算法如下

- 单词w不在字典中，加入(w,1)
- 单词w在字典中，词频+1

结束后按照词频降序对单词进行排序并编号，即词频越大的单词序号越小

统计的目的就是去掉低频词，因为低频词可能是命名实体或拼写错误，删去可以有效减小vocabulary，减少后续的overfitting的问题

##### 3. One-Hot Encoding

根据单词表对文本进行One-hot编码，step2中删去的低频词可以直接忽略或对应全0编码，这一步将一个包含n个token的句子转换成长度为n的sequence，sequence每个元素就是token在字典中的编号

##### 4. Align Sequences

数据对齐，one-hot编码之后的数据长度不同，需要处理成相同的长度

- 超过的部分删去cut off
- 不满的部分用0补齐zero padding（在左侧补）

##### 5. word embedding

词嵌入，把encoding映射到向量空间，减少数据规模

### RNN

#### 基本RNN

对于小规模数据有效，但是缺点在于记忆能力很弱

![](.\pics\3\1.png)

- 每一个time step输入一个单词的embedding vector，在下一个time step，将上一个隐藏层状态$h_{t-1}$​和$x_t$​​一起作为输入送入神经网络
- 每一个time step输出隐藏层状态$h_t$​，包含前序输入的的信息
- 整个RNN只有一个参数$A$​，随机初始化并训练
- 激活函数为双曲正切函数，目的在于在每次循环之后把$h$的值回复到[-1,1]的区间里，防止最后输出的$h$过大或趋于0

#### LSTM

一般现在提到的RNN都是LSTM

可以避免梯度消失问题，可有更长的记忆

![](.\pics\3\2.png)

LSTM有个有4个input，除了输入的feature，还有增加了3个gate

- input gate：打开时，Neural的output才能被写入Memory
- output gate：打开时，外界才能读Memory的值
- forget gate：决定什么时候忘掉或者format掉memory里存储的内容

<img src=".\pics\3\3.png"  />

- 遗忘门：$f_t=\sigma(W_f\cdot[x_t,h_{t-1}])$
- 输入门：$i_t=\sigma(W_i\cdot[x_t,h_{t-1}])$​

- 输入：$\widetilde{C}_t=\tanh(W_c\cdot[x_t,h_{t-1}])$​​​​​
- memory cell：$C_t=f_t\circ C_{t-1}+i_t\circ \widetilde{C}_t$​  (哈达玛积，对应位相乘)
- 输出门：$o_t=\sigma(W_o\cdot[x_t,h_{t-1}])$​
- 输出：$h_t=o_t\circ \tanh(C_t)$ 哈达玛积

#### RNN优化

##### stacked 多层

多个RNN层，前一层的输出作为后一层的输入

<img src=".\pics\3\4.png" style="zoom:80%;" />

##### bidirectional 双向

训练两条独立的RNN，一条从前往后，一条从后往前输入sequence，各自输出自己的状态向量

![](.\pics\3\5.png)

- 多层RNN的情况，把每一个单词对应的time step输出连接起来送入下一层
- 单层RNN，只需保留两条RNN最后输出的两个状态向量

总是比单向的好，因为双向的机制不会遗忘最开始看到的词

##### pre-train 预训练

对embedding层进行预训练，在大型数据集上训练embedding层，可以用与目标任务类似的任务，固定住embedding层，输入到下游任务模型

### 应用

#### 文本生成

训练数据：从文本获取，例如对一段文本，feature和label可以是一个包含n个字符的segment和下一个字符，通过确定segment长度和步长，就能通过平移取得一组训练数据

文本生成的任务就可以转化成一个多分类问题，输入一个片段，输出一个概率分布

Sci-Gen

#### 机器翻译

常用的模型是Seq2Seq

[训练用数据集获取](http://www.manythings.org/anki/)

encoder+decoder模式

**encoder**

编码器，用来将输入的sequence压缩为一个向量

就是一个只保留最终隐藏层输出的LSTM

**decoder**

解码器，用来将向量展开为一个sequence

就是一个文本生成器

# Seq2Seq

[视频](https://www.bilibili.com/video/BV1JE411g7XF?p=51) | [PPT](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2017/Lecture/Attain%20(v5).pdf)

https://zhuanlan.zhihu.com/p/136597401

参考论文：

- Encoder-decoder：[﻿Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](./paper/Encoder-Decoder.pdf)

- seq2seq：[﻿Sequence to Sequence Learning with Neural Networks](./paper/seq2seq.pdf)

- 注意力：[Neural Machine Translation by Jointly Learning to Align and Translate](./paper/align-translate.pdf)

## encoder+decoder

以翻译任务为例，输入和输出都可能是不定长的，使用encoder+decoder机制可以将输入序列编码成==上下文变量$\mathbf c$==​，然后用解码器将$\mathbf c$​​展开为输出的序列

在训练集中：

- encoder对每个输入sequence，在句尾加上`<EOS>`表示句子结束
- decoder最初的输入`<BOS>`表示句子开始

![](https://pic2.zhimg.com/80/v2-b7dfd3998c67b4e70b2a700260538899_720w.jpg)

### encoder





注意力机制

