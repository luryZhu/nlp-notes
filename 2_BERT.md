# BERT

[李宏毅课程 ](https://www.bilibili.com/video/av94522844/?p=17) | [PPT](http://speech.ee.ntu.edu.tw/~tlkagk/courses/DLHLP20/BERT%20train%20(v8).pdf)



**Pre-train**：在正式处理任务之前，先让机器阅读大量的未标注的文本进行预训练，得到一个能够**理解**这门语言的模型

**Fine-Tune**：面对具体的任务，用一些任务相关的带标注的数据对模型进行微调

<img src=".\pics\2\1.png" style="zoom:50%;" />

BERT就是一个很典型的可以读人类语言的一个pre-train model

## Pre-train Model

### Word Embedding

将每一个文字token表示为向量

#### 不考虑上下文

不考虑上下文，相同的单词对应相同的向量，相关技术有：

- Word2vec
- Glove
- [FastText](https://blog.csdn.net/sinat_26917383/article/details/54850933)：针对英文，解决单词过多的问题
- [Su, et al., EMNLP’17]：针对中文，通过图像识别偏旁部首

![](.\pics\2\2.png)

#### 考虑上下文 Contextualized

Contextualized Word Embedding

考虑上下文，先输入整个句子，encoder+decoder

相关技术有：

- LSTM
- Self-attention layers
- [Tree-based model](https://youtu.be/z0uOq2wEGcc )：在处理文法结构非常清晰（例如数学式）的任务时特别有用，但是其他情况比不上LSTM

### 怎样让BERT变小

BERT模型几乎不适合GPU，规模很大的BERT一般是大公司用

- Megatron
- Turing NLG

#### Compress BERT

规模较小的BERT有：

- Distill BERT [Sanh, et al., NeurIPS workshop’19]
- Tiny BERT [Jian, et al., arXiv’19] 
- Mobile BERT [Sun, et al., ACL’20] 
- Q8BERT [Zafrir, et al., NeurIPS workshop 2019] 
- **ALBERT** [Lan, et al., ICLR’20]：最知名

压缩BERT模型的相关技巧参考：

[All The Ways You Can Compress BERT](http://mitchgordon.me/machine/learning/2019/11/18/all-the-ways-to-compress-BERT.html)

#### Network Architecture

改进BERT的网络架构，也能使BERT变小，使机器能阅读更长的文章

- Transformer-XL: Segment-Level Recurrence with State Reuse [Dai, et al., ACL’19] 让机器可以读跨segment的token
- Reformer [Kitaev, et al., ICLR’20] 
- Longformer [Beltagy, et al., arXiv’20]

Reformer和Longformer是为了减少self-attention带来的计算量

## fine-tune

怎样根据任务相关的数据，来调整Pre-trained model？

### pre-train model 8类任务

怎样在pre-trained model加上一些东西，让它可以处理各种各样的NLP任务

依然是根据输入和输出对任务分成8类，逐一分析：

- 输入
  - 单输入
  - 多输入

- 输出
  - 输出一个分类
  - 输出每个token的分类
  - 输出从input中复制的sequence
  - 输出生成的sequence

#### 输入

单输入没有什么问题，考虑**多输入**的情况

例如：Search Engine任务需要输入Query和Document，NLI任务需要输入Premise和Hypothesis，也就是两个sequence，为了区分这两个sequence，需要用分隔符`[SEP]`将它们隔开

<img src=".\pics\2\3.png" style="zoom:50%;" />

#### 输出

##### one class

希望机器读完整个句子之后输出一个class

**思路1**：在pre-train的过程中要加入一个token `[CLS]`

- 机器读其他token输出代表这个token的word embedding
- 机器看到 `[CLS]`就输出与整个句子相关的embedding
- 把与整个句子相关的embedding丢进Classifier

**思路2**：没有`[CLS]`的情况

- 直接把所有输出的embedding都丢到任务相关的Classifier里

##### class for each token

例如任务相关模型用LSTM，把所有embedding读进去，每一个time step输出一个class

![](.\pics\2\4.png)

##### copy from input

例如抽取式QA任务，输入Query和Document，输出两个整数，分别代表答案在原文中的起始位置和结束位置

<img src=".\pics\2\5.png" style="zoom:50%;" />

**思路**：

任务相关的模型包含两个向量，分别用来侦测start和end的位置

侦测起始位置：

- 选择document对应的embedding
- 将他们与侦测起始位置的向量做点积dot product
- 结果丢尽Softmax，取最大

侦测结束位置同理

![](.\pics\2\6.png)

##### General sequence

怎么把pre-trained model用到seq2seq model里？

简单的做法，把pre-trained model看成encoder，Task Specific model看成decoder，但是这样子存在问题：Task Specific model并没有被预训练到，也就是decoder没有经过与训练，效果不好

**可行思路**：

在input sequence之后，加上一个分隔符`[SEP]`，之后就把model的输出作为下一个输入，直到输出`<EOS>`，这样pre-train model也可以当作decoder用

<img src=".\pics\2\7.png" style="zoom:50%;" />

# ==怎样fine-tune？==

**思路1：只fine-tune task-specific model**

- Pre-trained Model训练完以后就固定住了，变成了一个Feature Extractor，输入句子，输出embedding或feature

- feature丢进task-specific model，只fine-tune这部分

==**思路2：一起fine-tune**==

- 把pre-trained和task-specific拼在一起做成一个巨大的model
- pre-trained部分的参数已经训练过了，task-specific的部分是随机初始化的，因为pre-trained部分是主体，所以不太容易overfitting

<img src=".\pics\2\8.png" style="zoom:50%;" />

一般来说思路2的performance比较好

## Adaptor

一般采用思路2，将整个模型一起训练，但是会产生问题：

针对不同的task进行训练之后，pre-trained model部分的参数也会改变，也就是它们都变成了不同的model，会产生很大的存储开销

**改进思路：**

把Pre-trained model的一小部分（某几层）拿出来作为Adaptor，训练时只改变Adaptor部分的参数，其余部分保持不变，怎样就只需要存储Adaptor的部分了

![](.\pics\2\9.png)

参考文献：

[Houlsby, et al., ICML’19] : https://arxiv.org/abs/1902.00751

## Weighted Features

之前描述的做法都是把model的output丢进后续的任务

另一种做法：

- 对model某些layer的输出加权求和得到一个embedding

- layer乘上的权重参数可以看作是task-specific model的一部分learn出来

## 为什么用Pre-train + fine-tune

在training阶段，如果有pre-train过，会更快地fit

参考文献： https://arxiv.org/abs/1908.05620

---

# ==怎样Pre-train==

[李宏毅课程 ](https://www.bilibili.com/video/BV1EE411g7Uk?p=18) | [PPT](http://speech.ee.ntu.edu.tw/~tlkagk/courses/DLHLP20/BERT%20train%20(v8).pdf)

先pre-train好一个可以理解文字的模型，然后fine-tune到不同的任务上

pre-trained model要能完成的工作有：

- 输入token输出embedding vector
- embedding最好能考虑上下文contextualized
- 通常是unsupervised无监督学习

## Paper

### Context Vector (CoVe)

最早的pre-train文章

- Contextualized

- 不是用unsupervised，而是用Translation model来做
  - 用翻译任务的好处在于机器会重视整个句子，更能理解语义
  - 不好的地方在于需要很多paired data

self-supervised learning

也就是unsupervised learning，



