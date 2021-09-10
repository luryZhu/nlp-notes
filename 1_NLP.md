# NLP概述

[李宏毅课程 ](https://www.bilibili.com/video/BV1EE411g7Uk?p=16) | [PPT](http://speech.ee.ntu.edu.tw/~tlkagk/courses/DLHLP20/TaskShort%20(v9).pdf)

## 俺输入分类

NLP任务可以按照输出分为2类

| 模型    | 输入 | 输出      |
| ------- | ---- | --------- |
| BERT    | 文字 | 类别class |
| Seq2seq | 文字 | 文字      |

- 输入文字，输出class **BERT**

  ![](.\pics\1\1.png)

- 输入文字，输出文字 **seq2seq**

  编码器-解码器 + 注意力机制 + [复制机制]

  ![](.\pics\1\2.png)

## 任务种类

考虑输入单输入和多输入的情况，可以把任务分为8类：

| 输出\输入                                     | One Sequence                                                 | 单输入       | Multiple Sequences                          | 多输入         |
| --------------------------------------------- | ------------------------------------------------------------ | ------------ | ------------------------------------------- | -------------- |
| **One Class<br>输出整个句子分类**             | [Sentiment  Classification](#Sentiment Classification 情感分析) | 情感分类     | [NLI](#NLI)                                 | 自然语言推理   |
|                                               | [Stance Detection](#Stance Detection)                        | 立场侦测     | [Search Engine](#Search Engine)             | 搜索引擎       |
|                                               | [Veracity Prediction](#Veracity Prediction)                  | 可信度预测   | [Relation Extraction](#Relation Extraction) | 关系抽取       |
|                                               | [Intent Classification](#Intent Classification)              | 意图分类     |                                             |                |
|                                               | [Dialogue Policy](#Policy & State Tracker)                   | 对话策略     |                                             |                |
| **Class for each Token<br>输出每个token分类** | [POS tagging](#POS tagging)                                  | 词性标注     |                                             |                |
|                                               | [Word segmentation](#Word segmentation)                      | 词切分       |                                             |                |
|                                               | [Extractive Summarization](#Extractive Summarization)        | 摘要抽取     |                                             |                |
|                                               | [Slot Filling](#Slot Filling)                                | 槽填充       |                                             |                |
|                                               | [NER](#NER)                                                  | 命名实体识别 |                                             |                |
| **Copy from Input<br>直接复制输入**           |                                                              |              | [Extractive QA](#Extractive QA)             | 抽取式问答     |
| **General Sequence<br>输出文字**              | [Abstractive Summarization](#Abstractive Summarization)      | 摘要生成     | [General QA](#General QA )                  | 生成式问答     |
|                                               | [Translation](#Machine Translation)                          | 翻译         | [Task Oriented Dialogue](#Task Oriented)    | 任务导向型对话 |
|                                               | [Grammar Correction](#Grammar Error Correction)              | 语法校正     | [Chatbot](#)                                | 聊天机器人     |
|                                               | [NLG](#NLG)                                                  | 自然语言生成 |                                             |                |
| **Other?**                                    | [Parsing](#Parsing)                                          | 语法分析     |                                             |                |
|                                               | [Coreference Resolution](#Coreference Resolution)            | 指代消解     |                                             |                |

![](.\pics\1\27.png)

下面逐一介绍表中的任务

### POS Tagging

词性标注

为句子中每一个词标注词性，有助于下游的任务（例如翻译）更好地理解句子

![](.\pics\1\3.png)

### Word Segmentation

词切分

对中文而言，词是由多个字组成，在句子中找到词汇的边界

![](.\pics\1\4.png)

---

其他类

### Parsing

语法分析

产生一个语法树，通常是作为一个额外的输入送到后续的模型

![](.\pics\1\5.png)

### Coreference Resolution

指代消解

找出文章里那些词汇是指代同一个entity，可见[demo](https://demo.allennlp.org/coreference-resolution/)，一般也会先做指代消解，再把结果输入到后续任务

![](.\pics\1\6.png)

---

### Summarization

摘要

#### Extractive Summarization

摘要抽取

直接从文章里提取一些句子作为摘要

![](.\pics\1\7.png)

---

seq2seq

#### Abstractive Summarization

摘要生成

机器用自己的话总结文章，seq2seq

![](.\pics\1\8.png)

### Machine Translation

机器翻译

文字对文字，语音对文字，语音对语音，主要用半监督学习

![](.\pics\1\9.png)

### Grammar Error Correction

语法校正

可以用seq2seq+copy mechanism来做，但是有点杀鸡用牛刀

存在更简化版的方法，选择输出每个token的class，class对应三种操作：copy，replace，assert

![](.\pics\1\10.png)

---

### Sentiment Classification 

情感分析

输出文本是正面情绪还是负面情绪

![](.\pics\1\11.png)

### Stance Detection

立场侦测

例如针对一则推文，分析其回复的立场

立场分为4类： Support, Denying, Querying, and Commenting (SDQC) ，常被用于Veracity Prediction分析可信度

![](.\pics\1\12.png)

### Veracity Prediction

可信度预测

例如针对一则新闻，根据其回复，或者是wiki上的内容，分析可信度

![](.\pics\1\13.png)

### NLI

Natural Language Inference 自然语言推理

输入一个premise前提和一个hypothesis假设，判断能否从这个前提推出这个假设，输出3个class: contradiction矛盾，entailment蕴含，neutral中性

![](.\pics\1\14.png)

### Search Engine

搜索引擎

输入检索的句子和一些文章，输出检索内容和文章是否相关，根据相关度对文章进行排序，[参考](https://www.blog.google/products/search/search-language-understanding-bert/)

![](.\pics\1\15.png)

### Question Answer 

(QA) 问答

早期的QA：有非常复杂的model，资料库是系统化结构化的，如Watson

![](.\pics\1\16.png)

#### Reading Comprehension

阅读理解

现在的QA：输入问题和知识库，输出答案，通常这个知识库很可能是从search engine来的

从搜索引擎获取没有结构的文章，让机器阅读（过滤掉一些无关文章），然后回答问题，即阅读理解

![](.\pics\1\17.png)

#### Extractive QA

抽取式问答

答案就在文章里面，需要比较强的copy mechanism，直接输出文章里答案的位置

![](.\pics\1\18.png)

### Dialogue

对话

#### Chatbot

聊天机器人

输入过去对话的记录，输出新的回复，还可以考虑回复的性格、有同理心的回复、有知识的回复

![](.\pics\1\19.png)

#### Task Oriented

任务导向的会话

例如在订票、定餐厅、订旅馆的场景下进行对话

- 实现定义一组action，限定机器可以做的事情
- 输入一历史对话，输出要采取哪一个action
- 将action输入NLG模型，生成一个句子

![](.\pics\1\20.png)

整个Task Oriented的对话可以拆分成很多子模块

![](.\pics\1\23.png)

#### Policy & State Tracker

状态跟踪器

用state储存当前已经得到的信息，根据policy来决定下一个action（可以是简单的classification，也可以是reinforcement learning）

![](.\pics\1\21.png)

### NLU

Natural Language Understanding 自然语言理解

#### Intent Classification

意图分类

输入句子，输出分类（意图：例如提供信息，或询问），通常就把结果丢给State Tracker

#### Slot Filling

槽填充

预先定义一些slot，然后判断句子里面的token属于哪一个slot，其实是类似于POS tagging

![](.\pics\1\21.png)

---

### Knowledge Graph

知识图谱

从文章中抽取entity实体和relation关系

![](.\pics\1\24.png)

#### NER

Name Entity Recognition 命名实体识别，抽取entity

类似于POS tagging和slot filling，判断句子的每一个token属于什么name entity，也就是我们研究的一些实体，取决于实际应用

然后需要通过entity linking把相关的entity都关联起来才能形成entity

![](.\pics\1\25.png)

#### Relation Extraction

关系抽取，抽取 relation

输入句子和句子中关注的两个entity，输出这两个entity的关系，也可以输出们关系

![](.\pics\1\26.png)

# 评估NLP模型

## GLUE

General Language Understanding Evaluation (GLUE)

用来评估机器理解人类语言的能力

[英文版](https://gluebenchmark.com/) | [中文版](https://www.cluebenchmarks.com/)

- Corpus of Linguistic Acceptability (CoLA)
- Stanford Sentiment Treebank (SST-2) 
- Microsoft Research Paraphrase Corpus (MRPC) 
- Quora Question Pairs (QQP) 
- Semantic Textual Similarity Benchmark (STS-B) 
- Multi-Genre Natural Language Inference (MNLI) 
- Question-answering NLI (QNLI) 
- Recognizing Textual Entailment (RTE) 
- Winograd NLI (WNLI) 

[Super GLUE](https://super.gluebenchmark.com/)

增强版，主要是关于QA的评估

## DecaNLP

[DecaNLP](https://decanlp.com/)

有10个NLP任务，期待用同一个模型去解，比如说都看成QA问题，就用QA去解