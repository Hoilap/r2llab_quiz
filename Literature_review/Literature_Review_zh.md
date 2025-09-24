## seperate analysis

### Sparse Retrieval

#### **What are the authors trying to do?**

  基于相关性的检索权重计算方法，用于改进文档排序（文章假设文章相关性得分为权重之和）

#### **How was it done prior to their work, and what were the limits of current practice?**

  在此之前：使用的是 频率或布尔模型（term frequency / Boolean retrieval）。

- Boolean 模型：只考虑词是否出现（0/1），无法反映不同词的重要性。
- TF 模型：简单用词频作为权重，但没有考虑到 相关文档与非相关文档的区分。
  局限：
- 没有直接利用“相关性反馈”（relevance feedback）的信息。
- 无法动态调整词权重来优化排序。所有的查询，一个术语在不同的检索请求中会有相同的权重。而当考虑到相关文档时，同一个术语在不同请求中可能会有不同的权重。

#### **What is new in their approach, and why do they think it will be successful?**

  创新点：提出了 基于相关文档与非相关文档统计的权重公式（F1-F4）（即后来 BM 系列公式的雏形）。
  权重函数依赖于：

* **N**：文档库中**所有文档数**。
* **R**：对当前查询 $q$ 来说，**相关文档数**。
* **n**：包含某个词项 $t$ 的文档数。
* **r**：既相关又包含词项 $t$ 的文档数。
  值得注意的是，在估计R/r时，当依赖用户的相关性反馈Relevance Feedback，需要使用修正估计（+0.5偏置）。

$$
w_t \approx \log \frac{(r+0.5)/(R-r+0.5)}{(n-r+0.5)/(N-n-R+r+0.5)}(4)
$$

    对于查询$Q$（包含若干查询词 $q_i$）和文档 $D$，BM25 的打分通常写成：

$$
\text{score}(D,Q) = \sum_{q_i \in Q} \text{IDF}(q_i)\cdot \frac{f(q_i,D)\,(k_1+1)}{f(q_i,D) + k_1\cdot\big(1 - b + b\cdot\frac{|D|}{\text{avgdl}}\big)}
$$

其中：

* $f(q_i,D)$：查询词 $q_i$ 在文档 $D$ 中的词频（term frequency）；
* $|D|$：文档长度（通常以词数计）；
* $\text{avgdl}$：语料库中所有文档的平均长度；
* $k_1$ 与 $b$：可调超参数，分别控制**词频饱和（saturation）强度**与**长度归一化**的程度；k1不应该是线性的，因为一个词在一个文档中出现10次和出现20次，对相关性的贡献增加是边际递减的。b用于惩罚模型检索过长的文档，因为过长的文档更容易被检索到。
* $\text{IDF}(q_i)$：逆文档频率，常用 Robertson–Spärck Jones (RSJ) 的平滑形式：即公式4

#### **What are the mid-term and final “exams” to check for success?** (i.e., How is the method evaluated?)

1. 作者对比了F1-F4的Recall-Precision Curve。F4最佳
2. Miller的实验使用了作者提出的F1权重公式，并将其应用于一个真实的、大型的MEDLARS医学文献数据库。对比两种搜索方法：Probabilistic Search和Probabilistic Search， 通过对比precision和recall来对比

### Dense Retrieval

#### **What are the authors trying to do?** Articulate their objectives.

Karpukhin 等人在 DPR 中的目标是：用学习得到的密集向量表示（dense embeddings）替代传统稀疏词袋模型（如 BM25），以便捕捉词义、同义替换和语义匹配，从而在开放域 QA /段落检索任务中检索到更相关的段落，提升下游问答的准确率。

#### **How was it done prior to their work, and what were the limits of current practice?**

以前的IR：

- sparse: BM25
- dense: ORQA (Lee et al., 2019) 提出一种复杂的逆完形填空任务 (ICT) 目标，即预测包含被掩盖句子的块。ORQA优于BM25
- 使用MIPS算法对向量进行高效检索

limit: 1. ICT pretraining is computationally intensive 2. because the context encoder is not ﬁne-tuned using pairs of questions and answers, the corresponding representations could be suboptimal.
额外的预训练

#### **What is new in their approach, and why do they think it will be successful?**

在适当的训练设置下，仅仅对现有的问题-段落对 question-paragraph pair进行微调即可大幅超越BM25。额外的预训练可能并不是必要的。训练后可以通过向量索引（FAISS 等）做检索
验证了IR更高精度可以提高QA的准确性
使用负类采样方法Gold，采样技巧In-batch negatives：In-batch negatives巧妙地将在同一个batch内的其他段落作为当前问题的负样本。这些负样本通常与正样本在主题上有一定相似性（因为它们可能来自同一个语料库），因此是困难负样本（hard negatives）。

#### **What are the mid-term and final “exams” to check for success?**

检索层面的评估：top-k retrieval accuracy

End-to-end QA 评估：将检索器与阅读/生成模型联用，评估 QA 的 (Exact Match) Accuracy，和ORQA和REALM进行比较，体现出DPR仅通过qustion-answer对进行训练

### RAG

#### **What are the authors trying to do?**

把retriever和generator结合起来，动态检索相关文档提供额外的上下文。
这篇论文给出RAG的模型架构和训练方法。Langchain是RAG的工程化实现

#### **How was it done prior to their work, and what were the limits of current practice?**

1. 生成模型仅依靠模型参数，容易出现幻觉，
2. 依靠retriever+reader的组合，reader只会生成文本跨度，缺乏泛化性

#### **What is new in their approach, and why do they think it will be successful?**

这篇文章将retriever和generator结合起来，提供了非参数优化的LLM提升方法
有两种范式：RAG-Sequence/RAG-Token

#### **What are the mid-term and final “exams” to check for success?**

Open-domain QA/ Open-domain QA/ Jeopardy Question Generation / Fact verification
使用Exact Match (EM) scores进行评估

## concluding section

#### **Who cares? What difference does the author's results make?**

Who cares? 做检索/QA的人会关注
Relevance weighting od Search term: 利用相关性信息为搜索词赋予权重的统计技术，从而提升检索性能
DPR：DPR通过使用双编码器架构，将问题和文本段落编码到同一个高维语义空间中，通过计算向量的距离（如点积）来判断相关性。
RAG：将retriever和generator结合起来，将其提供给LLM作为生成答案的依据，降低模型幻觉

#### **What are the risks?** What are the potential failure modes or downsides of these approaches?

- Relevance Weighting：它完全依赖于词汇的字面匹配。如果用户的提问和知识库中的文档使用了不同的词语来描述同一个概念（同义词、近义词），BM25就可能找不到相关的文档。
- DPR: 训练DPR模型需要大量的GPU资源和高质量的标注数据；领域泛化问题需要重新训练；对关键词不敏感
- RAG：这种retriever+generator的范式没有什么downside。
  - 目前的检索器以文本检索为主。tabular data如何检索，多模态文档如何转换成存储为向量?
  - RAG 的本质是搜索，它能工作的前提在于根据用户的提问，可以 “搜” 到答案所在。但是在很多情况下，这个前提不存在，例如一些意图不明的笼统提问，以及一些需要在多个子问题基础之上综合才能得到答案的所谓 “多跳” 问答，这些问题的答案，都无法通过搜索提问来获得答案，因此提问和答案之间存在着明显的**语义鸿沟**。（目前的论文：graphRAG，利用LLM生成跨chunk的关联信息）
    https://blog.csdn.net/sinat_39620217/article/details/147386100

#### **Synthesis:** Briefly explain how these three technologies fit together. How do sparse and dense retrieval support the RAG framework? What are the pros and cons of using one retrieval method over the other in a RAG system?

他们是递进的关系
BM25和DPR都是检索器，区别是一个使用统计信息，另一个使用稠密向量。RAG由retriever和generator构成
dense retriever相比sparse retriever

- pro: 捕捉语义联系（即使两个词并不相同），比BM25有更优的性能
- cons: 消耗训练成本
  现实应用中，我们可以使用混合检索，将BM25得分和DPR得分，甚至是知识图谱的实体/关系/社区摘要以某种方式结合起来产生最终的排序，减少语义鸿沟。


RAG中文综述参考链接见onenote
