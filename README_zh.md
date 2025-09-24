* Includes a brief discussion of your results. Which retriever performed better? Why do you think that is? What are the performance trade-offs (e.g., speed, memory, retrieval quality) you observed?

## Command to run the code
- `pip install -r requirements.txt`
- `python download_scifact.py to` download the dataset
- run `python sparse_retriever.py` and `python dense_retriever.py` to generate the retriever result。
- run `python evaluation.py datasets/scifact results/sparse_results.json` and `python evaluation.py datasets/scifact results/dense_results.json` to evaluate.

## Result
### dense retriever
```
Evaluating results from results/dense_results.json...

Evaluation Scores:
[
    {
        //NDCG (Normalized Discounted Cumulative Gain) 衡量前 k 个检索结果的排序质量。
        "NDCG@10": 0.6402,
        "NDCG@100": 0.66904
        //MAP (Mean Average Precision)定义：对每个查询，先算它的 AP (Average Precision)，再取所有查询的平均。AP会把所有P@k都算一遍
        "MAP@10": 0.58873,
        "MAP@100": 0.59504
    },
    {
        //Recall@k 表示在前 k 个结果中，找回了多少比例的相关文档。
        "Recall@10": 0.78667,
        "Recall@100": 0.91667
    },
    {
        //P@k 表示前 k 个结果中有多少比例是相关文档。
        "P@10": 0.089,
        "P@100": 0.01047
    }
]

Embedding generation time: 192.64s,
Retrievering time: 4.94s
```
### sparse retriever
```
Evaluating results from results/sparse_results.json... //使用默认参数k1=1.5, b=0.75

Evaluation Scores:
[
    {
        "NDCG@10": 0.50698,
        "NDCG@100": 0.53738
    },
    {
        "MAP@10": 0.46348,
        "MAP@100": 0.46994
    },
    {
        "Recall@10": 0.62806,
        "Recall@100": 0.76517
    },
    {
        "P@10": 0.06867,
        "P@100": 0.0086
    }
]
Model initialization time: 0.42s,
Retrievering time: 6.92s
```
### sparse retriever_v2（分词预处理）
```
Evaluating results from results/sparse_results.json...

Evaluation Scores:
[
    {
        "NDCG@10": 0.63905,
        "NDCG@100": 0.67161
    },
    {
        "MAP@10": 0.59655,
        "MAP@100": 0.60396
    },
    {
        "Recall@10": 0.75639,
        "Recall@100": 0.90156
    },
    {
        "P@10": 0.08267,
        "P@100": 0.01007
    }
]
```
## Analysis
1. Which retriever performed better?
   稠密检索器 (Dense Retriever) 表现更好。
   排序质量 (NDCG & MAP): 稠密检索器的 NDCG@10 (0.6402) 和 MAP@10 (0.58873) 远高于稀疏检索器的 NDCG@10 (0.50698) 和 MAP@10 (0.46348)。这表明稠密检索器不仅能找到相关的文档，而且能将最相关的文档排在更高的位置。
   召回率 (Recall): 稠密检索器的 Recall@100 (0.91667) 显著高于稀疏检索器的 Recall@100 (0.76517)。这意味着在返回的前100个结果中，稠密检索器成功找回了全部相关文档中的约92%，而稀疏检索器只找回了约77%。
2. Why do you think that is?
    DPR can understanding semantic similarity.
3. What are the performance trade-offs you observed?
   * Retrieval Quality: DPR比BM25有更好的性能
   * Speed:
     * 在预处理阶段，DPR需要对文档embedding进行训练，耗时很长，而BM25的初始化（主要是构建倒排索引和计算词频统计）非常迅速。
     * 在查询阶段：两者都很快。DPR更快的原因可能是OS调度（数据集太小了）以及FAISS工具的高效性。
   * Memory:
     * DPR：LLM模型本身的内存占用和文档向量的内存占用（MiniLM-L-6 90MB, ）
     * BM25：仅统计信息
4. 实现性能比较
这是各种模型的retriever的榜单参考，我们从中摘录出相关项用于比较^[https://docs.google.com/spreadsheets/d/1L8aACyPaXrL8iEelJLGqlMqXKPX2oSP_R10pZoy77Ns/edit?gid=0#gid=0, https://eval.ai/web/challenges/challenge-page/1897/leaderboard/4475]



| 来源 (Source) | 模型 (Model) | 具体名称 / 参数 (Model Name / Parameters) | 训练数据 (Trained on) | SciFact (NDCG@10) |
| :--- | :--- | :--- | :--- | :--- |
| **BEIR 排行榜** | BM25 | `Elasticsearch` | - | 0.62 |
| **本地实验** | BM25 (Local) | `BM25Okapi` | - | 0.507 |
| **本地实验_v2** | BM25 + 分词处理 | `BM25Okapi` | - | **0.63905** |
| **BEIR 排行榜** | MiniLM-L-6 | `msmarco-MiniLM-L-6-v3` | MSMARCO | 0.495 |
| **本地实验** | MiniLM-L-6 (Local) | `all-MiniLM-L-6-v2` | 通用/NLI 数据 | **0.640** |

- 为什么同样是MiniLM-L-6，两种模型的性能有明显差别？all-MiniLM-L-6-v2训练数据来自多个领域的通用句子对（NLI/同义句对 / 翻译对 / QA 对）,而msmarco训练数据是MS MARCO查询–段落对。all-MiniLM-L-6利用更多复杂的数据从而生成更好的embedding.
- BM25在经过文本预处理后会有更好的性能
  - 文本预处理差异：Elasticsearch 默认会做：分词，小写化，词干化；BM25Okapi (rank_bm25 库)手动分词，默认不会做词干化/停用词过滤。
  - 我们做了消融实验，对比了是否进行词干化+小写化+去除标点对性能的影响。发现经过预处理后效果接近benchmark效果，说明实现正确。
- 多字段权重：BEIR 官方的 BM25 (Elasticsearch) 往往对文档的 title + abstract 等字段做加权检索。本地 BM25Okapi 实验里通常只用单字段文本。title 在检索里权重很大，加上它能显著提升效果。
