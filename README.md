## Project structure
```
|- R2LLAB_QUIZ
|   - datasets
|   - results
|   - download_scifact.py
|   - dense_retriever.py 
|   - sparse_retriever.py
|   - sparse_retriever_v2.py //add token preprocessing
|   - evaluation.py
|   - README.md //README_zh.md for draft
```
## Command to run the code
- `pip install -r requirements.txt`
- `python download_scifact.py` to download the dataset
- run `python sparse_retriever.py` and `python dense_retriever.py` to generate retriever results
- run `python evaluation.py datasets/scifact results/sparse_results.json` and `python evaluation.py datasets/scifact results/dense_results.json` to evaluate

## Result
### Dense retriever
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
### Sparse retriever
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
### Sparse retriever_v2 (self-implemented preprocessing)
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
   The Dense Retriever performed better.  
   - Ranking Quality (NDCG & MAP): Dense retriever’s NDCG@10 (0.6402) and MAP@10 (0.58873) are much higher than the sparse retriever’s NDCG@10 (0.50698) and MAP@10 (0.46348). This shows that the dense retriever not only finds relevant documents but also ranks the most relevant ones higher.  
   - Recall: Dense retriever’s Recall@100 (0.91667) is significantly higher than the sparse retriever’s Recall@100 (0.76517). This means that within the top 100 results, the dense retriever found about 92% of all relevant documents, while the sparse retriever found only about 77%.  

2. Why do you think that is?  
   DPR can capture semantic similarity, but BM25 do not.

3. What are the performance trade-offs you observed?  
   * Retrieval Quality: DPR outperforms BM25.  
   * Speed:  
     * Preprocessing stage: DPR requires training document embeddings, which is time-consuming, while BM25 initialization (mainly building inverted index and term frequency statistics) is very fast.  
     * Query stage: both are fast. DPR may appear faster due to OS scheduling (dataset is small) and the efficiency of FAISS.  
   * Memory:  
     * DPR: memory overhead comes from the model itself and the document embeddings (MiniLM-L-6 ~90MB).  
     * BM25: only requires statistics.  

4. Performance comparison with benchmarks  
Here are some retriever leaderboard references we used for comparison^[https://docs.google.com/spreadsheets/d/1L8aACyPaXrL8iEelJLGqlMqXKPX2oSP_R10pZoy77Ns/edit?gid=0#gid=0, https://eval.ai/web/challenges/challenge-page/1897/leaderboard/4475]  

| Source | Model | Model Name / Parameters | Trained on | SciFact (NDCG@10) |
| :--- | :--- | :--- | :--- | :--- |
| **BEIR Leaderboard** | BM25 | `Elasticsearch` | - | 0.62 |
| **Local Experiment** | BM25 (Local) | `BM25Okapi` | - | 0.507 |
| **Local Experiment_v2** | BM25 + Preprocessing | `BM25Okapi` | - | **0.63905** |
| **BEIR Leaderboard** | MiniLM-L-6 | `msmarco-MiniLM-L-6-v3` | MSMARCO | 0.495 |
| **Local Experiment** | MiniLM-L-6 (Local) | `all-MiniLM-L-6-v2` | General/NLI Data | **0.640** |

- Why do MiniLM-L-6 models perform so differently?  
  `all-MiniLM-L-6-v2` is trained on diverse datasets across multiple domains (NLI/STS/paraphrase/translation/QA pairs), while `msmarco-MiniLM-L-6-v3` is trained only on MS MARCO query–passage pairs. The broader training of `all-MiniLM-L-6-v2` leads to better embeddings.  
- BM25 benefits significantly from text preprocessing.  
  - Text preprocessing differences: Elasticsearch performs tokenization, lowercasing, stemming by default; BM25Okapi (from rank_bm25 library) requires manual tokenization and does not perform stemming/stopword removal by default.  
  - Our ablation experiments comparing stemming + lowercasing + punctuation removal showed that preprocessing brought the performance close to benchmark results, confirming our implementation was correct.  
