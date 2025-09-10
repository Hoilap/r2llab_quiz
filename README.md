## command
- pip install -r requirements.txt
- python download_scifact.py 下载数据集。
- 运行 python sparse_retriever.py 和 python dense_retriever.py 生成检索结果。
- 用 python evaluation.py datasets/scifact results/sparse_results.json 和 python evaluation.py datasets/scifact results/dense_results.json 评测。
## result
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
```
```
Evaluating results from results/sparse_results.json...

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
```
各种模型的retriever的榜单参考 
* `https://docs.google.com/spreadsheets/d/1L8aACyPaXrL8iEelJLGqlMqXKPX2oSP_R10pZoy77Ns/edit?gid=0#gid=0`
* `https://eval.ai/web/challenges/challenge-page/1897/leaderboard/4475`