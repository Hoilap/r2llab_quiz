# 稀疏检索（BM25）脚本
import os
import json
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from beir.datasets.data_loader import GenericDataLoader

# 数据集路径
data_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'scifact')
corpus, queries, qrels = GenericDataLoader(data_dir).load(split="test")

# 构建语料库列表
doc_texts = [corpus[doc_id]['text'] for doc_id in corpus]
doc_ids = list(corpus.keys())
# 分词
doc_tokens = [doc.split() for doc in doc_texts]

bm25 = BM25Okapi(doc_tokens)

results = {}
for qid, query in tqdm(queries.items()):
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    top_n = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[:100]
    results[qid] = {doc_id: float(score) for doc_id, score in top_n}

# 保存结果
os.makedirs('results', exist_ok=True)
with open('results/sparse_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print('BM25 检索结果已保存到 results/sparse_results.json')
