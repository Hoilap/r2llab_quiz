# 稀疏检索（BM25）脚本
import os
import json
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from beir.datasets.data_loader import GenericDataLoader
import time
# 数据集路径
data_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'scifact')
corpus, queries, qrels = GenericDataLoader(data_dir).load(split="test")


# 构建语料库列表
doc_texts = [corpus[doc_id]['text'] for doc_id in corpus]
doc_ids = list(corpus.keys())
num_docs = len(doc_texts)
num_queries = len(queries)



# 分词
doc_tokens = [doc.split() for doc in doc_texts]

# bm25接收一个由多个文档组成的语料库 (doc_tokens)。它会对这些文档进行内部处理，计算后续评分所需的统计信息
start_time = time.time()
bm25 = BM25Okapi(doc_tokens, k1=0.9, b=0.4 )
end_time = time.time()
print(f"BM25 模型初始化耗时: {end_time - start_time:.2f} 秒")


results = {}
start_time = time.time()
for qid, query in tqdm(queries.items()):
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    top_n = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)[:100]
    results[qid] = {doc_id: float(score) for doc_id, score in top_n}
end_time = time.time()
total_query_time = end_time - start_time
print(f"检索 {num_queries} 个查询总耗时: {total_query_time:.2f} 秒")



# 保存结果
os.makedirs('results', exist_ok=True)
with open('results/sparse_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print('BM25 检索结果已保存到 results/sparse_results.json')
