# 稠密检索（Sentence Transformers + FAISS）脚本
import os
import json
import numpy as np
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
import faiss
import time

# 数据集路径
data_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'scifact')
corpus, queries, qrels = GenericDataLoader(data_dir).load(split="test")

# 加载语料文本和ID
doc_texts = [corpus[doc_id]['text'] for doc_id in corpus]
num_docs = len(doc_texts)
num_queries = len(queries)
doc_ids = list(corpus.keys())

# 加载模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 生成语料嵌入
start_time = time.time()
corpus_emb = model.encode(doc_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
end_time = time.time()
embedding_dim = corpus_emb.shape[1]
print(f"生成 {num_docs} 个文档嵌入耗时: {end_time - start_time:.2f} 秒")

# 构建 FAISS 索引
start_time = time.time()
index = faiss.IndexFlatIP(corpus_emb.shape[1])
index.add(corpus_emb)
end_time = time.time()
print(f"构建 FAISS 索引耗时: {end_time - start_time:.2f} 秒")

results = {}
start_time = time.time()
for qid, query in tqdm(queries.items()):
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(query_emb, 100)
    # D 是每个查询的前 100 个相似文档的相似度分数（这里是点积，因为用的是 IndexFlatIP）。
    # I 是每个查询对应的前 100 个文档在原始 doc_ids 列表中的索引（即文档的编号）。
    top_docs = [(doc_ids[i], float(D[0][rank])) for rank, i in enumerate(I[0])]
    results[qid] = {doc_id: score for doc_id, score in top_docs}
end_time = time.time()
total_query_time = end_time - start_time
print(f"检索 {num_queries} 个查询总耗时: {total_query_time:.2f} 秒")


# 保存结果
os.makedirs('results', exist_ok=True)
with open('results/dense_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print('Dense 检索结果已保存到 results/dense_results.json')
