# rag/retriever.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, index_path="rag/faiss.index", meta_path="rag/faqs.parquet", model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        self.meta = pd.read_parquet(meta_path)

    def retrieve(self, query: str, top_k: int = 3):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            row = self.meta[self.meta['id']==int(idx)].iloc[0].to_dict()
            row['score'] = float(score)  # cosine-like
            results.append(row)
        return results
