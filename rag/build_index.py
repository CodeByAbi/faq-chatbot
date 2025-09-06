# rag/build_index.py
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pandas as pd
from rag.load_data import load_faq

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_index(data_path: str, out_dir: str = "rag", model_name: str = MODEL_NAME):
    os.makedirs(out_dir, exist_ok=True)
    df = load_faq(data_path)
    if df.empty:
        raise ValueError("Dataset FAQ kosong.")
    model = SentenceTransformer(model_name)
    texts = df['question'].tolist()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64)
    # L2 normalize embeddings (so inner-product ~ cosine)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product on normalized vectors => cosine
    index = faiss.IndexIDMap(index)
    ids = df['id'].astype('int64').to_numpy()
    index.add_with_ids(embeddings, ids)
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    df.to_parquet(os.path.join(out_dir, "faqs.parquet"), index=False)
    print("Index saved to", out_dir)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python build_index.py data/FAQ_Nawa.xlsx")
    else:
        build_index(sys.argv[1])
