# evaluation/scorer.py
def cosine_to_score(similarity: float) -> int:
    # similarity diharapkan di range [-1,1] (faiss inner-product on normalized => [0,1])
    s = float(similarity)
    if s < 0:
        s = 0.0
    if s > 1.0:
        s = 1.0
    return int(round(s * 100))
