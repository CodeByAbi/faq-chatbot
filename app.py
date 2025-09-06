# app.py
import streamlit as st
from rag.retriever import Retriever
from llm.llm_client import LLMClient
from security.guards import detect_prompt_injection, sanitize_contexts, RateLimiter
from evaluation.scorer import cosine_to_score
import faiss, os

st.set_page_config(page_title="FAQ Chatbot (RAG)", layout="centered")

@st.cache_resource
def init_components():
    # inisialisasi retriever & llm (lambat pada pertama kali)
    retriever = Retriever(index_path="rag/faiss.index", meta_path="rag/faqs.parquet")
    llm = LLMClient()
    return retriever, llm

# init
if not os.path.exists("rag/faiss.index") or not os.path.exists("rag/faqs.parquet"):
    st.warning("FAISS index tidak ditemukan. Jalankan `python rag/build_index.py data/FAQ_Nawa.xlsx` terlebih dahulu.")
retriever, llm = init_components()

if "history" not in st.session_state:
    st.session_state.history = []
if "rate_limiter" not in st.session_state:
    st.session_state.rate_limiter = RateLimiter(max_requests=20, per_seconds=60)  # contoh 20 requests / menit

st.title("FAQ Chatbot (Nawatech) — RAG + FLAN-T5")

with st.form("query_form", clear_on_submit=True):
    q = st.text_input("Tanyakan sesuatu (bahasa Indonesia):")
    submitted = st.form_submit_button("Kirim")

if submitted:
    if not q or q.strip()=="":
        st.warning("Masukkan pertanyaan terlebih dahulu.")
    else:
        allowed = st.session_state.rate_limiter.allow()
        if not allowed:
            st.error("Terlalu banyak permintaan. Coba lagi nanti.")
        else:
            # guard prompt injection pada pertanyaan user
            inj, pat = detect_prompt_injection(q)
            if inj:
                st.error("Pertanyaan terindikasi berisi instruksi berbahaya. Diabaikan.")
            else:
                with st.spinner("Mencari jawaban..."):
                    try:
                        topk = retriever.retrieve(q, top_k=5)
                        if len(topk) == 0 or max([r['score'] for r in topk]) < 0.5:
                            st.warning("Pertanyaan yang Anda tanyakan tidak sesuai.")
                            st.session_state.history.append({
                                "question": q,
                                "answer": "Pertanyaan yang Anda tanyakan tidak sesuai.",
                                "score": 0
                        })
                        else:
                            # sanitize contexts
                            sanitized = sanitize_contexts(topk)
                            answer = llm.answer(q, sanitized)
                            score = cosine_to_score(max([r['score'] for r in topk]))
                            st.success("Jawaban:")
                            st.write(answer)
                            st.info(f"Quality score (berdasarkan cosine similarity top result): {score}/100")
                            # show sources
                            with st.expander("Tampilkan sumber (top-k)"):
                                for i, r in enumerate(topk, start=1):
                                    st.markdown(f"**{i}. Q:** {r['question']}")
                                    st.markdown(f"**A:** {r['answer']}")
                                    st.caption(f"Similarity: {r['score']:.4f}")
                            st.session_state.history.append({"question": q, "answer": answer, "score": score})
                    except Exception as e:
                        st.error(f"Terjadi error: {e}")

# history
if st.session_state.history:
    st.markdown("---")
    st.subheader("History")
    for item in reversed(st.session_state.history[-20:]):
        st.markdown(f"**Q:** {item['question']}")
        st.markdown(f"**A:** {item['answer']}")
        st.caption(f"Score: {item['score']}/100")
