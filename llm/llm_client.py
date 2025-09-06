# llm/llm_client.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict

MODEL_NAME = "google/flan-t5-base"

class LLMClient:
    def __init__(self, model_name=MODEL_NAME, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def answer(self, question: str, contexts: List[Dict], max_new_tokens: int=256) -> str:
        # contexts: list of {question, answer, score}
        snippets = "\n\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in contexts])
        prompt = (
            "Jawab pertanyaan berikut menggunakan hanya potongan FAQ yang diberikan. "
            "Jika jawabannya tidak ada di potongan FAQ, jawab 'Maaf, saya tidak menemukan jawaban yang sesuai.'\n\n"
            f"{snippets}\n\nPertanyaan: {question}\nJawaban:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4, early_stopping=True)
        answer = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return answer.strip()
