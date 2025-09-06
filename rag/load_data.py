# rag/load_data.py
import pandas as pd
import pdfplumber
import re
from pathlib import Path

def load_faq(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File {path} tidak ditemukan.")
    
    # --- jika file Excel (.xls / .xlsx)
    if p.suffix.lower() in ('.xls', '.xlsx'):
        df = pd.read_excel(path)
        cols = {c.lower(): c for c in df.columns}
        qcol = next((cols[c] for c in cols if 'question' in c), None)
        acol = next((cols[c] for c in cols if 'answer' in c), None)
        if qcol is None or acol is None:
            raise ValueError("Excel harus punya kolom 'Question' dan 'Answer'.")
        df = df[[qcol, acol]].rename(columns={qcol:'question', acol:'answer'})
        df['question'] = df['question'].astype(str).str.strip()
        df['answer'] = df['answer'].astype(str).str.strip()
        df = df.reset_index().rename(columns={'index':'id'})
        return df[['id','question','answer']]
    
    # --- jika file PDF
    elif p.suffix.lower() == '.pdf':
        texts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or '')
        text = "\n".join(texts)

        # split by double newlines → cari pasangan Q/A
        parts = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        qa = []
        i = 0
        while i < len(parts):
            part = parts[i]
            if '?' in part and len(part) < 400:
                q = part.strip()
                a = parts[i+1].strip() if i+1 < len(parts) else ""
                qa.append((q, a))
                i += 2
            else:
                # fallback: pecah baris
                lines = [ln.strip() for ln in part.splitlines() if ln.strip()]
                for j in range(len(lines)-1):
                    if '?' in lines[j]:
                        qa.append((lines[j], lines[j+1]))
                i += 1
        df = pd.DataFrame(qa, columns=['question','answer'])
        df = df.reset_index().rename(columns={'index':'id'})
        return df[['id','question','answer']]
    
    else:
        raise ValueError("Format file tidak didukung. Gunakan .xlsx atau .pdf")