# security/guards.py
import re
import time

SUSPICIOUS_PATTERNS = [
    r'ignore (previous|all|above) instructions',
    r'do not follow',
    r'follow these instructions',
    r'expose secret',
    r'api key',
    r'password',
]

def detect_prompt_injection(text: str):
    t = text.lower()
    for p in SUSPICIOUS_PATTERNS:
        if re.search(p, t):
            return True, p
    return False, None

def sanitize_contexts(contexts):
    sanitized = []
    for c in contexts:
        # buang baris instruksi eksplisit
        filtered_lines = [ln for ln in c['answer'].splitlines() if not re.search(r'(ignore|do not|follow these|password|api key)', ln.lower())]
        c2 = c.copy()
        c2['answer'] = "\n".join(filtered_lines)
        sanitized.append(c2)
    return sanitized

class RateLimiter:
    def __init__(self, max_requests=30, per_seconds=60):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.times = []

    def allow(self):
        now = time.time()
        # keep only recent
        self.times = [t for t in self.times if now - t < self.per_seconds]
        if len(self.times) < self.max_requests:
            self.times.append(now)
            return True
        return False
