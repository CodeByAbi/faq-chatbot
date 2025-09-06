# Gunakan base image yang stabil (Debian Bookworm)
FROM python:3.10-slim-bookworm

# Pastikan semua package terbaru dan install dependency dasar
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements.txt dulu (biar caching docker optimal)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir numpy==1.26.4 \
    && pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy semua source code ke container
COPY . .

# Expose port Streamlit
EXPOSE 8501

# Jalankan Streamlit (alamat 0.0.0.0 biar bisa diakses dari host)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
