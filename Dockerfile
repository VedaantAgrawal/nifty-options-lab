FROM python:3.13-slim

WORKDIR /app

# curl for the HEALTHCHECK below; build-essential in case any pinned wheel
# (scipy/scikit-learn/pyarrow) needs to build from source on this platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the app + engine/data/model code -- no tests, scripts, or docs.
COPY app.py .
COPY data/ ./data/
COPY engine/ ./engine/
COPY models/ ./models/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
