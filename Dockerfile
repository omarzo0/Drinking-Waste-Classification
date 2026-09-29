FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLBACKEND=Agg \
    DATA_PATH=/app/data/dataset_70_15_15.npz

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: fetch the dataset if it is missing, then run the XAI comparison.
CMD ["sh", "-c", "python scripts/download_data.py && python explain.py"]
