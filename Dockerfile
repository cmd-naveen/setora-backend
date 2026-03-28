FROM python:3.11-slim-bookworm

# Install system deps: libgomp1 (LightGBM OpenMP), libgomp is required at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Expose port (Railway sets $PORT)
ENV PORT=8000
EXPOSE $PORT

CMD cd backend && python -m uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
