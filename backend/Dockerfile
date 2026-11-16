FROM python:3.11-slim

WORKDIR /workspace

# Install essential runtime and rendering dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt /workspace/backend/requirements.txt
RUN pip install --no-cache-dir -r /workspace/backend/requirements.txt

# Copy code and default directories
COPY backend /workspace/backend
COPY data /workspace/data
COPY artifacts /workspace/artifacts
COPY training /workspace/training

ENV PYTHONPATH=/workspace:/workspace/backend
EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
