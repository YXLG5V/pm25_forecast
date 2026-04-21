# ===== BUILD STAGE =====
FROM python:3.14-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libgomp1

COPY requirements_inference.txt .
RUN pip install --user --no-cache-dir -r requirements_inference.txt



# ===== RUNTIME STAGE =====
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH

COPY src/ ./src/
COPY models/ ./models/
COPY artifacts/ ./artifacts/

EXPOSE 8000

CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]