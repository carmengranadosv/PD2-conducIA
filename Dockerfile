FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV KERAS_BACKEND="jax"
ENV XLA_PYTHON_CLIENT_PREALLOCATE="false"
ENV XLA_PYTHON_CLIENT_ALLOC_FRACTION=".10"

WORKDIR /app

# Instalamos solo lo esencial para compilar
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del proyecto
COPY . .

EXPOSE 8000

CMD ["uvicorn", "despliegue.main:app", "--host", "0.0.0.0", "--port", "8000"]