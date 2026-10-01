FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY . .
RUN pip install --no-cache-dir .
RUN mkdir -p data/synthetic data/processed artifacts
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s CMD python -c "import json,urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health')); assert data['models_available']"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
