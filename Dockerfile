FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    MAX_DATA_BYTES=1073741824 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /app
COPY requirements-runtime.lock ./
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements-runtime.lock \
    && groupadd --gid 10001 handwriting \
    && useradd --uid 10001 --gid 10001 --no-create-home handwriting \
    && mkdir /data && chown 10001:10001 /data

COPY handwriting ./handwriting
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "handwriting.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "8", "--timeout-keep-alive", "5", "--no-access-log"]
