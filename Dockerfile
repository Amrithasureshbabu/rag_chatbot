FROM python:3.11-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# system deps required for building some packages and pdf processing
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       poppler-utils \
       libpoppler-cpp-dev \
       pkg-config \
       libopenblas-dev \
       git \
       curl \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install (prefer binary wheels to avoid long source builds)
COPY requirements.txt /app/requirements.txt

# create non-root user for runtime
RUN useradd -m appuser || true

# upgrade pip and install dependencies (prefer binary wheels)
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefer-binary -r /app/requirements.txt

# copy app and set ownership to non-root user
COPY --chown=appuser:appuser . /app

USER appuser

EXPOSE 8080

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080"]
FROM python:3.11-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# system deps required for building some packages and pdf processing
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       poppler-utils \
       libpoppler-cpp-dev \
       pkg-config \
       libopenblas-dev \
       git \
       curl \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install (prefer binary wheels to avoid long source builds)
COPY requirements.txt /app/requirements.txt

# create non-root user for runtime
RUN useradd -m appuser || true

# upgrade pip and install dependencies (prefer binary wheels)
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefer-binary -r /app/requirements.txt

# copy app and set ownership to non-root user
COPY --chown=appuser:appuser . /app

USER appuser

EXPOSE 8080

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8080"]
