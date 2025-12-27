FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# copy app
COPY . /app

EXPOSE 8080

CMD [ gunicorn, -w, 1, -k, uvicorn.workers.UvicornWorker, main:app, --bind, 0.0.0.0:8080]
