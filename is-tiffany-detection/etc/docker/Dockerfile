FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS builder

RUN apt-get update && \
    apt-get install -y python3.10-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY src/ .

ENV PATH="/opt/venv/bin:$PATH"

CMD ["python", "main.py"]
# docker buildx build --platform linux/amd64 -f etc/docker/Dockerfile -t is-tiffany-detection:latest .