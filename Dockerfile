# Base: NVIDIA NGC TensorFlow container (CUDA & cuDNN baked in)
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# System packages you actually use (xxd for C-array conversion, git optional)
RUN apt-get update && \
    apt-get install -y --no-install-recommends xxd git && \
    rm -rf /var/lib/apt/lists/*

# Headless matplotlib by default (your code should NOT force TkAgg)
ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy requirements first to leverage Docker layer cache
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# default command
ENV APP_CMD="python vocalization_classifier/main.py"
CMD bash -lc "$APP_CMD"
