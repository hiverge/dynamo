FROM python:3.11

# install git, pip, wget dependencies
RUN apt-get update -y && apt-get install -y git python3-pip wget tmux && rm -rf /var/lib/apt/lists/*

# Rust Setup
RUN apt-get update && apt-get install -y \
    curl \
    build-essential ca-certificates \
    libhwloc-dev libudev-dev pkg-config libclang-dev \
    protobuf-compiler python3-dev cmake \
 && rm -rf /var/lib/apt/lists/*
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustc --version && cargo --version

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:${PATH}"

WORKDIR /app/repo

RUN pip install maturin

# Context of docker build should be root of dynamo repo
COPY . /app/repo

ENV DYNAMO_HOME=/app/repo

# Initial compile
RUN uv venv .venv && \
    cd lib/bindings/python && \
    maturin develop --release --uv --strip && \
    cd /app/repo && \
    uv pip install -e lib/gpu_memory_service && \
    uv pip install -e .