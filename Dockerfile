# Use NVIDIA's CUDA 11.8 base image with UBI 8 (Red Hat Universal Base Image)
FROM nvidia/cuda:11.8.0-runtime-ubi8

# Install required dependencies
RUN dnf update -y && \
    dnf install -y \
    gcc \
    # g++ needed for pytorch-fast-transformers build
    gcc-c++ \
    openssl-devel \
    bzip2-devel \
    libffi-devel \
    zlib-devel \
    wget \
    make

# Download and install Python 3.10
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar xzf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations && \
    make altinstall

# Set Python 3.10 as the default python version
RUN ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3

# Clean up
RUN dnf clean all && \
    rm -rf Python-3.10.0 Python-3.10.0.tgz

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set up a working directory
WORKDIR /workspace

# Copy pyproject.toml to the working directory (install before copying source)
COPY pyproject.toml .

# Install additional dependencies
RUN python3 -m pip install poetry
RUN poetry install --no-cache --no-interaction --no-root
RUN poetry run pip install pytorch-fast-transformers --no-build-isolation --no-cache-dir

# Copy application code
COPY . .
RUN poetry install

# Set default command
CMD ["poetry", "run", "python", "molformer_inference/serve.py"]
