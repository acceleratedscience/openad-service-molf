FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# update and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    libsm6 libxext6 libxrender-dev curl git \
    && rm -rf /var/lib/apt/lists/*

# install python
RUN add-apt-repository ppa:deadsnakes/ppa &&  \
    apt-get install -y build-essential python3.10 python3.10-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN update-alternatives --install /usr/local/bin/python python \
    /usr/bin/python3.10 10

# Install deps
WORKDIR /src
COPY . .

# RUN pip install setuptools
RUN pip install virtualenv
RUN virtualenv .venv
ENV PATH="/src/.venv/bin:$PATH"
RUN cd install && ./install.sh
RUN pip install -e .

# generate definitions
RUN ["/src/.venv/bin/python", "/src/molformer_inference/generate_property_service_defs.py"]

EXPOSE 8080 80

CMD ["/src/.venv/bin/python", "/src/molformer_inference/service.py"]