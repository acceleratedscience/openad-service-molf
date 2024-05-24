FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# update and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Install deps
WORKDIR /src
COPY . .

RUN \
    pip install -r install/requirements.txt && \
    rm -rf install

# generate definitions
RUN ["python", "/src/molformer_inference/generate_property_service_defs.py"]

EXPOSE 8080 80

CMD ["python", "/src/molformer_inference/service.py"]