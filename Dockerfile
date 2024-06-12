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
    pip install -r install/requirements.txt --no-cache --no-cache-dir && \
    rm -rf install

# generate definitions
# RUN ["python", "/src/molformer_inference/generate_property_service_defs.py"]

# set permissions for OpenShift
# from https://docs.openshift.com/container-platform/4.5/openshift_images/create-images.html#images-create-guide-general_create-images
RUN mkdir -p ./oracle /.config /.cache /.gt4sd /.paccmann && \
    chgrp -R 0 ./oracle /.config /.cache /.gt4sd /.paccmann && \
    chmod -R g=u ./oracle /.config /.cache /.gt4sd /.paccmann
# excluding the .venv directory from recursive permissions
RUN find /src -path /src/.venv -prune -o -print | xargs chgrp 0 && \
    find /src -path /src/.venv -prune -o -exec chmod g=u {} +

EXPOSE 8080 80

CMD ["python", "/src/molformer_inference/service.py"]