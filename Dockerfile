FROM python:3.10.14-bullseye

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && \
    apt install -y build-essential cmake lsb-release

WORKDIR /src
COPY gt4sd-molformer .

RUN pip install -r requirements_torch.txt
RUN pip install -r requirements.txt
RUN pip install -e .

# CMD [ "executable" ]