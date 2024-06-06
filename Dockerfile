FROM continuumio/miniconda3:22.11.1
LABEL maintainer="d.kotik@hzdr.de"

# Update the image and install essential packages
RUN apt-get --allow-releaseinfo-change update && apt-get upgrade -y && \
    apt-get install --no-install-recommends -y \
    build-essential \
    libz-dev \
    swig \
    unzip \
    wget \
    cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Choose 'cpu' or 'gpu'
ARG DEVICE=cpu
COPY install/mala_${DEVICE}_environment.yml .
RUN conda env create -f mala_${DEVICE}_environment.yml && rm -rf /opt/conda/pkgs/*

# Install optional MALA dependencies into Conda environment with pip
RUN /opt/conda/envs/mala-${DEVICE}/bin/pip install --no-input --no-cache-dir \
    pytest \
    oapackage==2.6.8 \
    pqkmeans

RUN echo "source activate mala-${DEVICE}" > ~/.bashrc
ENV PATH /opt/conda/envs/mala-${DEVICE}/bin:$PATH

