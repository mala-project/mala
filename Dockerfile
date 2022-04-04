FROM continuumio/miniconda3:4.9.2
MAINTAINER Daniel Kotik <d.kotik@hzdr.de>

# Update the image to the latest packages
RUN apt-get --allow-releaseinfo-change update && apt-get upgrade -y

RUN apt-get install --no-install-recommends -y build-essential libz-dev swig git-lfs cmake
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Choose 'cpu' or 'gpu'
ARG DEVICE=cpu
COPY install/mala_${DEVICE}_environment.yml .
RUN conda env create -f mala_${DEVICE}_environment.yml && rm -rf /opt/conda/pkgs/*

# Install optional MALA dependencies into Conda environment with pip
RUN /opt/conda/envs/mala-${DEVICE}/bin/pip install --no-input --no-cache-dir pytest oapackage==2.6.8 pqkmeans

RUN echo "source activate mala-${DEVICE}" > ~/.bashrc
ENV PATH /opt/conda/envs/mala-${DEVICE}/bin:$PATH
