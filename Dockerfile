FROM continuumio/miniconda3:4.9.2

# Choose 'cpu' or 'gpu'
ARG DEVICE=cpu

# Update the image to the latest packages
RUN apt-get update && apt-get upgrade -y

RUN apt-get install --no-install-recommends -y build-essential libz-dev swig git-lfs
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

COPY install/mala_${DEVICE}_environment.yml .
RUN conda env create -f mala_${DEVICE}_environment.yml && rm -rf /opt/conda/pkgs/*

RUN echo "source activate mala-${DEVICE}" > ~/.bashrc
ENV PATH /opt/conda/envs/mala-${DEVICE}/bin:$PATH
