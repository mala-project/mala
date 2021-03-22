FROM continuumio/miniconda3:4.9.2

# Update the image to the latest packages
RUN apt-get update && apt-get upgrade -y

RUN apt-get install --no-install-recommends -y build-essential \
                                               libz-dev swig git-lfs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY install/fesl_cpu_environment.yml .
RUN conda env create -q -f fesl_cpu_environment.yml && rm -rf /opt/conda/pkgs/*

RUN echo "source activate fesl-cpu" > ~/.bashrc
ENV PATH /opt/conda/envs/fesl-cpu/bin:$PATH
