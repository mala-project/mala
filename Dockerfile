FROM continuumio/miniconda3:4.9.2

# Update the image to the latest packages
RUN apt-get update && apt-get upgrade -y

RUN apt-get install --no-install-recommends -y build-essential libz-dev swig git-lfs
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

COPY install/mala_cpu_environment.yml .
RUN conda env create -f mala_cpu_environment.yml && rm -rf /opt/conda/pkgs/*

RUN echo "source activate mala-cpu" > ~/.bashrc
ENV PATH /opt/conda/envs/mala-cpu/bin:$PATH
