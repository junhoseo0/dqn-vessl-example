FROM mambaorg/micromamba:bionic-cuda-11.6.2

# Fixes for VESSL integration
ENV PATH=/opt/conda/bin:$PATH

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get -y install --no-install-recommends \
        git \
        openssh-server \
        vim \
    && apt-get clean

USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
