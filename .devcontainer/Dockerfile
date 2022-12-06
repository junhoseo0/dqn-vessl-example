FROM mambaorg/micromamba:bionic-cuda-11.6.2
COPY --chown=$MAMBA_USER:$MAMBA_USER .devcontainer/env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
