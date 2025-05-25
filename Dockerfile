ARG PYTHON_VERSION=3.11
ARG BASE_IMAGE=python:${PYTHON_VERSION}-slim-bookworm

FROM ${BASE_IMAGE}

# download and install `uv`
RUN apt update && apt install -y --no-install-recommends curl ca-certificates build-essential gcc g++ cmake vim
ADD https://astral.sh/uv/install.sh /tmp/installer.sh
RUN sh /tmp/installer.sh && mv /root/.local/bin/uv /usr/local/bin/ && rm /tmp/installer.sh

# environment variables
ENV UV_LINK_MODE="copy"
ENV USER=kserve
ENV GROUP=kserve

# create default required user `kserve`
RUN useradd ${USER} -m -u 1000 -d /home/${USER}
USER ${USER}
WORKDIR /workspace/
RUN chown -R ${USER}:${GROUP} /workspace/

# copy project configuration file and install project's dependencies
COPY --chown=${USER}:${GROUP} pyproject.toml /workspace/
RUN uv venv
RUN uv pip install -r pyproject.toml

# copy source code and install
ADD --chown=${USER}:${GROUP} . /workspace/
RUN uv sync --locked --extra storage

# add entrypoint
ENTRYPOINT [ "uv", "run", "kserve-custom-runtimes" ]
