FROM ubuntu:23.04

COPY poetry.lock pyproject.toml .tool-versions ./

# WORKDIR /.devcontainer/

RUN apt-get update -y && \
    apt-get install -y -q --allow-unauthenticated \
    sudo \
    make \
    curl \
    git \
    build-essential \ 
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libxml2-dev \
    libxmlsec1-dev \
    llvm \
    tk-dev \
    wget \
    xz-utils \
    zlib1g-dev

RUN NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)" && \
    (echo; echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"') >> /root/.profile && \
    eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

ENV PATH="/home/linuxbrew/.linuxbrew/bin:${PATH}"

RUN brew install asdf && \
    asdf plugin add python && \
    asdf install python 3.11.5 && \
    asdf global python 3.11.5 && \
    asdf plugin add poetry && \
    asdf install poetry 1.6.1 && \
    asdf global poetry 1.6.1

ENV PATH="${PATH}:/root/.asdf/installs/poetry/1.6.1/bin"

# ENV POETRY_HOME="/opt/poetry" \
#     POETRY_VIRTUALENVS_CREATE=true \
#     POETRY_VIRTUALENVS_IN_PROJECT=false \
#     POETRY_NO_INTERACTION=1 \
#     POETRY_VERSION=1.6.1

RUN poetry env use /root/.asdf/installs/python/3.11.5/bin/python && \
    poetry install