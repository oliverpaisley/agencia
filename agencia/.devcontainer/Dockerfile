FROM python:3.8-slim

COPY poetry.lock pyproject.toml .

RUN pip install poetry; \
    poetry config virtualenvs.create false; \
    poetry install