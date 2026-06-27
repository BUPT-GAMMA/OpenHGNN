FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/OpenHGNN

COPY constraints.txt requirements.txt pyproject.toml setup.py ./
COPY openhgnn ./openhgnn
COPY README.md README_EN.md ./

RUN pip install --upgrade pip setuptools wheel \
    && pip install -c constraints.txt -r requirements.txt \
    && pip install -e .

CMD ["openhgnn", "env"]
