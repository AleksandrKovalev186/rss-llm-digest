FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /rss-llm-digest

RUN pip install --no-cache-dir poetry==1.8.2 && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --only=main --no-root

COPY . .

ENV PYTHONPATH=/rss-llm-digest

CMD ["python", "agent.py"]