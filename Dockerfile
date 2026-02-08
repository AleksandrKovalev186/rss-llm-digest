FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /rss-llm-digest


RUN pip install --no-cache-dir poetry==1.8.2
COPY pyproject.toml poetry.lock ./

# Установить зависимости (только production)
RUN poetry install --only=main --no-root

COPY .. /numer_bot

VOLUME ["/rss-llm-digest/bot/logs"]

ARG CI_COMMIT_TAG

ENV PYTHONPATH=/rss-llm-digest
ENV LOG_PATH=/rss-llm-digest/bot/logs/$CI_COMMIT_TAG/

CMD ["python", "bot/rss-llm-digest.py"]
