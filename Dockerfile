FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /rss-llm-digest

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY .. /numer_bot

VOLUME ["/rss-llm-digest/bot/logs"]

ARG CI_COMMIT_TAG

ENV PYTHONPATH=/rss-llm-digest
ENV LOG_PATH=/rss-llm-digest/bot/logs/$CI_COMMIT_TAG/

CMD ["python", "bot/rss-llm-digest.py"]
