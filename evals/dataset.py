"""
Creates (or updates) the LangSmith dataset from local fixtures.

Run once before the first eval:
    poetry run python -m evals.dataset

What happens:
  - If the dataset does not exist yet — it is created and all fixtures are uploaded.
  - If the dataset already exists — the script skips creation and prints the existing URL.
    This means re-running is safe: it will not duplicate examples.

LangSmith concepts used here:
  - Client          — the entry point for all LangSmith API calls.
  - Dataset         — a named collection of examples stored in LangSmith cloud.
  - Example         — one input/output pair inside a dataset.
"""

from dotenv import load_dotenv
from langsmith import Client
from loguru import logger

from evals.fixtures import FIXTURES

# Load .env before Client() reads LANGSMITH_API_KEY from the environment.
load_dotenv()

DATASET_NAME = "rss-digest-summarizer"


def create_dataset() -> None:
    client = Client()

    existing = [ds for ds in client.list_datasets() if ds.name == DATASET_NAME]

    if existing:
        dataset = existing[0]
        existing_count = sum(1 for _ in client.list_examples(dataset_id=dataset.id))
        if existing_count == len(FIXTURES):
            logger.info("Dataset already up to date ({} examples): {}", existing_count, dataset.url)
            return
        logger.info(
            "Fixture count changed ({} → {}). Recreating dataset...",
            existing_count, len(FIXTURES)
        )
        client.delete_dataset(dataset_id=dataset.id)

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Offline eval dataset for the RSS summarizer LLM node. "
            "Each example provides pre-fetched article text (as returned by the rss_feed tool) "
            "and tests one specific summarization case: "
            "normal article, training/cert article, vague/promo article, no-content article."
        ),
    )

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"articles_text": f["articles_text"]} for f in FIXTURES],
        outputs=[{"description": f["description"]} for f in FIXTURES],
    )

    logger.info("Dataset created with {} examples: {}", len(FIXTURES), dataset.url)


if __name__ == "__main__":
    create_dataset()
