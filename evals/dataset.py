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

from evals.fixtures import FIXTURES

# Load .env before Client() reads LANGSMITH_API_KEY from the environment.
load_dotenv()

DATASET_NAME = "rss-digest-summarizer"


def create_dataset() -> None:
    # Client() automatically reads LANGSMITH_API_KEY from the environment.
    # It also reads LANGCHAIN_ENDPOINT if you use a self-hosted LangSmith instance.
    client = Client()

    # Check if dataset already exists to avoid duplicating examples on re-runs.
    existing = [ds for ds in client.list_datasets() if ds.name == DATASET_NAME]

    if existing:
        dataset = existing[0]
        print(f"Dataset already exists: {dataset.url}")
        return

    # create_dataset creates a new empty dataset in LangSmith.
    # description is shown in the LangSmith UI — useful for teammates.
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Offline eval dataset for the RSS summarizer LLM node. "
            "Each example provides pre-fetched article text (as returned by the rss_feed tool) "
            "and tests one specific summarization case: "
            "normal article, training/cert article, vague/promo article, no-content article."
        ),
    )

    # create_examples uploads all fixtures as examples in one API call (batch).
    # inputs  — what the target function receives.
    # outputs — the reference answer. We leave it None because our evaluators
    #            are reference-free (they check format and rules, not exact match).
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"articles_text": f["articles_text"]} for f in FIXTURES],
        outputs=[{"description": f["description"]} for f in FIXTURES],
    )

    print(f"Dataset created with {len(FIXTURES)} examples: {dataset.url}")


if __name__ == "__main__":
    create_dataset()
