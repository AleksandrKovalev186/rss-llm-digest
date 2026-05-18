"""
Entry point for running the RSS summarizer evaluation.

Usage:
    LANGSMITH_API_KEY=<your_key> poetry run python -m evals.run_eval

What this does:
  1. Calls evaluate() with our target function and the dataset we created in dataset.py.
  2. For each example in the dataset, evaluate() runs:
       a. target(inputs)         → gets the LLM summary
       b. evaluator_1(...)       → scores format_compliance
       c. evaluator_2(...)       → scores has_source_urls
       d. evaluator_3(...)       → scores no_large_verbatim_copy
  3. Results are uploaded to LangSmith and printed to stdout.
  4. A URL to the experiment is printed at the end — open it to see the full results.

evaluate() runs examples concurrently by default (max_concurrency=4).
We set max_concurrency=1 because our local HF model can only handle one request at a time.
"""

from dotenv import load_dotenv
from langsmith.evaluation import evaluate

from evals.dataset import DATASET_NAME
from evals.evaluators import format_compliance, has_source_urls, no_large_verbatim_copy
from evals.target import summarize

# Must be called before any LangSmith Client() instantiation (happens inside evaluate()).
load_dotenv()


def main() -> None:
    results = evaluate(
        summarize,

        data=DATASET_NAME,

        evaluators=[
            format_compliance,
            has_source_urls,
            no_large_verbatim_copy,
        ],

        experiment_prefix="summarizer-v1",

        max_concurrency=1,
    )

    print("\n=== Eval complete ===")
    for r in results:
        example = r["example"]
        eval_results = r["evaluation_results"]

        inputs = example.inputs if hasattr(example, "inputs") else example.get("inputs", {})
        preview = inputs.get("articles_text", "")[:60].replace("\n", " ")

        result_list = (
            eval_results.results
            if hasattr(eval_results, "results")
            else eval_results.get("results", [])
        )
        scores = {er.key: er.score for er in result_list}

        print(f"\nExample : {preview}...")
        for key, score in scores.items():
            print(f"  {key}: {score}")


if __name__ == "__main__":
    main()
