"""
Static RSS article examples for the LangSmith eval dataset.

Each fixture simulates the text output of the rss_feed tool —
so we can evaluate the LLM summarizer without real network calls.

Five cases are covered intentionally:
  1. Normal tech news article   — expects full TITLE/SUMMARY/WHY_IT_MATTERS/SOURCE output
  2. Training / certification   — expects TARGET_AUDIENCE field to appear
  3. Vague / promotional        — expects "Limited information available"
  4. No content (link only)     — expects "No content preview available"
  5. Multi-article (real feeds) — expects separate summary per article, no repeated sentences
"""

# Each fixture maps to one LangSmith Example.
# Key "articles_text" is what the target function receives as input.
# Key "description" is only for humans reading this file — not sent to LangSmith.

FIXTURES = [
    {
        "description": "Normal tech news — new Python release",
        "articles_text": (
            "1. Title: Python 3.14 Released with Experimental Free-Threaded Mode\n"
            "   Link: https://python.org/news/python-314-release\n"
            "   Content: The Python Software Foundation has released Python 3.14, "
            "featuring experimental support for free-threaded execution (PEP 703). "
            "The GIL can now be disabled at runtime with PYTHON_GIL=0. "
            "Performance benchmarks show up to 40% speedup on CPU-bound parallel workloads. "
            "The release also includes a new template string syntax (PEP 750) and "
            "improved error messages for type mismatches.\n"
        ),
    },
    {
        "description": "Training / certification article — expects TARGET_AUDIENCE field",
        "articles_text": (
            "1. Title: Google Launches Free Generative AI Course for Beginners\n"
            "   Link: https://cloud.google.com/blog/generative-ai-course-2024\n"
            "   Content: Google Cloud has released a free 8-hour course on Generative AI "
            "fundamentals via its Skills Boost platform. The course covers prompt engineering, "
            "embeddings, and fine-tuning basics. No prior ML experience is required. "
            "Participants who pass the final quiz receive a shareable digital badge. "
            "The course is available in English, Spanish, and Portuguese.\n"
        ),
    },
    {
        "description": "Vague / promotional article — expects 'Limited information available'",
        "articles_text": (
            "1. Title: Exciting Things Are Coming to Our Platform!\n"
            "   Link: https://somestartup.io/blog/exciting-announcement\n"
            "   Content: We are thrilled to announce that big things are on the way. "
            "Stay tuned for more updates from our amazing team. "
            "We can't wait to share what we've been working on. Sign up for our newsletter!\n"
        ),
    },
    {
        "description": "Article with no content — link only, expects 'No content preview available'",  # noqa: E501
        "articles_text": (
            "1. Title: OpenAI Releases GPT-5 Technical Report\n"
            "   Link: https://openai.com/research/gpt-5-technical-report\n"
            "   Content: No content\n"
        ),
    },
    {
        "description": (
            "Multi-article from real feeds — expects separate summaries per article, "
            "no repeated sentences, plain-text field names"
        ),
        "articles_text": (
            "1. Title: AWS Announces General Availability of Amazon Bedrock Guardrails\n"
            "   Link: https://aws.amazon.com/blogs/aws/amazon-bedrock-guardrails-now-ga\n"
            "   Content: Amazon Web Services announced the general availability of Amazon "
            "Bedrock Guardrails, a set of safeguards for generative AI applications. "
            "Guardrails can filter harmful content, redact PII, and block prompt injection "
            "attacks. Customers can configure up to 100 topic denial policies per guardrail. "
            "Pricing is based on the number of text units processed.\n"
            "\n"
            "2. Title: Show HN: I built a terminal-based Git client in Rust\n"
            "   Link: https://news.ycombinator.com/item?id=39912345\n"
            "   Content: A developer released Gitui 0.26, a blazing-fast terminal UI for Git "
            "written in Rust. The tool supports staging, committing, branching, and diff viewing "
            "without leaving the terminal. It uses the crossterm library for cross-platform "
            "rendering. The project has 17k stars on GitHub and is available via cargo install.\n"
            "\n"
            "3. Title: Python 3.13.3 maintenance release\n"
            "   Link: https://blog.python.org/2024/04/python-3133-maintenance-release\n"
            "   Content: Python 3.13.3 has been released as a maintenance update. "
            "It includes 30 bug fixes across the interpreter, standard library, and documentation. "
            "Notable fixes address a regression in asyncio task cancellation and a memory leak "
            "in the sqlite3 module. Users on 3.13.x are encouraged to upgrade.\n"
            "\n"
            "4. Title: PyTorch 2.3 Released with FlexAttention and AOTInductor\n"
            "   Link: https://pytorch.org/blog/pytorch-2-3-release\n"
            "   Content: PyTorch 2.3 introduces FlexAttention, a flexible API for writing "
            "custom attention kernels that compiles with torch.compile. AOTInductor is now "
            "stable, enabling ahead-of-time compilation for deployment without a Python runtime. "
            "The release also improves torch.compile cold-start time by 2x on large models.\n"
        ),
    },
]
