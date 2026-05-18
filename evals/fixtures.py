"""
Static RSS article examples for the LangSmith eval dataset.

Each fixture simulates the text output of the rss_feed tool —
so we can evaluate the LLM summarizer without real network calls.

Four cases are covered intentionally:
  1. Normal tech news article   — expects full TITLE/SUMMARY/WHY_IT_MATTERS/SOURCE output
  2. Training / certification   — expects TARGET_AUDIENCE field to appear
  3. Vague / promotional        — expects "Limited information available"
  4. No content (link only)     — expects "No content preview available"
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
        "description": "Article with no content — link only, expects 'No content preview available'",
        "articles_text": (
            "1. Title: OpenAI Releases GPT-5 Technical Report\n"
            "   Link: https://openai.com/research/gpt-5-technical-report\n"
            "   Content: No content\n"
        ),
    },
]
