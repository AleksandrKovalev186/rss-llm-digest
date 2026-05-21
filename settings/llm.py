import os
from functools import lru_cache

from huggingface_hub import snapshot_download
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from settings.config import settings

os.environ.setdefault("USER_AGENT", "rss-llm-digest/0.1.0")

# XetHub protocol does not render tqdm bars; force standard HTTP downloads
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


@lru_cache(maxsize=1)
def get_chat_llm() -> ChatHuggingFace:
    logger.info("Loading LLM: {}", settings.hf_model_id)

    logger.info("Step 1/3 — downloading model files to cache (skip if already cached)...")
    local_path = snapshot_download(
        repo_id=settings.hf_model_id,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        token=settings.hf_token,
    )

    logger.info("Step 2/3 — loading tokenizer and weights into memory...")
    tokenizer = AutoTokenizer.from_pretrained(local_path, clean_up_tokenization_spaces=False)
    model = AutoModelForCausalLM.from_pretrained(local_path, dtype="auto", device_map="auto")
    logger.info("Step 3/3 — building pipeline...")
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=2000,
        max_length=None,
    )

    logger.info("LLM ready.")
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))
