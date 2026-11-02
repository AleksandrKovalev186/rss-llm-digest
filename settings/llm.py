import json
import os
import re
import uuid
from functools import lru_cache

import torch
from huggingface_hub import snapshot_download
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from settings.config import settings

os.environ.setdefault("USER_AGENT", "rss-llm-digest/0.1.0")

# XetHub protocol does not render tqdm bars; force standard HTTP downloads
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
PARAMETER_PATTERN = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
ROLES = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}


def _to_chat_dict(message: BaseMessage) -> dict:
    chat_dict = {"role": ROLES[message.type], "content": message.content}
    if isinstance(message, AIMessage) and message.tool_calls:
        chat_dict["tool_calls"] = [
            {"name": tc["name"], "arguments": tc["args"]} for tc in message.tool_calls
        ]
    return chat_dict


def _parse_param(value: str, schema: dict):
    # Values arrive as raw text; decode JSON only for non-string params, so "2026" stays a str.
    if schema.get("type") == "string":
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _parse_reply(text: str, tools: list[dict] | None = None) -> AIMessage:
    properties = {
        t["function"]["name"]: t["function"]["parameters"].get("properties", {})
        for t in tools or []
    }
    tool_calls = [
        {
            "name": name,
            "args": {
                k: _parse_param(v, properties.get(name, {}).get(k, {}))
                for k, v in PARAMETER_PATTERN.findall(body)
            },
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "tool_call",
        }
        for name, body in TOOL_CALL_PATTERN.findall(text)
        if name in properties
    ]
    return AIMessage(content=TOOL_CALL_PATTERN.sub("", text).strip(), tool_calls=tool_calls)


class ToolCallingChatHuggingFace(ChatHuggingFace):
    """
    ChatHuggingFace drops bound tools for a local HuggingFacePipeline and never
    parses tool calls, so agents built on it never run their tools. This passes
    the tools to the model's chat template and parses its <tool_call> blocks.

    shortcut: parses only the Qwen3.5 XML tool-call format; other model families
    answer in plain text without tool calls, add their format if HF_MODEL_ID changes.
    """

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        tools = kwargs.pop("tools", None)
        kwargs.pop("tool_choice", None)  # not enforceable on a raw pipeline; the model decides
        prompt = self.tokenizer.apply_chat_template(
            [_to_chat_dict(m) for m in messages],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        result = self.llm._generate([prompt], stop=stop, run_manager=run_manager, **kwargs)
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=_parse_reply(g.text, tools), generation_info=g.generation_info
                )
                for g in result.generations[0]
            ],
            llm_output=result.llm_output,
        )


@lru_cache(maxsize=None)
def load_chat_llm(model_id: str, max_new_tokens: int) -> ChatHuggingFace:
    logger.info("Loading LLM: {}", model_id)

    logger.info("Step 1/3 — downloading model files to cache (skip if already cached)...")
    local_path = snapshot_download(
        repo_id=model_id,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        token=settings.hf_token,
    )

    logger.info("Step 2/3 — loading tokenizer and weights into memory...")
    tokenizer = AutoTokenizer.from_pretrained(local_path, clean_up_tokenization_spaces=False)
    # On Apple Silicon "auto" sizes by free RAM at load time and may offload layers to disk.
    device_map = "mps" if torch.backends.mps.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(local_path, dtype="auto", device_map=device_map)
    # Qwen3.5 ships no generation_config.json, so the default EOS is only <|endoftext|>
    # and generation runs past the end of the chat turn; stop on <|im_end|> as well.
    default_eos = model.generation_config.eos_token_id
    default_eos = default_eos if isinstance(default_eos, list) else [default_eos]
    eos_token_ids = [
        i for i in dict.fromkeys([tokenizer.eos_token_id, *default_eos]) if i is not None
    ]
    logger.info("Step 3/3 — building pipeline...")
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        return_full_text=False,
        clean_up_tokenization_spaces=False,
        max_new_tokens=max_new_tokens,
        max_length=None,
        eos_token_id=eos_token_ids,
    )

    logger.info("LLM ready.")
    # Pass the tokenizer, otherwise ChatHuggingFace loads it from disk a second time.
    return ToolCallingChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe), tokenizer=tokenizer)


def get_chat_llm() -> ChatHuggingFace:
    return load_chat_llm(settings.hf_model_id, 2000)
