from pathlib import Path

import yaml


def load_system_prompt() -> str:
    data = yaml.safe_load(Path("system_prompt.yaml").read_text())
    messages_data = {msg["role"]: msg["content"] for msg in data["messages"]}
    return messages_data["system"]
