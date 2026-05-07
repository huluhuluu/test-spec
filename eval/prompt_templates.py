from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROMPT_TEMPLATES_PATH = ROOT / "prompt_templates.json"


def load_prompt_templates() -> dict[str, Any]:
    return json.loads(PROMPT_TEMPLATES_PATH.read_text(encoding="utf-8"))


PROMPT_TEMPLATES = load_prompt_templates()


def render_template(text: str, sample: dict[str, Any]) -> str:
    choices = sample.get("choices")
    if isinstance(choices, dict):
        choice_text = "\n".join(f"{key}. {value}" for key, value in choices.items())
    else:
        choice_text = ""

    replacements = {
        "{{question}}": str(sample.get("question", "")),
        "{{subject}}": str(sample.get("subject", "")),
        "{{choices}}": choice_text,
        "{{prompt}}": str(sample.get("prompt", "")),
    }
    rendered = text
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def build_prompt_messages(dataset_name: str, sample: dict[str, Any]) -> list[dict[str, str]]:
    template = PROMPT_TEMPLATES.get(dataset_name)
    if template is None:
        raise ValueError(dataset_name)

    if dataset_name == "mtbench":
        system = template.get("system")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": sample["turns"][0]})
        return messages

    messages = []
    for message in template.get("messages", []):
        content = render_template(message["content"], sample)
        messages.append({"role": message["role"], "content": content})
    return messages
